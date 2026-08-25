"""按标题层级把技术文档拆成「章节」，用于分步骤教学。

相比 ingest.py 的固定字符切分，这里保留文档的语义结构，
更适合"按章讲解 + 按章出题"的教学场景。
"""

import glob
import io
import os
import re

import chromadb
from pypdf import PdfReader

from config import CHROMA_DIR, DOCS_DIR
from embedding import OllamaEmbedding
from ingest import docx_to_text  # 复用 ingest.py 的修复版 docx 解析


def load_documents(docs_dir: str):
    loaded = []
    for path in glob.glob(os.path.join(docs_dir, "**", "*"), recursive=True):
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in (".md", ".txt", ".text"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    loaded.append((path, f.read()))
            elif ext == ".pdf":
                reader = PdfReader(path)
                content = "\n".join((p.extract_text() or "") for p in reader.pages)
                loaded.append((path, content))
            elif ext == ".docx":
                loaded.append((path, docx_to_text(path)))
        except Exception as e:
            print(f"读取失败 {path}: {e}")
    return loaded


# ---------- 按标题层级拆章节 ----------

# Markdown 标题：# 一级、## 二级 ... 最多到 ####
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
# 数字编号标题：1. / 1.2. / 4.3.4. / 1 Introduction 等（学术论文/手册常见）
# 分组1=编号，分组2=标题文本（必须是"名词短语"，不含公式/句子符号）
_NUM_HEADING_RE = re.compile(
    r"^(?:(\d+(?:\.\d+)*)[\.、) \t]+)([A-Za-z][^=\n]{2,90})$", re.MULTILINE
)
# 标题中出现这些字符 → 判定为公式/句子，不是标题
_FORMULA_CHARS = set(";∉∘∈𝜕Γ𝔗𝔈𝔐𝛿≃≈⋄∪−⊆𝛾𝛿′∗")


_SENTENCE_STARTERS = {
    "the", "a", "an", "for", "every", "each", "all", "we", "this", "that",
    "these", "those", "if", "when", "then", "by", "with", "from", "since",
    "note", "proof", "assume", "given", "let", "such", "there", "where",
    "which", "whose", "our", "their", "its",
}


def _looks_like_title(title: str) -> bool:
    """启发式判断一行是否像真正的章节标题（排除数学公式/长句/枚举项/正文句子）。"""
    t = title.strip()
    if not t:
        return False
    # 长度：过短(纯符号)或过长(句子)都排除
    if len(t) < 4 or len(t) > 60:
        return False
    # 必须大写字母开头（章节标题规范）
    if not t[0].isupper():
        return False
    # 以小写句子引导词开头（且非专有名词）→ 排除（大概率是正文句子）
    if t[0].isupper() and t[:1]:
        first_word = t.split()[0].strip("()")
        if first_word.lower() in _SENTENCE_STARTERS and not t.split()[0].istitle():
            return False
    # 含公式/特殊数学字符 → 排除
    if any(c in _FORMULA_CHARS for c in t):
        return False
    # 含等号/分号/圆括号(常见于公式/句子) → 排除（标题一般不用）
    for ch in ("=", ";", "(", ")", "∈", "⊆"):
        if ch in t:
            return False
    return True


def _detect_headings(text: str):
    """返回 [(start, end, level, title), ...] 的标题候选，按位置排序、Markdown 优先。"""
    md = [(m.start(), m.end(), len(m.group(1)), m.group(2).strip())
          for m in _HEADING_RE.finditer(text)]
    # 数字标题：层级 = 编号中点的个数+1
    num = []
    for m in _NUM_HEADING_RE.finditer(text):
        num_part, title = m.group(1), m.group(2).strip()
        if not _looks_like_title(title):
            continue
        level = num_part.count(".") + 1
        num.append((m.start(), m.end(), level, f"{num_part}. {title}"))
    return sorted(md + num, key=lambda x: x[0])


def split_into_chapters(text: str, min_len: int = 120):
    """把单篇文档拆成带层级的章节列表。

    支持标题格式：
    - Markdown：#、##、###、####
    - 数字编号：1. / 1.2. / 4.3.4. / 1 Introduction（学术论文/手册常见）
    - 无标题纯文本：归为「概述」一章

    返回: [{"title", "level", "content", "source"}, ...]
    """
    headings = _detect_headings(text)
    chapters = []
    if not headings:
        if text.strip():
            chapters.append({"title": "概述", "level": 1, "content": text.strip(), "source": ""})
        return chapters

    for i, (start, _, level, title) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        content = text[start:end].strip()
        if len(content) < min_len:
            # 太短的章节内容并入上一章，避免碎片化出题；若无上一章则跳过
            if chapters:
                chapters[-1]["content"] += f"\n\n{title}\n{content}"
            continue
        chapters.append({"title": title, "level": level, "content": content, "source": ""})

    # 过滤空章节：避免把空内容丢给模型导致编造（幻觉）题目
    chapters = [ch for ch in chapters if len(ch["content"].strip()) >= min_len]

    return chapters


def build_course(docs_dir: str = DOCS_DIR):
    """加载全部文档并拆成章节，返回扁平化的章节列表。"""
    raw = load_documents(docs_dir)
    course = []
    for path, text in raw:
        for ch in split_into_chapters(text):
            ch["source"] = os.path.basename(path)
            course.append(ch)
    return course


def parse_uploaded(file_bytes: bytes, filename: str) -> str:
    """解析 Streamlit 上传的文档字节为纯文本（支持 md/txt/pdf/docx）。"""
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".md", ".txt", ".text"):
        return file_bytes.decode("utf-8", errors="ignore")
    if ext == ".pdf":
        import io
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if ext == ".docx":
        return docx_to_text_bytes(file_bytes)
    raise ValueError(f"不支持的文件类型：{ext or filename}")


def docx_to_text_bytes(file_bytes: bytes) -> str:
    """从上传的 docx 字节解析文本（复用 ingest.py 的结构化解析逻辑）。"""
    import html
    import zipfile
    from ingest import _HEADING_STYLE, _extract_runs

    with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    out = []
    for p in re.split(r"</w:p>", xml):
        m_style = re.search(r'<w:pStyle w:val="(\d+)"', p)
        level = _HEADING_STYLE.get(m_style.group(1)) if m_style else None
        text = _extract_runs(p).strip()
        if not text:
            continue
        if level:
            out.append("#" * level + " " + text)
        else:
            out.append(text)
    raw = "\n".join(out)
    raw = re.sub(r"(?<=.)\s*(</w:tr>)\s*", "\n", raw)
    return re.sub(r"\n{3,}", "\n\n", raw)


if __name__ == "__main__":
    course = build_course()
    print(f"共解析出 {len(course)} 个章节：")
    for i, ch in enumerate(course):
        print(f"  {i+1}. [L{ch['level']}] {ch['title']}  ({len(ch['content'])} 字, 来源 {ch['source']})")
