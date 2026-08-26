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
# 数字编号标题：1. / 1.2. / 4.3.4. / 1 Introduction / 第一章 / 第1章 等
# 分组1=编号，分组2=标题文本
_NUM_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:(?:第\s*)?(\d+(?:\.\d+)*)\s*[.、)章\s\t]*)"  # 1.  / 1.2  / 第1章  / 1章
    r"|(?:第\s*[一二三四五六七八九十百\d]+\s*章\s*)"      # 第一章 / 第十章
    r")"
    r"([^\n=;，。．]{2,60})$", re.MULTILINE
)
# 标题中出现这些字符 → 判定为公式/句子，不是标题
_FORMULA_CHARS = set(";∉∘∈𝜕Γ𝔗𝔈𝔐𝛿≃≈⋄∪−⊆𝛾𝛿′∗=+×÷±≤≥≠→←⇒⇔∫∑∏∂∇")


_SENTENCE_STARTERS = {
    "the", "a", "an", "for", "every", "each", "all", "we", "this", "that",
    "these", "those", "if", "when", "then", "by", "with", "from", "since",
    "note", "proof", "assume", "given", "let", "such", "there", "where",
    "which", "whose", "our", "their", "its",
    # 中文句子引导词
    "因此", "所以", "但是", "然而", "根据", "由于", "通过", "对于", "关于",
    "本文", "本节", "我们", "这种", "该", "这个", "这些", "一般", "通常",
}


def _is_cjk(text: str) -> bool:
    """是否以中日韩字符开头（中文标题不以大写字母开头，需单独处理）。"""
    return bool(re.search(r"[\u4e00-\u9fff]", text[:1]))


def _looks_like_title(title: str) -> bool:
    """启发式判断一行是否像真正的章节标题（排除数学公式/长句/枚举项/正文句子）。

    同时支持中文与英文标题：中文标题不以大写字母开头，用长度/标点/句式启发式判断。
    """
    t = title.strip()
    if not t:
        return False
    # 长度：过短(纯符号)或过长(句子)都排除
    if len(t) < 2 or len(t) > 60:
        return False
    # 以句子结束标点结尾（。．.!?）的，大概率是正文句子 → 排除
    if t[-1] in "。．.!?！？；;":
        return False
    # 以句子引导词开头（且非专有名词）→ 排除（大概率是正文句子）
    first_word = t.split()[0].strip("()“”\"")
    if first_word.lower() in _SENTENCE_STARTERS:
        return False
    # 含公式/特殊数学字符 → 排除
    if any(c in _FORMULA_CHARS for c in t):
        return False
    # 中文标题：允许中文开头，只要不是上述排除项即视为候选
    if _is_cjk(t):
        return True
    # 英文标题：要求首字母大写（章节标题规范）
    if t[0].isalpha() and not t[0].isupper():
        return False
    return True


def _detect_headings(text: str):
    """返回 [(start, end, level, title), ...] 的标题候选，按位置排序、Markdown 优先。

    支持：Markdown 标题、英文数字编号（1. / 1.2.）、中文编号（第一章 / 第1章）。
    """
    md = [(m.start(), m.end(), len(m.group(1)), m.group(2).strip())
          for m in _HEADING_RE.finditer(text)]
    # 数字标题：层级 = 编号中点的个数+1；"第X章"按出现顺序给 level=1
    num = []
    for m in _NUM_HEADING_RE.finditer(text):
        num_part, title = m.group(1), m.group(2).strip()
        if not _looks_like_title(title):
            continue
        if num_part and num_part[0].isdigit():
            level = num_part.count(".") + 1
        else:
            level = 1
        prefix = (num_part + " ") if num_part else ""
        num.append((m.start(), m.end(), level, f"{prefix}{title}"))
    return sorted(md + num, key=lambda x: x[0])


def split_into_chapters(text: str, min_len: int = 120, max_chapters: int = 80):
    """把单篇文档拆成带层级的章节列表。

    支持标题格式：
    - Markdown：#、##、###、####
    - 数字编号：1. / 1.2. / 4.3.4. / 1 Introduction（学术论文/手册常见）
    - 无标题纯文本：归为「概述」一章

    对超大文档的保护：
    - 单章过长（> SINGLE_CHAPTER_MAX 字）会再按段落均分为更小的子章，
      避免把整篇几万字直接喂给模型导致超时/幻觉。
    - 章节总数超过 max_chapters 时，后面的内容截断丢弃，防止逐章
      LLM 生成因章节过多而无休止地转圈（看起来像卡死）。

    返回: [{"title", "level", "content", "source"}, ...]
    """
    SINGLE_CHAPTER_MAX = 4000  # 单章内容上限，超出则再切分

    def _split_long(title, content, level):
        """把过长的一章切成多个不超过上限的子章。

        优先按空行分段；若单个段落本身就超过上限，则退化为按固定字数硬切，
        保证任何超长文本都不会整段喂给模型。
        """
        if len(content) <= SINGLE_CHAPTER_MAX:
            return [{"title": title, "level": level, "content": content, "source": ""}]

        # 先按空行分段
        parts = [p for p in re.split(r"\n\s*\n", content) if p.strip()]
        # 若分段后仍有超长段落，则把超长段按固定字数再切
        safe = []
        for p in parts:
            if len(p) <= SINGLE_CHAPTER_MAX:
                safe.append(p)
            else:
                for i in range(0, len(p), SINGLE_CHAPTER_MAX):
                    safe.append(p[i:i + SINGLE_CHAPTER_MAX])

        subs, buf, idx = [], "", 1
        for part in safe:
            if buf and len(buf) + len(part) + 2 > SINGLE_CHAPTER_MAX:
                subs.append({"title": f"{title}（{idx}）", "level": level,
                             "content": buf.strip(), "source": ""})
                idx += 1
                buf = part
            else:
                buf = (buf + "\n\n" + part).strip() if buf else part
        if buf.strip():
            subs.append({"title": f"{title}（{idx}）", "level": level,
                         "content": buf.strip(), "source": ""})
        return subs

    headings = _detect_headings(text)
    chapters = []
    if not headings:
        if text.strip():
            chapters.extend(_split_long("概述", text.strip(), 1))
        return _finalize(chapters, min_len, max_chapters)[0]

    # 第一个标题之前的内容（前言/封面）也保留为一章，避免丢失
    pre = text[: headings[0][0]].strip()
    if pre:
        chapters.extend(_split_long("前言", pre, 1))

    for i, (start, _, level, title) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        content = text[start:end].strip()
        if len(content) < min_len:
            # 太短的章节内容并入上一章，避免碎片化出题；若无上一章则跳过
            if chapters:
                chapters[-1]["content"] += f"\n\n{title}\n{content}"
            continue
        chapters.extend(_split_long(title, content, level))

    return _finalize(chapters, min_len, max_chapters)[0]


def _finalize(chapters, min_len, max_chapters):
    """过滤空章节 + 章节总数上限保护（两分支共用）。

    返回 (chapters, truncated)：truncated=True 表示因超出章节上限而截断，
    多余内容被丢弃（不再无限合并成巨章，避免卡死）。
    """
    # 过滤空章节：避免把空内容丢给模型导致编造（幻觉）题目
    chapters = [ch for ch in chapters if len(ch["content"].strip()) >= min_len]

    # 最终兜底：任何仍超过上限的章节强制按固定字数再切，确保单章可控
    SINGLE_CHAPTER_MAX = 4000
    safe = []
    for ch in chapters:
        if len(ch["content"]) <= SINGLE_CHAPTER_MAX:
            safe.append(ch)
            continue
        n = (len(ch["content"]) + SINGLE_CHAPTER_MAX - 1) // SINGLE_CHAPTER_MAX
        for i in range(n):
            seg = ch["content"][i * SINGLE_CHAPTER_MAX:(i + 1) * SINGLE_CHAPTER_MAX]
            safe.append({"title": f"{ch['title']}（{i+1}/{n}）", "level": ch["level"],
                         "content": seg, "source": ch["source"]})
    chapters = safe

    truncated = False
    # 总章节数保护：超出上限则直接截断（不合并成巨章），多余内容丢弃
    if len(chapters) > max_chapters:
        truncated = True
        chapters = chapters[:max_chapters]

    return chapters, truncated


def build_course(docs_dir: str = DOCS_DIR):
    """加载全部文档并拆成章节，返回扁平化的章节列表。"""
    raw = load_documents(docs_dir)
    course = []
    for path, text in raw:
        for ch in split_into_chapters(text):
            ch["source"] = os.path.basename(path)
            course.append(ch)
    return course


# PDF 解析保护：页数上限，超出则截断，避免超大/扫描版 PDF 卡死
PDF_MAX_PAGES = 300
# 单页文本提取硬超时（秒），某些损坏/扫描版 PDF 单页可能无限期挂起
PDF_PAGE_TIMEOUT = 8


def _pdf_to_text(file_bytes: bytes) -> str:
    import io

    def _extract():
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = reader.pages[:PDF_MAX_PAGES]
        out = []
        for p in pages:
            try:
                out.append(p.extract_text() or "")
            except Exception:
                out.append("")  # 单页失败不影响整体
        return "\n".join(out)

    # 用线程包裹，给整体解析一个硬超时，防止 PdfReader 内部挂死
    import threading

    result, err = {}, [None]

    def _run():
        try:
            result["v"] = _extract()
        except Exception as e:  # noqa
            err[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(PDF_PAGE_TIMEOUT * (PDF_MAX_PAGES // 30 + 1))  # 随页数放大上限
    if t.is_alive():
        raise TimeoutError("PDF 解析超时（可能文件过大或为扫描版图片，无可用文本层）。")
    if err[0] is not None:
        raise err[0]
    return result.get("v", "")


def parse_uploaded(file_bytes: bytes, filename: str) -> str:
    """解析 Streamlit 上传的文档字节为纯文本（支持 md/txt/pdf/docx）。

    对大文件/扫描版 PDF 加页数上限与硬超时保护，避免 UI 卡死。
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".md", ".txt", ".text"):
        return file_bytes.decode("utf-8", errors="ignore")
    if ext == ".pdf":
        if len(file_bytes) > 60 * 1024 * 1024:  # 60MB 以上直接拒绝
            raise ValueError("PDF 超过 60MB，请先压缩或拆分成更小的文档。")
        return _pdf_to_text(file_bytes)
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
