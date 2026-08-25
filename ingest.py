import glob
import os
import re
import html
import zipfile

import chromadb
from pypdf import PdfReader

from config import CHROMA_DIR, DOCS_DIR
from embedding import OllamaEmbedding


def _extract_runs(block: str) -> str:
    """从一段 XML 块中提取可见文本（<w:t> 内容），保留 <w:tab>/<w:br> 为空白。"""
    block = block.replace("<w:tab/>", "\t").replace("<w:br/>", "\n").replace("<w:cr/>", "\n")
    texts = re.findall(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", block, re.DOTALL)
    return html.unescape("".join(texts))


# Word 标题样式 pStyle val -> Markdown 标题层级（无映射则不视为标题）
_HEADING_STYLE = {"1": 1, "2": 2, "3": 3, "4": 4}


def docx_to_text(path: str) -> str:
    """用内置 zipfile 解析 .docx（word/document.xml），零额外依赖。

    修复点：
    1. 按 <w:p> 段落边界提取文本并补换行（旧实现只抽 <w:t> 内部文本，
       会把整篇文档压成无换行的单行）。
    2. 识别 Word 标题样式（pStyle val 1-4），转换为 Markdown 的 #~####，
       使下游章节切分能正确按标题层级拆分。
    """
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    out = []
    for p in re.split(r"</w:p>", xml):
        # 该段落的标题层级（若有 pStyle 且命中映射）
        m_style = re.search(r'<w:pStyle w:val="(\d+)"', p)
        level = _HEADING_STYLE.get(m_style.group(1)) if m_style else None
        text = _extract_runs(p).strip()
        if not text:
            continue
        if level:
            out.append("#" * level + " " + text)
        else:
            out.append(text)
    # 表格行补换行分隔（不破坏已生成的标题行）
    raw = "\n".join(out)
    raw = re.sub(r"(?<=.)\s*(</w:tr>)\s*", "\n", raw)
    return re.sub(r"\n{3,}", "\n\n", raw)


def load_documents(docs_dir: str):
    """递归加载目录下所有 .md/.txt/.pdf/.docx 文档。"""
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
            else:
                print(f"跳过不支持的文件: {path}")
        except Exception as e:
            print(f"读取失败 {path}: {e}")
    return loaded


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100):
    """按字符长度切分文本，带重叠以避免切断语义。"""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
        if start <= 0:
            break
    return chunks


def main():
    if not os.path.isdir(DOCS_DIR):
        print(f"文档目录不存在: {DOCS_DIR}，请先创建并放入技术文档。")
        return

    print("正在加载文档...")
    docs = load_documents(DOCS_DIR)
    if not docs:
        print("没有找到任何可索引的文档（支持 .md/.txt/.pdf/.docx）。")
        return

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # 每次重建索引，保证与文档目录一致
    try:
        client.delete_collection("docs")
    except Exception:
        pass
    collection = client.get_or_create_collection("docs", embedding_function=OllamaEmbedding())

    ids, documents, metadatas = [], [], []
    for path, text in docs:
        for i, chunk in enumerate(split_text(text)):
            ids.append(f"{os.path.basename(path)}-{i}")
            documents.append(chunk)
            metadatas.append({"source": path})

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"索引完成：共写入 {len(documents)} 个片段，来源文件 {len(docs)} 个。")


if __name__ == "__main__":
    main()
