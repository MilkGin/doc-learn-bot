"""答疑模块：学员在学习中随时提问，基于文档内容回答。

两种检索源：
1. ask_from_chapters(question, chapters) —— 基于当前会话已解析的章节内容（推荐，支持上传文档）
2. ask_doc(question) —— 基于 chroma_db 全局向量库（docs/ 目录建索引的文档）

回答只来自给定资料、可溯源；答错（学员不认可 / 自检不通过）可记入错题本。
"""

import math

import chromadb

from config import CHROMA_DIR, TOP_K
from embedding import OllamaEmbedding
from llm_client import generate

_TUTOR_TMPL = """你是工业技术文档的答疑助教。请仅根据下面提供的【文档片段】回答学员的问题。
如果片段中没有答案，如实说"文档中未提及"。不要编造。

【文档片段】
{context}

【学员问题】
{question}

【回答】"""


def _cosine(a, b):
    if a is None or b is None or len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _chunk(text: str, size: int = 800, overlap: int = 100):
    """把长文本切成带重叠的小块，避免长章节直接撑爆上下文。"""
    n = len(text)
    if n <= size:
        return [text] if text.strip() else []
    chunks, start = [], 0
    while start < n:
        end = min(start + size, n)
        seg = text[start:end].strip()
        if seg:
            chunks.append(seg)
        start += size - overlap
    return chunks


def ask_from_chapters(question: str, chapters: list):
    """基于当前文档章节内容做检索增强答疑（不依赖 chroma_db，支持上传文档）。

    检索方式：将每章切成小块（800 字、重叠 100），对块做向量相似度，
    仅取最相关的 TOP_K 块进上下文，从而对任意长度的文档都稳定、不溢出。

    chapters: [{"title","content","source"}, ...]（原型机已解析好的当前文档章节）
    返回 (answer, sources, kind)，kind="doc" 表示纯文档来源。
    """
    embed = OllamaEmbedding()
    q_vec = [float(x) for x in embed([question])[0]]

    # 切片后按块算相似度
    scored = []
    for ch in chapters:
        for i, seg in enumerate(_chunk(ch["content"])):
            c_vec = [float(x) for x in embed([seg])[0]]
            sim = _cosine(q_vec, c_vec)
            scored.append((sim, ch["title"], seg))
    scored.sort(key=lambda x: x[0], reverse=True)

    docs, sources = [], []
    for sim, title, seg in scored[:TOP_K]:
        if sim < 0.1:
            continue
        docs.append(seg)
        sources.append(title)
    if not docs:
        return "在当前文档中未检索到相关内容。", [], "doc"

    context = "\n---\n".join(docs)
    prompt = _TUTOR_TMPL.format(context=context, question=question)
    answer = generate(prompt)
    return answer, sources, "doc"


def ask_doc(question: str):
    """基于 chroma_db 全局向量库检索答疑（docs/ 目录建索引的文档）。"""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection("docs", embedding_function=OllamaEmbedding())
    except Exception:
        return "尚未建立索引，请先运行：python ingest.py", [], "doc"
    results = collection.query(query_texts=[question], n_results=TOP_K)
    docs = results["documents"][0]
    sources = sorted({m["source"] for m in results["metadatas"][0]})
    context = "\n---\n".join(docs)
    prompt = _TUTOR_TMPL.format(context=context, question=question)
    answer = generate(prompt)
    return answer, sources, "doc"


# 来源标识：用于在前端醒目提示答案出处
SOURCE_BADGE = {
    "doc": "📚 本文档（可溯源，可信赖）",
    "web": "🌐 联网/在线模型（外部资料，仅供参考）",
    "model": "🧠 模型推理/通用知识（非文档原文，仅供参考）",
}


_WEB_TMPL = """你是工业技术文档的答疑助教。下面提供了通过联网搜索得到的【外部资料片段】，
请结合这些公开资料，对学员关于技术文档的问题做补充解析，帮助其更深入理解。
如果资料不足，可基于你自身的通用知识补充，但需注明"以下含外部/通用知识补充"。
不要编造具体文献来源。

【外部资料片段】
{context}

【学员问题】
{question}

【解析】"""


_ONLINE_TMPL = """你是工业技术文档的答疑助教。学员正在学习一份技术文档，但允许你结合自身的
广博知识（含联网可获取的通识）对其问题做补充解析，帮助更深入理解。
请注意：你的回答可能超出当前文档范围，属于外部/通用知识补充，不要声称它来自该文档。
不要编造具体文献来源。

【学员问题】
{question}

【解析】"""


def ask_web(question: str, max_results: int = 4):
    """「允许联网」模式的深度解析：返回 (answer, sources, kind)。

    kind 标识答案来源，前端必须醒目展示：
      - "web"  ：真实联网检索成功，或走在线模型（DeepSeek）的联网/通用知识
      - "model"：本地离线模型、联网失败，仅基于模型自身通用知识（非文档）

    设计：在线后端（DeepSeek）直接用其通用知识/联网能力解析（kind=web）；
    本地后端（Ollama）尝试真实 DuckDuckGo 爬网，成功 kind=web，失败 kind=model。
    """
    from config import LLM_PROVIDER

    if LLM_PROVIDER == "deepseek":
        # 在线模型：本就具备广博知识与联网能力，直接让其在通用知识下解析
        try:
            answer = generate(_ONLINE_TMPL.format(question=question))
            return answer, [], "web"
        except Exception as e:
            return f"（在线模型调用失败）{e}", [], "model"

    # 本地后端：尝试真实联网爬网
    try:
        snippets, sources = _web_search(question, max_results)
    except Exception:
        # 离线 + 联网失败：只能用本地模型自身通用知识，明确标注非文档
        answer = generate(_ONLINE_TMPL.format(question=question))
        return answer, [], "model"
    if not snippets:
        answer = generate(_ONLINE_TMPL.format(question=question))
        return answer, [], "model"
    context = "\n---\n".join(snippets)
    prompt = _WEB_TMPL.format(context=context, question=question)
    answer = generate(prompt)
    return answer, sources, "web"


def ask_tutor(question: str, chapters: list = None):
    """统一答疑入口：优先 chroma_db 全局向量库（方案五），失败/无索引则回退章节检索。

    返回 (answer, sources, kind)。这样长文档既稳定（向量库按块检索），
    又兼容「上传即学」的会话内章节检索。
    """
    try:
        ans, sources, kind = ask_doc(question)
        if sources:  # 有命中来源才算成功，否则回退
            return ans, sources, kind
    except Exception:
        pass
    if chapters:
        return ask_from_chapters(question, chapters)
    return "尚未建立索引，且该文档不在当前会话中，无法答疑。", [], "doc"


def _web_search(query: str, max_results: int = 4):
    """用 DuckDuckGo HTML 接口做轻量联网检索（仅标准库，无需 bs4）。

    返回 (snippets, sources)。联网不可用或解析失败则抛异常，由 ask_web 回退。
    """
    import urllib.parse
    import urllib.request
    from html.parser import HTMLParser

    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    class _DDParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.snippets = []
            self.sources = []
            self._in_snip = False
            self._cur = []

        def handle_starttag(self, tag, attrs):
            d = dict(attrs)
            cls = d.get("class", "")
            if "result__snippet" in cls:
                self._in_snip = True
                self._cur = []

        def handle_endtag(self, tag):
            if self._in_snip and tag == "a":
                text = " ".join(self._cur).strip()
                if text:
                    self.snippets.append(text)
                self._in_snip = False

        def handle_data(self, data):
            if self._in_snip:
                self._cur.append(data)

    p = _DDParser()
    p.feed(html)
    return p.snippets[:max_results], p.sources[:max_results]
