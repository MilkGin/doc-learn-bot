"""文档语义变更检测。

解决的问题：技术文档改版后，用户其实不需要**整篇重学**——往往只有个别章节
内容真的变了。文本 diff 做不到这点（改个错别字、调整措辞也会命中），
所以这里改用**向量相似度**：把每章压成一个向量，新旧版本逐章比对，
只把"含义真的变了"的章节标出来。

判定规则（对新版每个章节，取旧版中最相似的一章）：

    similarity >= SAME_THRESHOLD      -> 语义未变，可直接跳过
    CHANGED_THRESHOLD <= sim < SAME   -> 内容有修改，建议重学
    similarity < CHANGED_THRESHOLD    -> 新增章节
    旧版中从未被匹配上的章节           -> 已删除

快照存本地 `doc_snapshot.json`（已在 .gitignore 中），不依赖数据库。
"""

import difflib
import json
import math
import os
import re
import time

_PATH = os.path.join(os.path.dirname(__file__), "doc_snapshot.json")

# 章节向量取「标题 + 正文前 N 字」：覆盖主干语义，同时控制嵌入开销
VECTOR_CHARS = 800

# 阈值：可按文档类型调。中文技术文档改写措辞时相似度通常在 0.9 以上
SAME_THRESHOLD = 0.93
CHANGED_THRESHOLD = 0.80

# 标题相似度阈值：内容对不上但"还是同一节"时，判为「内容重写」而非「删除+新增」
TITLE_SAME = 0.90

# 向量小数位：保留 5 位足够判别，能把快照体积压掉大半
_VEC_ROUND = 5


# ---------- 快照读写 ----------

def _load() -> dict:
    if not os.path.exists(_PATH):
        return {}
    try:
        with open(_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        # 快照损坏不影响主流程，直接当作没有历史
        return {}


def _save(data: dict) -> None:
    """原子写入：先写临时文件再替换，避免中途失败留下半个 JSON。"""
    tmp = _PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, _PATH)
    except OSError as e:
        print(f"[doc_watch] 快照写入失败: {e}")


# ---------- 基础工具 ----------

def _cosine(a, b) -> float:
    """余弦相似度。维度不一致或零向量时返回 0.0（视为不相关）。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b):
        fx, fy = float(x), float(y)
        dot += fx * fy
        norm_a += fx * fx
        norm_b += fy * fy
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def identity_of(chapters: list) -> str:
    """文档身份标识 —— 必须**与内容无关**，才能在改版后认出是同一篇文档。

    优先用来源文件名（上传的文件名通常稳定）；没有来源时回退到首章标题。
    注意不要用内容哈希：内容一变哈希就变，就认不出是同一篇了。
    """
    if not chapters:
        return "unknown"
    sources = sorted({c.get("source", "") for c in chapters if c.get("source")})
    if sources:
        return "|".join(sources)
    return chapters[0].get("title", "未命名文档")


def chapter_text(ch: dict) -> str:
    """章节用于向量化的文本：标题 + 正文前 N 字。"""
    return f"{ch.get('title', '')}\n{(ch.get('content') or '')[:VECTOR_CHARS]}"


# 去掉标题里的编号前缀（"1. " / "3.2 " / "第1章 " / "一、"），只留文字。
# 要求编号后面必须跟分隔符，否则不剥离 —— 避免把"一般说明"这类标题误伤成"般说明"。
_NUM_PREFIX_RE = re.compile(
    r"^\s*(?:第\s*[0-9一二三四五六七八九十]+\s*[章节]"
    r"|[0-9]+(?:\.[0-9]+)*"
    r"|[一二三四五六七八九十]+)"
    r"\s*[.、．)）:：\-—\s]+"
)


def _norm_title(title: str) -> str:
    """标题归一化，用于判断新版这一节是不是旧的同一节。"""
    return _NUM_PREFIX_RE.sub("", (title or "").strip()).strip().lower()


def _title_similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm_title(a), _norm_title(b)).ratio()


# ---------- 向量化（模型层，单独隔离以便离线测试） ----------

def embed_chapters(chapters: list):
    """把章节批量转成向量。

    返回 list[list[float]]；**Ollama 不可用或模型缺失时返回 None**，
    调用方应静默降级（变更检测只是增强功能，不该阻断学习流程）。
    """
    try:
        from embedding import OllamaEmbedding  # 延迟导入，避免无 Ollama 时 import 即失败
        fn = OllamaEmbedding()
        texts = [chapter_text(c) for c in chapters]
        return [list(v) for v in fn(texts)]
    except Exception as e:  # noqa: BLE001
        print(f"[doc_watch] 章节向量化失败，本次跳过变更检测: {e}")
        return None


def _round_vec(vec) -> list[float]:
    return [round(float(x), _VEC_ROUND) for x in vec]


# ---------- 比对核心（纯函数，可脱离模型测试） ----------

def compare(old_chapters: list, chapters: list, vectors: list) -> dict:
    """旧快照 vs 新章节，逐章分类。

    old_chapters: [{"title", "level", "vec"}, ...]
    chapters / vectors: 新版章节与对应向量（等长）

    对新版每一章，在旧版中找最相似的一章 → 未变 / 变更 / 新增；
    旧版里从未被匹配的章节 → 已删除。
    """
    result = {"same": [], "changed": [], "added": [], "removed": []}

    if not old_chapters:
        result["added"] = [
            {"idx": i, "title": ch.get("title", ""), "similarity": 0.0}
            for i, ch in enumerate(chapters)
        ]
        result["needs_relearn"] = [a["idx"] for a in result["added"]]
        result["is_update"] = bool(chapters)
        return result

    matched_old = set()
    for i, (ch, vec) in enumerate(zip(chapters, vectors)):
        title = ch.get("title", "")

        # 只在「尚未被占用的」旧章节里找最相似的一章，保证匹配是一对一的
        best_j, best_sim = -1, -1.0
        for j, oc in enumerate(old_chapters):
            if j in matched_old:
                continue
            sim = _cosine(vec, oc.get("vec") or [])
            if sim > best_sim:
                best_j, best_sim = j, sim

        if best_j >= 0 and best_sim >= CHANGED_THRESHOLD:
            # 内容对得上：未变 或 有修改
            matched_old.add(best_j)
            sim = round(best_sim, 4)
            if best_sim >= SAME_THRESHOLD:
                result["same"].append({"idx": i, "title": title, "similarity": sim})
            else:
                result["changed"].append({
                    "idx": i,
                    "title": title,
                    "similarity": sim,
                    "was": old_chapters[best_j].get("title", ""),
                })
            continue

        # 内容对不上：再看标题能否指向旧版同一节。
        # 是 → 这一节被整段重写（报「变更」比报「删除 + 新增」更符合直觉）
        # 否 → 才算真正新增的内容
        same_slot = None
        if best_j >= 0 and _title_similar(title, old_chapters[best_j].get("title", "")) >= TITLE_SAME:
            same_slot = best_j
        if same_slot is None:
            same_slot = next(
                (j for j, oc in enumerate(old_chapters)
                 if j not in matched_old
                 and _title_similar(title, oc.get("title", "")) >= TITLE_SAME),
                None,
            )

        if same_slot is not None:
            matched_old.add(same_slot)
            result["changed"].append({
                "idx": i,
                "title": title,
                "similarity": round(best_sim, 4) if best_sim > 0 else 0.0,
                "was": old_chapters[same_slot].get("title", ""),
                "reason": "同一节内容被重写",
            })
        else:
            result["added"].append({
                "idx": i,
                "title": title,
                "similarity": round(best_sim, 4) if best_sim > 0 else 0.0,
            })

    result["removed"] = [
        oc.get("title", "") for j, oc in enumerate(old_chapters) if j not in matched_old
    ]
    result["needs_relearn"] = sorted(
        [c["idx"] for c in result["changed"]] + [a["idx"] for a in result["added"]]
    )
    # 只有在"内容真的变了"时才算更新；纯排版调整不打扰用户
    result["is_update"] = bool(result["changed"] or result["added"] or result["removed"])
    return result


def build_entry(chapters: list, vectors: list) -> dict:
    return {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_chapters": len(chapters),
        "chapters": [
            {"title": c.get("title", ""), "level": c.get("level", 1), "vec": _round_vec(v)}
            for c, v in zip(chapters, vectors)
        ],
    }


# ---------- 对外主入口 ----------

def check(chapters: list, persist: bool = True):
    """检测该文档相对上次快照的语义变更，并把新快照写回。

    返回变更报告 dict；以下情况返回 None：
      - 首次见到该文档（没有历史可比）
      - 向量化不可用（Ollama 没跑 / 模型缺失）

    报告字段：
      same / changed / added / removed  —— 四类章节明细
      needs_relearn                     —— 建议重学的章节序号（changed + added）
      is_update                         —— 是否有实质变更
    """
    if not chapters:
        return None

    vectors = embed_chapters(chapters)
    if vectors is None or len(vectors) != len(chapters):
        return None

    key = identity_of(chapters)
    store = _load()
    old = store.get(key)

    report = None
    if old and old.get("chapters"):
        report = compare(old["chapters"], chapters, vectors)
        report["identity"] = key

    if persist:
        store[key] = build_entry(chapters, vectors)
        _save(store)

    return report


def main():
    """自测：构造一版文档，修改后再比对，验证分类是否正确。"""
    def mk(title, content):
        return {"title": title, "level": 1, "content": content, "source": "demo.md"}

    def fake_vec(text):
        """确定性伪向量：按词袋散列，语义相近的文本向量也相近。"""
        dim = 64
        vec = [0.0] * dim
        for tok in str(text).replace("\n", " ").split():
            vec[hash(tok) % dim] += 1.0
        return vec

    old = [mk("安装", "安装步骤 依赖 环境"), mk("配置", "配置项 说明 默认值"),
           mk("常见问题", "报错 排查 处理")]
    new = [mk("安装", "安装步骤 依赖 环境"),            # 未变
           mk("配置", "配置项 说明 默认值 新增参数"),      # 变更
           mk("进阶用法", "高级 技巧 性能优化")]           # 新增（旧版"常见问题"消失）

    old_vecs = [fake_vec(chapter_text(c)) for c in old]
    new_vecs = [fake_vec(chapter_text(c)) for c in new]

    rep = compare([dict(c, vec=v) for c, v in zip(old, old_vecs)], new, new_vecs)
    print("未变:", [c["title"] for c in rep["same"]])
    print("变更:", [c["title"] for c in rep["changed"]])
    print("新增:", [c["title"] for c in rep["added"]])
    print("删除:", rep["removed"])
    print("需重学章节序号:", rep["needs_relearn"], "| 是否更新:", rep["is_update"])


if __name__ == "__main__":
    main()
