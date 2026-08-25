"""学习进度本地保存：每个文档独立记录学习状态（断点续学）。

以 doc_key（文档内容哈希前 10 位）为索引，持久化到 progress.json：
- 当前章节 idx、当前题目 q_idx
- 已答 done、答对 score、连击 streak
- 上次学习时间
- 文档显示名（用于书架）

跨会话保存，关闭浏览器再打开能回到上次进度。
"""

import json
import os
import time

_PATH = os.path.join(os.path.dirname(__file__), "progress.json")
_SHELF_DIR = os.path.join(os.path.dirname(__file__), "docshelf")


def _load() -> dict:
    if not os.path.exists(_PATH):
        return {}
    try:
        with open(_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _save(data: dict):
    with open(_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_doc(doc_key: str):
    return _load().get(doc_key)


def update(doc_key: str, name: str, idx: int, q_idx: int,
           done: int, score: int, streak: int, total_chapters: int):
    """保存某个文档的学习进度。"""
    data = _load()
    data[doc_key] = {
        "name": name,
        "idx": idx,
        "q_idx": q_idx,
        "done": done,
        "score": score,
        "streak": streak,
        "total_chapters": total_chapters,
        "updated": time.time(),
    }
    _save(data)


def list_docs():
    """返回所有已学习文档的元信息列表（用于书架展示），按最近学习排序。"""
    data = _load()
    docs = []
    for key, v in data.items():
        pct = 0
        if v.get("total_chapters"):
            pct = round(100 * v["idx"] / v["total_chapters"])
        docs.append({
            "key": key,
            "name": v.get("name", "未命名文档"),
            "idx": v.get("idx", 0),
            "q_idx": v.get("q_idx", 0),
            "done": v.get("done", 0),
            "score": v.get("score", 0),
            "streak": v.get("streak", 0),
            "total_chapters": v.get("total_chapters", 0),
            "percent": pct,
            "updated": v.get("updated", 0),
        })
    docs.sort(key=lambda d: d["updated"], reverse=True)
    return docs


def save_chapters(doc_key: str, chapters: list):
    """保存某文档的章节内容（用于书架恢复任意上传文档）。"""
    os.makedirs(_SHELF_DIR, exist_ok=True)
    p = os.path.join(_SHELF_DIR, f"{doc_key}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(chapters, f, ensure_ascii=False)


def load_chapters(doc_key: str):
    """读取某文档的章节内容；不存在返回 None。"""
    p = os.path.join(_SHELF_DIR, f"{doc_key}.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return None
