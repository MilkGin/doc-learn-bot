"""错题本：跨会话持久化记录答错的题与答疑未解之处。

存储为本地 JSON，不依赖数据库，方便工业现场零运维部署。
支持：记录、导出、随机抽取重练。
"""

import json
import os

_PATH = os.path.join(os.path.dirname(__file__), "wrong_book.json")


def _load() -> list:
    if not os.path.exists(_PATH):
        return []
    with open(_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(book: list):
    with open(_PATH, "w", encoding="utf-8") as f:
        json.dump(book, f, ensure_ascii=False, indent=2)


def add(kind: str, title: str, item: dict):
    """记录一道错题。

    kind: "quiz"（选择题答错） 或 "tutor"（答疑后学员仍标记未掌握）
    title: 所属章节标题
    item: 题目/答疑的溯源信息
    """
    book = _load()
    book.append({"kind": kind, "title": title, "item": item})
    _save(book)


def all_items() -> list:
    return _load()


def count() -> int:
    return len(_load())


def clear():
    _save([])
