# -*- coding: utf-8 -*-
"""P0-2(a) 评测集装配与校验。

评测条目由 GLM（本会话）通读 eval/corpus 语料后人工撰写，存于 eval/qa_authored/*.jsonl，
每条带 gold_locators（答案所在原文的唯一子串）。本脚本负责：
1. 从 eval/chroma_nomic 读出真实 chunk（id/文本），把 locator 解析为 gold_chunk_ids；
2. 校验：locator 找不到 -> 丢弃该条；no_answer 的 absent_keywords 在语料中出现 -> 丢弃；
3. 输出 eval/qa_set.jsonl（字段 id/type/question/reference_answer/gold_chunk_ids/gold_doc），
   以及解析质量报告。可重复运行（固定输入 -> 固定输出，无随机性）。

生成方式说明（如实记录）：题目文本由 GLM 会话内生成（等价于"用 GLM 生成"），
本脚本只做确定性解析与校验，不用模型、不引入随机性。
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import chromadb  # noqa: E402

CHROMA_DIR = os.path.join(ROOT, "eval", "chroma_nomic")
AUTHORED = os.path.join(HERE, "qa_authored")


def norm(s: str) -> str:
    """去掉所有空白，规避换行/空格导致的匹配失败。"""
    return re.sub(r"\s+", "", s)


def load_chunks():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection("docs")
    r = col.get(include=["documents", "metadatas"])
    chunks = []
    for cid, doc, meta in zip(r["ids"], r["documents"], r["metadatas"]):
        base = os.path.basename(meta["source"])
        chunks.append({"id": cid, "doc": base, "text": doc, "ntext": norm(doc)})
    return chunks


def main():
    chunks = load_chunks()
    by_doc = {}
    for c in chunks:
        by_doc.setdefault(c["doc"], []).append(c)

    records, dropped = [], []
    for path in sorted(glob.glob(os.path.join(AUTHORED, "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    out, multihop_collapsed = [], 0
    for rec in records:
        doc = rec["doc"]
        pool = by_doc.get(doc, [])
        if not pool:
            dropped.append((rec["id"], f"doc {doc} 不在索引中"))
            continue
        rtype = rec["type"]
        gold = []
        ok = True
        for loc in rec.get("gold_locators", []):
            nloc = norm(loc)
            hit_ids = [c["id"] for c in pool if nloc in c["ntext"]]
            if not hit_ids:
                dropped.append((rec["id"], f"locator 未命中: {loc[:50]}"))
                ok = False
                break
            gold.extend(hit_ids)
        if not ok:
            continue
        gold = sorted(set(gold))

        if rtype == "no_answer":
            corpus_text = "".join(c["ntext"] for c in pool)
            for kw in rec.get("absent_keywords", []):
                if norm(kw) in corpus_text:
                    dropped.append((rec["id"], f"absent_keyword 实际出现在语料: {kw}"))
                    ok = False
                    break
            if not ok:
                continue
            gold = []

        if rtype == "multi_hop" and len(gold) < 2:
            multihop_collapsed += 1

        out.append({
            "id": rec["id"],
            "type": rtype,
            "question": rec["question"],
            "reference_answer": rec["reference_answer"],
            "gold_chunk_ids": gold,
            "gold_doc": doc,
            "note": rec.get("note", ""),
        })

    out.sort(key=lambda r: r["id"])
    with open(os.path.join(HERE, "qa_set.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 统计
    from collections import Counter
    tc = Counter(r["type"] for r in out)
    n_ans = tc["factoid"] + tc["multi_hop"] + tc["hard_negative"]
    print(f"输入 {len(records)} 条，保留 {len(out)} 条，丢弃 {len(dropped)} 条")
    print("类型分布:", dict(tc))
    print(f"比例: " + ", ".join(f"{k}={v/len(out)*100:.1f}%" for k, v in tc.items()))
    print(f"multi_hop 中 gold 块数<2（退化为单块）: {multihop_collapsed} 条")
    print(f"answerable={n_ans}, no_answer={tc['no_answer']}")
    if dropped:
        print("\n丢弃明细:")
        for qid, why in dropped:
            print(f"  - {qid}: {why}")
    if len(out) < 200:
        print(f"\n!! 少于 200 条（{len(out)}），不达标")
        sys.exit(1)
    print(f"\nOK -> eval/qa_set.jsonl（{len(out)} 条）")


if __name__ == "__main__":
    main()
