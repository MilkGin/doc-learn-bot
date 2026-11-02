# -*- coding: utf-8 -*-
"""端到端闭环实验：bge-m3 检索修复后的 RAG 答案正确率。

补三块拼图：
1. Part A：longctx 实验的 17 题重跑 RAG 路线（bge-m3 top-4 检索）——验证
   `eval/longctx_vs_rag.md` 的预测"RAG 输在检索，修复后应翻盘"（nomic 版 RAG 是 30.8%）。
2. Part B：分层抽样 50 题（按类型等比、类型内按 id 等距），bge top-4 端到端正确率。
3. Part C：同 50 题 bge top-8（P0-3 配置 C：TOP_K 4→8 的端到端实测）。

生成器与 longctx 实验一致（qwen2.5:7b，num_ctx=8192，同 prompt 模板），
唯一变量是检索配置。判卷：GLM-5.3-Flash（与生成器异源），标准同 longctx。
查询向量复用 eval/embed_cache/queries_bge.npz（run_eval 已缓存，确定性）。
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import chromadb  # noqa: E402
import ollama  # noqa: E402
from config import OLLAMA_BASE_URL, LLM_MODEL  # noqa: E402

PROMPT_TMPL = """你是工业技术文档的答疑助教。请仅根据下面提供的【文档片段】回答学员的问题。
如果片段中没有答案，如实说"文档中未提及"。不要编造。

【文档片段】
{context}

【学员问题】
{question}

【回答】"""


def load_all():
    qa = [json.loads(l) for l in open(os.path.join(HERE, "qa_set.jsonl"), encoding="utf-8") if l.strip()]
    blob = np.load(os.path.join(HERE, "embed_cache", "chunks_bge.npz"), allow_pickle=True)
    chunk_ids, chunk_vecs = list(blob["ids"]), blob["vectors"].astype(np.float32)
    qcache = np.load(os.path.join(HERE, "embed_cache", "queries_bge.npz"), allow_pickle=True)
    qindex = {str(q): i for i, q in enumerate(qcache["questions"])}
    col = chromadb.PersistentClient(path=os.path.join(HERE, "chroma_nomic")).get_collection("docs")
    got = col.get(include=["documents"])
    texts = {cid: doc for cid, doc in zip(got["ids"], got["documents"])}
    return qa, chunk_ids, chunk_vecs, qindex, qcache["vectors"].astype(np.float32), texts


def retrieve(qvec, chunk_vecs, topk):
    qn = qvec / max(float(np.linalg.norm(qvec)), 1e-9)
    sims = chunk_vecs @ qn
    return np.argsort(-sims)[:topk]


def stratify(qa, n=50):
    """按类型等比分层 50 题，类型内按 id 等距取样（确定性）。"""
    from collections import Counter
    total = len(qa)
    cnt = Counter(r["type"] for r in qa)
    target, allocated = {}, 0
    for t in ["factoid", "hard_negative", "multi_hop", "no_answer"]:
        target[t] = round(n * cnt[t] / total)
        allocated += target[t]
    target["factoid"] += n - allocated
    out = []
    for t, k in target.items():
        pool = sorted([r for r in qa if r["type"] == t], key=lambda r: r["id"])
        step = max(1, len(pool) // k)
        out.extend(pool[i] for i in range(0, len(pool), step)[:k])
    return out


def main():
    qa, chunk_ids, chunk_vecs, qindex, qvecs, texts = load_all()
    client = ollama.Client(host=OLLAMA_BASE_URL, timeout=600)

    longctx_qids = sorted({json.loads(l)["id"] for l in open(os.path.join(HERE, "longctx_raw.jsonl"), encoding="utf-8")})
    qa_by_id = {r["id"]: r for r in qa}
    partA = [qa_by_id[qid] for qid in longctx_qids]
    partB = stratify(qa, 50)

    results = []

    def run(route, recs, topk):
        for r in recs:
            q = r["question"]
            qv = qvecs[qindex[q]]
            idx = retrieve(qv, chunk_vecs, topk)
            ctx = "\n---\n".join(texts[chunk_ids[i]] for i in idx)
            prompt = PROMPT_TMPL.format(context=ctx, question=q)
            t0 = time.time()
            try:
                resp = client.generate(model=LLM_MODEL, prompt=prompt, stream=False,
                                       options={"num_ctx": 8192})
                ans, ptok, err = resp["response"], resp.get("prompt_eval_count", 0), ""
            except Exception as e:
                ans, ptok, err = "", 0, f"{type(e).__name__}: {str(e)[:120]}"
            results.append({"part": route, "id": r["id"], "type": r["type"], "question": q,
                            "reference_answer": r["reference_answer"], "topk": topk,
                            "latency_s": round(time.time() - t0, 2), "prompt_tokens": ptok,
                            "answer": ans, "error": err})
            print(f"[{route}|{r['id']}] {results[-1]['latency_s']}s err={err[:40]}", flush=True)

    run("A_bge_top4_longctx17", partA, 4)
    run("B_bge_top4_strat50", partB, 4)
    run("C_bge_top8_strat50", partB, 8)
    with open(os.path.join(HERE, "e2e_bge_raw.jsonl"), "w", encoding="utf-8") as f:
        for x in results:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"saved {len(results)} -> eval/e2e_bge_raw.jsonl")


if __name__ == "__main__":
    main()
