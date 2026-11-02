# -*- coding: utf-8 -*-
"""P1-4 长上下文 vs RAG 对比实验。

设计（与任务书 P1-4 的两点偏离，均在任务书 §8 降级预案允许范围内）：
1. 长上下文路线不用 GLM API（无脚本调用通道），改用本地 qwen2.5:7b、num_ctx=32768。
   好处：两条路线生成器相同，唯一变量是"检索片段 vs 全文"，对比更公平。
2. 语料取 eval/corpus/kb_demo.md（三份语料中最小，69,244 字符）。
   预期发现：32k token 上下文装不下 6.9 万字符的中文文档——这本身就是
   "工业现场离线场景下长上下文路线不可用"的直接证据。

档位：
  T1 单节：4.2 产品保修期整节
  T2 双章：#4 售后政策 + #5 售后服务SOP
  T3 全篇：整份文档（超限时记录失败，并测 30,000 字符截断版）
路线：
  longctx：上述上下文直接进 prompt
  rag：nomic top-4 检索（与基线评测同配置）进同一 prompt 模板

每题记录：prompt_tokens / completion_tokens / 延迟 / 原始回答。裁判为 GLM-5.3-Flash
（与出题同源、与生成器 qwen2.5:7b 不同源），判定规则见 longctx_vs_rag.md。
"""
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import ollama  # noqa: E402
from config import OLLAMA_BASE_URL, LLM_MODEL  # noqa: E402

DOC = os.path.join(HERE, "corpus", "kb_demo.md")
PROMPT_TMPL = """你是工业技术文档的答疑助教。请仅根据下面提供的【文档片段】回答学员的问题。
如果片段中没有答案，如实说"文档中未提及"。不要编造。

【文档片段】
{context}

【学员问题】
{question}

【回答】"""


def section(text, start_pat, end_pat):
    s = re.search(start_pat, text)
    e = re.search(end_pat, text)
    assert s and e and s.start() < e.start(), (start_pat, end_pat)
    return text[s.start():e.start()]


def load_tiers():
    text = open(DOC, encoding="utf-8").read()
    t1 = section(text, r"## 4\.2 Products' Warranty Period", r"\n4\.3 Aftersales Team Contact")
    t2 = section(text, r"# 4\. Aftersales Policy 售后政策", r"# 6\. Pre-sales and In-sale")
    return {"T1_single_section": t1, "T2_two_chapters": t2, "T3_full": text}


def pick_questions(n_per_type=(10, 3, 4)):
    """从评测集抽该语料题：factoid 10 / multi_hop 3 / no_answer 4，固定取样（按 id 排序等距）。"""
    qa = [json.loads(l) for l in open(os.path.join(HERE, "qa_set.jsonl"), encoding="utf-8") if l.strip()]
    out = []
    for ttype, n in zip(["factoid", "multi_hop", "no_answer"], n_per_type):
        pool = sorted([r for r in qa if r["type"] == ttype and r["gold_doc"] == "kb_demo.md"],
                      key=lambda r: r["id"])
        step = max(1, len(pool) // n)
        out.extend(pool[i] for i in range(0, len(pool), step)[:n])
    return out


def retrieve_top4(question, chunk_ids, chunk_vecs, qvec):
    qn = qvec / max(float(np.linalg.norm(qvec)), 1e-9)
    sims = chunk_vecs @ qn
    top = np.argsort(-sims)[:4]
    return [chunks_text[i] for i in top]


def main():
    global chunks_text
    tiers = load_tiers()
    for k, v in tiers.items():
        print(f"{k}: {len(v)} chars")
    questions = pick_questions()
    print(f"questions: {len(questions)}")

    # RAG 检索材料：与基线评测同一份块向量缓存
    blob = np.load(os.path.join(HERE, "embed_cache", "chunks_nomic.npz"), allow_pickle=True)
    chunk_ids = list(blob["ids"])
    chunk_vecs = blob["vectors"].astype(np.float32)
    kb_pos = [i for i, c in enumerate(chunk_ids) if c.startswith("kb_demo.md")]
    chunks_text = [None] * len(chunk_ids)
    import chromadb
    col = chromadb.PersistentClient(path=os.path.join(HERE, "chroma_nomic")).get_collection("docs")
    got = col.get(include=["documents"])
    for cid, doc in zip(got["ids"], got["documents"]):
        chunks_text[chunk_ids.index(cid)] = doc
    qcache = np.load(os.path.join(HERE, "embed_cache", "queries_nomic.npz"), allow_pickle=True)
    qindex = {q: i for i, q in enumerate(qcache["questions"])}

    client = ollama.Client(host=OLLAMA_BASE_URL, timeout=1800)
    results = []
    for r in questions:
        q = r["question"]
        qvec = qcache["vectors"][qindex[q]]
        ctxs = {k: v for k, v in tiers.items()}
        ctxs["RAG_top4"] = "\n---\n".join(retrieve_top4(q, chunk_ids, chunk_vecs, qvec))
        for route, ctx in ctxs.items():
            prompt = PROMPT_TMPL.format(context=ctx, question=q)
            num_ctx = 8192 if route == "RAG_top4" else 32768
            t0 = time.time()
            try:
                resp = client.generate(model=LLM_MODEL, prompt=prompt, stream=False,
                                       options={"num_ctx": num_ctx})
                answer = resp["response"]
                ptok, ctok = resp.get("prompt_eval_count", 0), resp.get("eval_count", 0)
                err = ""
            except Exception as e:
                answer, ptok, ctok, err = "", 0, 0, f"{type(e).__name__}: {e}"
            dt = time.time() - t0
            results.append({"id": r["id"], "type": r["type"], "route": route,
                            "question": q, "reference_answer": r["reference_answer"],
                            "ctx_chars": len(ctx), "prompt_tokens": ptok,
                            "completion_tokens": ctok, "latency_s": round(dt, 2),
                            "answer": answer, "error": err})
            print(f"[{r['id']}|{route}] {dt:5.1f}s ptok={ptok} ctok={ctok} err={err[:60]}", flush=True)
            with open(os.path.join(HERE, "longctx_raw.jsonl"), "w", encoding="utf-8") as f:
                for x in results:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("saved -> eval/longctx_raw.jsonl")


if __name__ == "__main__":
    main()
