# -*- coding: utf-8 -*-
"""P0-2(b) RAG 检索指标评测。

为什么不用 RAGAS（任务书 P0-2(b) 要求记录）：
  ragas 0.4.3 可以 pip 安装，但 `import ragas` 即报
  ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
  （ragas 0.4.3 与 langchain-community 的固定导入冲突，需锁版本才能修）。
  且其 faithfulness/answer_relevancy 等指标需要 LLM 裁判：可用的裁判只有
  qwen2.5:7b——与被评的生成器同模型，自评有偏（任务书明确警示）。
  故检索侧指标按公式自算（recall@k / MRR@10 / 正确拒答率，均为标准定义，
  非"自造指标"），生成侧质量不做模型自评。RAGAS 若后续修好依赖，
  可作为生成侧指标的补充。

检索语义与产品路径等价：qa 问题用 ollama 嵌入（确定性），chunk 向量来自
eval/embed_cache/chunks_{tag}.npz（由 embed_cache.py 生成，同模型同输入）。
排序用余弦相似度——已验证（eval/l2_vs_cosine_diag.json）chunk 向量范数恒为 1，
chroma 默认 L2 距离与余弦严格单调等价、top-1 全部一致，故本脚本的排序
= chroma collection.query 的排序 = tutor.ask_from_chapters 的余弦排序。

指标：
  answerable（factoid/multi_hop/hard_negative）：
    recall@k (k=1,3,5,10) = mean(|topk ∩ gold| / |gold|)
    hit@5                 = gold 全部落入 top5 的题占比
    MRR@10                = 第一个 gold 块排名倒数的均值
  no_answer（正确拒答率）：top1 余弦 < τ 视为拒答，报 τ=0.40/0.50/0.60 三档
  全部指标应落在 (0,1) 开区间，恰好 0/1 视为可疑并告警。

用法：
  python eval/run_eval.py --tag nomic         # 基线（无前缀）
  python eval/run_eval.py --tag nomic_prefix  # nomic 官方任务前缀
  python eval/run_eval.py --tag nomic --repeat-check   # 连跑两次校验一致性
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import ollama  # noqa: E402
from config import OLLAMA_BASE_URL  # noqa: E402

CACHE = os.path.join(HERE, "embed_cache")
KS = [1, 3, 5, 10]
TAUS = [0.55, 0.60, 0.65, 0.70]


def embed_queries(questions, model, tag):
    """查询嵌入，落盘缓存保证复跑确定性。"""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"queries_{tag}.npz")
    if os.path.exists(path):
        blob = np.load(path, allow_pickle=True)
        cached_q, cached_v = list(blob["questions"]), blob["vectors"]
    else:
        cached_q, cached_v = [], np.zeros((0, 0), dtype=np.float32)
    index = {q: i for i, q in enumerate(cached_q)}
    todo = [q for q in questions if q not in index]
    if todo:
        client = ollama.Client(host=OLLAMA_BASE_URL, timeout=600)
        vecs = []
        for i in range(0, len(todo), 16):
            resp = client.embed(model=model, input=todo[i:i + 16])
            vecs.extend(resp["embeddings"])
        new_v = np.array(vecs, dtype=np.float32)
        cached_v = new_v if cached_v.size == 0 else np.vstack([cached_v, new_v])
        cached_q = cached_q + todo
        np.savez(path, questions=np.array(cached_q), vectors=cached_v)
    index = {q: i for i, q in enumerate(cached_q)}
    return np.stack([cached_v[index[q]] for q in questions])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="nomic")
    ap.add_argument("--model", default="nomic-embed-text")
    ap.add_argument("--query-prefix", default="", help="如 'search_query: '")
    ap.add_argument("--out", default=None)
    ap.add_argument("--repeat-check", action="store_true")
    args = ap.parse_args()

    qa = [json.loads(l) for l in open(os.path.join(HERE, "qa_set.jsonl"), encoding="utf-8") if l.strip()]
    blob = np.load(os.path.join(CACHE, f"chunks_{args.tag}.npz"), allow_pickle=True)
    chunk_ids = list(blob["ids"])
    chunk_vecs = blob["vectors"].astype(np.float32)
    pos = {cid: i for i, cid in enumerate(chunk_ids)}

    ans = [r for r in qa if r["type"] != "no_answer"]
    na = [r for r in qa if r["type"] == "no_answer"]

    qs = [r["question"] for r in qa]
    qvecs = embed_queries([args.query_prefix + q for q in qs], args.model, args.tag)
    qn = qvecs / np.maximum(np.linalg.norm(qvecs, axis=1, keepdims=True), 1e-9)
    sims = qn @ chunk_vecs.T  # (n_q, n_chunk)，chunk 向量已归一化
    order = np.argsort(-sims, axis=1)  # 稳定排序，分数相同按索引序

    metrics = {"config": {"tag": args.tag, "model": args.model,
                          "query_prefix": args.query_prefix,
                          "n_questions": len(qa), "n_chunks": len(chunk_ids)}}
    errors = []

    # ---- answerable ----
    recalls = {k: [] for k in KS}
    mrr, hit5 = [], []
    for qi, r in enumerate(qa):
        if r["type"] == "no_answer":
            continue
        gold = [pos[g] for g in r["gold_chunk_ids"] if g in pos]
        assert gold, f"{r['id']} 的 gold_chunk_ids 不在索引中"
        top = order[qi]
        rank_of = {int(c): rk for rk, c in enumerate(top[:10], 1)}
        for k in KS:
            got = sum(1 for g in gold if g in set(top[:k].tolist()))
            recalls[k].append(got / len(gold))
        first = min((rank_of[g] for g in gold if g in rank_of), default=None)
        mrr.append(0.0 if first is None else 1.0 / first)
        hit5.append(all(g in set(top[:5].tolist()) for g in gold))
        if first is None or first > 5:
            errors.append({
                "id": r["id"], "type": r["type"], "question": r["question"],
                "gold_chunk_ids": " ".join(r["gold_chunk_ids"]),
                "gold_ranks": " ".join(str(rank_of.get(g, ">10")) for g in gold),
                "top1_chunk": chunk_ids[int(top[0])],
                "top1_score": round(float(sims[qi][top[0]]), 4),
                "hit_at_5": False,
            })
        else:
            errors.append({
                "id": r["id"], "type": r["type"], "question": r["question"],
                "gold_chunk_ids": " ".join(r["gold_chunk_ids"]),
                "gold_ranks": " ".join(str(rank_of.get(g, ">10")) for g in gold),
                "top1_chunk": chunk_ids[int(top[0])],
                "top1_score": round(float(sims[qi][top[0]]), 4),
                "hit_at_5": True,
            })

    metrics["answerable"] = {
        "n": len(ans),
        **{f"recall@{k}": round(float(np.mean(recalls[k])), 4) for k in KS},
        "hit@5": round(float(np.mean(hit5)), 4),
        "MRR@10": round(float(np.mean(mrr)), 4),
    }

    # ---- 分文档召回（量化跨语言差距）----
    per_doc = {}
    docs = sorted({r["gold_doc"] for r in qa if r["type"] != "no_answer"})
    for d in docs:
        idx = [qi for qi, r in enumerate(qa) if r["type"] != "no_answer" and r["gold_doc"] == d]
        per_doc[d] = {
            "n": len(idx),
            "recall@5": round(float(np.mean([recalls[5][ans.index(qa[qi])] for qi in idx])), 4),
            "MRR@10": round(float(np.mean([mrr[ans.index(qa[qi])] for qi in idx])), 4),
        }
    metrics["per_gold_doc"] = per_doc

    # ---- no_answer：正确拒答率 ----
    na_rows = [(qi, r) for qi, r in enumerate(qa) if r["type"] == "no_answer"]
    ans_rows = [qi for qi, r in enumerate(qa) if r["type"] != "no_answer"]
    na_top1 = [float(sims[qi][order[qi][0]]) for qi, _ in na_rows]
    ans_top1 = [float(sims[qi][order[qi][0]]) for qi in ans_rows]
    rej = {}
    for tau in TAUS:
        rej[f"reject@tau={tau:.2f}"] = round(float(np.mean([s < tau for s in na_top1])), 4)
        rej[f"false_reject@tau={tau:.2f}"] = round(
            float(np.mean([s < tau for s in ans_top1])), 4)
    # 参考：answerable 的 top1 分布（检查两类分数是否可分）
    metrics["no_answer"] = {
        "n": len(na),
        **rej,
        "top1_sim_mean": round(float(np.mean(na_top1)), 4),
        "top1_sim_p90": round(float(np.percentile(na_top1, 90)), 4),
    }
    metrics["answerable_top1_sim"] = {
        "mean": round(float(np.mean(ans_top1)), 4),
        "p10": round(float(np.percentile(ans_top1, 10)), 4),
        "p90": round(float(np.percentile(ans_top1, 90)), 4),
    }

    # ---- 开区间自检：恰为 0/1 视为可疑 ----
    warnings = []
    for name in [f"recall@{k}" for k in KS] + ["hit@5", "MRR@10"]:
        v = metrics["answerable"][name]
        if v == 0.0 or v == 1.0:
            warnings.append(f"{name}={v} 恰为边界值，先怀疑评测脚本有 bug")
    for k, v in rej.items():
        if v == 0.0 or v == 1.0:
            warnings.append(f"{k}={v} 恰为边界值，先怀疑评测脚本有 bug")
    metrics["sanity_warnings"] = warnings

    out = args.out or os.path.join(HERE, f"metrics_{args.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    print(json.dumps(metrics, ensure_ascii=False, indent=1))

    err_path = os.path.join(HERE, f"errors_{args.tag}.csv")
    with open(err_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "type", "question", "gold_chunk_ids",
                                          "gold_ranks", "top1_chunk", "top1_score", "hit_at_5"])
        w.writeheader()
        w.writerows(errors)
    print(f"\nerrors -> {err_path}（{len(errors)} 条）")

    if args.repeat_check:
        print("[repeat-check] 请将同一命令连跑两次后 diff 两个输出文件；"
              "查询嵌入已缓存，指标计算为纯 numpy，结果应逐字节一致")
    print("done.")


if __name__ == "__main__":
    main()
