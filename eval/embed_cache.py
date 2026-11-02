# -*- coding: utf-8 -*-
"""嵌入缓存与诊断：
1. 从 chroma 取全部 chunk，用 ollama 批量 embed（.embed 接口，快于项目内逐条 .embeddings），
   按 --model 存 eval/embed_cache/{tag}.npz。
2. 报告向量范数分布；对冒烟问题对比 chroma-L2 排序与余弦排序（诊断基线失败原因）。

只读 chroma 库，产物全部写 eval/，不改任何现有代码。
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import chromadb  # noqa: E402
import ollama  # noqa: E402
from config import OLLAMA_BASE_URL  # noqa: E402

CACHE = os.path.join(HERE, "embed_cache")


def get_chunks(chroma_dir="eval/chroma_nomic", collection="docs"):
    client = chromadb.PersistentClient(path=os.path.join(ROOT, chroma_dir))
    col = client.get_collection(collection)
    r = col.get(include=["documents", "metadatas"])
    return list(r["ids"]), list(r["documents"]), list(r["metadatas"])


def embed_batch(texts, model, batch=32):
    client = ollama.Client(host=OLLAMA_BASE_URL, timeout=600)
    vecs = []
    for i in range(0, len(texts), batch):
        resp = client.embed(model=model, input=list(texts[i:i + batch]))
        vecs.extend(resp["embeddings"])
        if (i // batch) % 5 == 0:
            print(f"  embedded {i + len(resp['embeddings'])}/{len(texts)}", flush=True)
    return np.array(vecs, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nomic-embed-text")
    ap.add_argument("--tag", default="nomic")
    ap.add_argument("--prefix-doc", default="", help="文档侧前缀，如 'search_document: '")
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    ids, docs, metas = get_chunks()
    texts = [(args.prefix_doc + d) if args.prefix_doc else d for d in docs]
    print(f"chunks={len(ids)}, model={args.model}, prefix_doc={args.prefix_doc!r}")
    vecs = embed_batch(texts, args.model)

    norms = np.linalg.norm(vecs, axis=1)
    print(f"norm: min={norms.min():.3f} max={norms.max():.3f} mean={norms.mean():.3f} std={norms.std():.3f}")

    out = os.path.join(CACHE, f"chunks_{args.tag}.npz")
    np.savez_compressed(out, ids=np.array(ids), vectors=vecs,
                        sources=np.array([m["source"] for m in metas]))
    print("saved ->", out)

    # 冒烟问题诊断：L2 排序 vs 余弦排序
    smoke = json.load(open(os.path.join(HERE, "smoke_results.json"), encoding="utf-8"))
    qvecs = embed_batch([s["question"] for s in smoke], args.model)
    nv = vecs / np.maximum(norms, 1e-9)[:, None]
    report = []
    for s, qv in zip(smoke, qvecs):
        q = qv / max(float(np.linalg.norm(qv)), 1e-9)
        cos = nv @ q
        l2 = ((vecs - qv) ** 2).sum(axis=1)
        cos_rank = np.argsort(-cos)[:4]
        l2_rank = np.argsort(l2)[:4]
        l2_from_chroma = {h["rank"] - 1: h["text"][:40] for h in s["hits"]}
        row = {
            "qid": s["qid"], "question": s["question"],
            "chroma_l2_top1": s["hits"][0]["text"][:60].replace("\n", " "),
            "cosine_top1": docs[int(cos_rank[0])][:60].replace("\n", " "),
            "cos_top1_score": float(cos[cos_rank[0]]),
            "l2_nouns_top1": docs[int(l2_rank[0])][:60].replace("\n", " "),
            "agree_top1": int(cos_rank[0]) == int(l2_rank[0]),
            "cosine_top4_idx": [int(i) for i in cos_rank],
        }
        report.append(row)
        print(f"\n[{row['qid']}] {row['question']}")
        print(f"  chroma-L2 top1: {row['chroma_l2_top1']}")
        print(f"  cosine    top1: {row['cosine_top1']} (cos={row['cos_top1_score']:.4f})")
        print(f"  agree_top1 = {row['agree_top1']}")
    with open(os.path.join(HERE, "l2_vs_cosine_diag.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("\nsaved -> eval/l2_vs_cosine_diag.json")


if __name__ == "__main__":
    main()
