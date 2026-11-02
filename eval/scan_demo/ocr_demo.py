# -*- coding: utf-8 -*-
"""P1-5 扫描件 demo：扫描页图片 → 结构化文本（保留标题层级）→ 现有 chapter_split.py。

用本地多模态模型 qwen2.5vl:3b（Ollama，完全离线）做"视觉 OCR"，
证明 README 已知限制①（扫描版 PDF 无文本层）可以由多模态模型补足，
且产物直接进产品现有的章节切分管线，不需要新链路。

输入页图片由 eval/scan_demo/render_pages.py 从 设备B 手册 PDF 渲染（150dpi）。
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import ollama  # noqa: E402
from config import OLLAMA_BASE_URL  # noqa: E402
from chapter_split import split_into_chapters  # noqa: E402

VL_MODEL = "qwen2.5vl:3b"
PROMPT = (
    "这是技术手册的一页扫描图。请把整页内容转录为 Markdown 纯文本："
    "标题用 #/##/### 标注层级，正文按原文照抄（包括编号和参数），"
    "页眉页脚和页码可以省略，不要添加任何解释或评论，只输出转录内容。"
)


def ocr_page(path):
    client = ollama.Client(host=OLLAMA_BASE_URL, timeout=600)
    resp = client.chat(model=VL_MODEL, messages=[{
        "role": "user",
        "content": PROMPT,
        "images": [path],
    }])
    return resp["message"]["content"]


def main():
    pages = sorted(glob.glob(os.path.join(HERE, "page_*.png")))
    outs = []
    for p in pages:
        print(f"OCR {os.path.basename(p)} ...", flush=True)
        text = ocr_page(p)
        out_path = p.replace(".png", ".transcribed.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        outs.append((os.path.basename(p), text))
        print(f"  -> {out_path} ({len(text)} chars)")

    # 拼接转录文本，直接走产品现有的章节切分
    full = "\n\n".join(t for _, t in outs)
    chapters = split_into_chapters(full)
    print(f"\nchapter_split 解析出 {len(chapters)} 章：")
    for ch in chapters:
        print(f"  [L{ch['level']}] {ch['title'][:50]}  ({len(ch['content'])} 字)")

    # 与 pypdf 文本层的粗略对照（转录质量参考）
    import re
    def norm(s):
        return re.sub(r"\s+", "", s)
    ground = {
        "page_rth.png": "自动返航", "page_rc.png": "遥控器", "page_toc.png": "目录",
    }
    report = {"pages": [], "chapters": [{"title": c["title"], "level": c["level"],
                                          "chars": len(c["content"])} for c in chapters]}
    for name, text in outs:
        key = [g for n, g in ground.items() if n in name]
        hit = norm(key[0]) in norm(text) if key else None
        report["pages"].append({"page": name, "transcribed_chars": len(text),
                                "contains_expected_keyword": hit})
        print(f"  转录校验 {name}: 关键词'{key[0] if key else ''}'命中={hit}")
    with open(os.path.join(HERE, "scan_demo_result.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("saved -> eval/scan_demo_result.json")


if __name__ == "__main__":
    main()
