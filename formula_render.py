"""数学公式渲染组件：把文本中的 LaTeX 公式渲染为排版公式（KaTeX）。

支持格式：
- 独立公式  $$ ... $$
- 行内公式  \\( ... \\)   或   $...$
- 混合文本（普通文字 + 公式）

渲染资源优先用项目本地 static/katex（离线可用），缺失时回退到 CDN。

用法：
    from formula_render import render_text, render_latex
    render_text("效果函数定义：\\( \\mathcal{E}_\\Gamma \\to \\Gamma \\)")
    render_latex("\\mathcal{E}_\\Gamma \\to \\Gamma")   # 独立公式
"""

import os
import re
import tempfile
import uuid
from pathlib import Path

import streamlit as st

HERE = os.path.dirname(__file__)
_KATEX_DIR = os.path.join(HERE, "static", "katex")

# 匹配 $$...$$（独立公式）
_BLOCK_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
# 匹配 \( ... \)（行内公式）
_INLINE_BRACE_RE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
# 匹配 $...$（行内公式，非 $$ 开头）
_INLINE_DOLLAR_RE = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")


def _read_local(name: str):
    p = os.path.join(_KATEX_DIR, name)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return None


def _asset_tags() -> str:
    """返回 KaTeX 的 CSS/JS 标签：优先本地内联，否则 CDN。"""
    css = _read_local("katex.min.css")
    js = _read_local("katex.min.js")
    auto = _read_local("auto-render.min.js")

    if css and js and auto:
        # 全部本地可用 → 内联（离线可靠）
        return (
            f"<style>{css}</style>\n"
            f"<script>{js}</script>\n"
            f"<script>{auto}</script>\n"
        )
    # 本地缺失 → CDN 兜底
    return (
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">'
        '<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>'
        '<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>'
    )


def _latex_span(expr: str) -> str:
    esc = expr.replace("\\", "\\\\").replace('"', "&quot;")
    return f'<span class="katex-expr">{esc}</span>'


def render_html(markup: str) -> str:
    """把含 LaTeX 的文本转成完整 HTML 字符串。"""
    out = _BLOCK_RE.sub(lambda m: _latex_span(m.group(1)), markup)
    out = _INLINE_BRACE_RE.sub(lambda m: _latex_span(m.group(1)), out)
    out = _INLINE_DOLLAR_RE.sub(lambda m: _latex_span(m.group(1)), out)
    # 换行保留
    out = out.replace("\n", "<br/>")
    assets = _asset_tags()
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">{assets}</head>
<body style="font-family: inherit; font-size: 15px; line-height: 1.7; padding: 2px; margin: 0;">
<div>{out}</div>
<script>
window.onload = function() {{
  try {{
    renderMathInElement(document.body, {{
      delimiters: [
        {{left: "\\\\(", right: "\\\\)", display: false}},
        {{left: "$", right: "$", display: false}},
        {{left: "$$", right: "$$", display: true}}
      ],
      throwOnError: false
    }});
  }} catch(e) {{ console.error(e); }}
}};
</script>
</body></html>"""


def render_text(text: str, height: int = None):
    """在页面渲染含公式的文本。height 为 None 时按行数自动估算。

    用 st.iframe 渲染（写临时 HTML 文件，支持本地 KaTeX，离线可用）。
    """
    if not text:
        st.empty()
        return
    content = render_html(text)
    lines = text.count("\n") + 1
    h = height or max(80, min(600, lines * 24 + 40))

    # 写入临时文件供 st.iframe 加载
    tmp_path = os.path.join(tempfile.gettempdir(), f"_katex_{uuid.uuid4().hex}.html")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    st.iframe(Path(tmp_path), height=h)


def render_latex(expr: str):
    """渲染一个独立 LaTeX 公式（用 st.latex，原生支持）。"""
    st.latex(expr)
