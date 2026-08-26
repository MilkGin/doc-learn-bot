"""技术文档学习机器人 — 网页原型机（Streamlit）。

功能：
- 主页"书架"：像微信读书一样，把学习中的文档分开列成卡片，点击继续学习
- 文档来源：上传任意文档（md/txt/pdf/docx）或粘贴文本
- 每章：讲解 → 即时选择题（即时反馈 + 连击）
- 答疑（基于当前文档章节内容）
- 错题本自动记录
- 学习进度本地保存（断点续学）
- 设置区：本地 Ollama / 在线 DeepSeek API 切换
- 文档更新检测 → 再学习弹窗
- 学习报告 + 得分

运行：streamlit run app_learn.py
"""

import hashlib
import os

import streamlit as st

from chapter_split import parse_uploaded, split_into_chapters
from course_builder import build_lesson, grade
from formula_render import render_text
from tutor import ask_from_chapters, ask_web, ask_tutor, SOURCE_BADGE
from wrong_book import add as wrong_add, count as wrong_count, all_items
from progress import (get_doc, update as save_progress, list_docs,
                      save_chapters, load_chapters)
import settings

HERE = os.path.dirname(__file__)


def doc_key_of(chapters: list) -> str:
    return hashlib.md5(
        "".join(f"{c['title']}|{c['content'][:200]}" for c in chapters).encode("utf-8")
    ).hexdigest()[:10]


def doc_name_of(chapters: list) -> str:
    """文档显示名：取第一个来源文件名或第一章标题。"""
    src = chapters[0].get("source", "")
    if src and src != "粘贴文本":
        return src
    return chapters[0].get("title", "未命名文档")


def _init_state(doc_key: str, chapters: list):
    """初始化/恢复某文档的学习会话状态。"""
    if st.session_state.get("doc_key") == doc_key and "chapters" in st.session_state:
        return
    st.session_state.doc_key = doc_key
    st.session_state.chapters = chapters
    st.session_state.generated = {}
    # 尝试恢复进度
    p = get_doc(doc_key)
    st.session_state.idx = p.get("idx", 0) if p else 0
    st.session_state.q_idx = p.get("q_idx", 0) if p else 0
    st.session_state.score = p.get("score", 0) if p else 0
    st.session_state.done = p.get("done", 0) if p else 0
    st.session_state.streak = p.get("streak", 0) if p else 0


def _persist():
    """把当前学习进度 + 章节内容写入本地。"""
    save_progress(
        st.session_state.doc_key,
        st.session_state.get("doc_name", "未命名"),
        st.session_state.idx,
        st.session_state.q_idx,
        st.session_state.done,
        st.session_state.score,
        st.session_state.streak,
        len(st.session_state.chapters),
    )
    save_chapters(st.session_state.doc_key, st.session_state.chapters)


def render_settings():
    """侧边栏设置区：本地/在线 API 切换。"""
    st.sidebar.title("⚙️ 设置")

    # 🔥 添火：设置栏内的下拉项，仅放开关，不改变功能与按钮样式
    with st.sidebar.expander("🔥 添火", expanded=True):
        cur = settings.get_fire()
        on = st.toggle("🔥 添火", value=cur, key="fire_toggle")
        if on != cur:
            settings.set_fire(on)
            # 切换后清空已生成缓存，确保重新生成对应模式
            st.session_state.generated = {}
            st.session_state.add_fire = on

    with st.sidebar.expander("大模型后端", expanded=True):
        provider = settings.get_provider()
        idx = 0 if provider == "ollama" else 1
        choice = st.radio("后端", ["本地 (Ollama)", "在线 (DeepSeek)"],
                          index=idx, key="provider_radio",
                          help="本地=离线不出本机；在线=DeepSeek API 更智能")
        new_provider = "ollama" if choice.startswith("本地") else "deepseek"
        if new_provider != provider:
            settings.set_provider(new_provider)
            st.toast("后端已切换，重启生效")
        if new_provider == "deepseek":
            key = st.text_input("DeepSeek API Key", value=settings.get_deepseek_key(),
                                type="password", key="dsk_key")
            model = st.text_input("模型名", value=settings.get_deepseek_model(),
                                  key="dsk_model")
            if st.button("保存 DeepSeek 配置"):
                settings.set_deepseek_key(key)
                settings.set_deepseek_model(model)
                st.success("已保存，重启生效")
        st.caption("当前：\u2191 修改后需重启应用生效")

    with st.sidebar.expander("🔍 答案解析来源", expanded=True):
        src = settings.get_analysis_source()
        src_idx = 0 if src == "doc" else 1
        src_choice = st.radio("解析时是否允许联网",
                              ["仅限文档（离线）", "允许联网搜索"],
                              index=src_idx, key="src_radio",
                              help="仅限文档=纯离线、不泄露内容；允许联网=结合网络公开资料做补充解析")
        new_src = "doc" if src_choice.startswith("仅限") else "web"
        if new_src != src:
            settings.set_analysis_source(new_src)
            st.toast(f"答案解析来源已设为：{src_choice}")
        st.caption(f"当前：{src_choice}")


def render_shelf():
    """主页书架：微信读书式文档卡片。"""
    docs = list_docs()
    st.subheader("📚 我的书架")
    if not docs:
        st.info("书架为空，先在下方上传一个技术文档开始学习吧。")
    else:
        cols = st.columns(2)
        for i, d in enumerate(docs):
            col = cols[i % 2]
            with col:
                with st.container(border=True):
                    st.markdown(f"**{d['name']}**")
                    st.progress(d["percent"] / 100, text=f"进度 {d['percent']}%")
                    st.caption(f"已答 {d['done']} 题 · 得分 {d['score']} · 共 {d['total_chapters']} 章")
                    if st.button("📖 继续学习", key=f"cont_{d['key']}"):
                        # 从持久化进度恢复：重新解析章节
                        st.session_state.target_doc_key = d["key"]
                        st.session_state.load_shelf_doc = True
                        st.rerun()


def load_shelf_doc(key: str):
    """根据书架选中的文档，从本地持久化的章节内容恢复学习。"""
    from progress import _load
    data = _load()
    meta = data.get(key)
    if meta is None:
        st.warning("找不到该文档的进度记录。")
        return
    chapters = load_chapters(key)
    if not chapters:
        st.warning("找不到该文档的章节缓存，请重新上传原文档。")
        return
    _init_state(key, chapters)
    st.session_state.doc_name = meta.get("name", "未命名")


def render_doc_input():
    """新建文档：上传/粘贴入口。"""
    with st.expander("📄 新建文档（上传文件或粘贴文本）", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.file_uploader(
                "上传文档（.md/.txt/.pdf/.docx，可多选）",
                type=["md", "txt", "pdf", "docx"],
                accept_multiple_files=True, key="uploaded_files")
        with col2:
            st.text_area("或粘贴文档内容", height=120, key="pasted_text")
        if st.button("➕ 生成并加入书架", type="primary"):
            chapters = parse_source()
            if not chapters:
                st.warning("未获取到内容，请上传文件或粘贴文本。")
            else:
                key = doc_key_of(chapters)
                _init_state(key, chapters)
                st.session_state.doc_name = doc_name_of(chapters)
                _persist()
                st.success(f"已加入书架：{st.session_state.doc_name}")
                st.rerun()


def parse_source() -> list:
    uploaded = st.session_state.get("uploaded_files") or []
    pasted = (st.session_state.get("pasted_text") or "").strip()
    chapters = []
    files = list(uploaded) + ([None] if pasted else [])
    for fi, uf in enumerate(uploaded):
        try:
            with st.status(f"正在解析 {uf.name}（{len(uf.getvalue())/1024/1024:.1f} MB）…",
                           expanded=False) as s:
                text = parse_uploaded(uf.getvalue(), uf.name)
                n = 0
                for ch in split_into_chapters(text):
                    ch["source"] = uf.name
                    chapters.append(ch)
                    n += 1
                s.update(label=f"已解析 {uf.name}：{n} 个章节")
            if len(uploaded) > 1:
                st.progress((fi + 1) / len(uploaded))
        except Exception as e:
            st.error(f"解析 {uf.name} 失败：{e}")
    if pasted:
        for ch in split_into_chapters(pasted):
            ch["source"] = "粘贴文本"
            chapters.append(ch)
    if len(chapters) > 80:
        st.warning(f"文档较大，已切分为 {len(chapters)} 个章节（含合并保护）。"
                   "首次进入每章会按需生成讲解与题目，可能稍慢，请耐心等待。")
    return chapters


def get_lesson(idx: int) -> dict:
    if idx in st.session_state.generated:
        return st.session_state.generated[idx]
    ch = st.session_state.chapters[idx]
    add_fire = settings.get_fire()
    try:
        with st.spinner(f"正在为「{ch['title']}」生成讲解与题目（约 10-30 秒）…"):
            lesson = build_lesson(ch["title"], ch["content"], ch["source"], n_quiz=3,
                                  use_cache=True, add_fire=add_fire)
    except Exception as e:
        return {"title": ch["title"], "source": ch["source"],
                "explain": f"⚠️ 本章生成失败：{e}\n\n可能是内容过长导致本地模型超时。"
                           "可尝试：① 在「设置」切换为 DeepSeek 后端；② 拆分文档为更小的章节后重试。",
                "quiz": []}
    st.session_state.generated[idx] = lesson
    return lesson


def render_sidebar():
    st.sidebar.title("📚 学习导航")
    chapters = st.session_state.chapters
    st.sidebar.metric("章节数", len(chapters))
    st.sidebar.metric("已答", st.session_state.done)
    st.sidebar.metric("得分", st.session_state.score)
    st.sidebar.metric("🔥 连击", st.session_state.streak)
    st.sidebar.metric("错题本", wrong_count())
    st.sidebar.divider()
    for i, ch in enumerate(chapters):
        label = f"{i+1}. {ch['title']}"
        if st.sidebar.button(label, key=f"nav{i}", use_container_width=True):
            st.session_state.idx = i
            st.session_state.q_idx = 0
            _persist()
            st.rerun()


def render_lesson():
    chapters = st.session_state.chapters
    if not chapters:
        return
    idx = st.session_state.idx
    if idx >= len(chapters):
        render_report()
        return

    lesson = get_lesson(idx)
    st.progress((idx + 1) / max(len(chapters), 1))
    st.header(f"第 {idx+1}/{len(chapters)} 章：{lesson['title']}")
    st.caption(f"来源：{lesson['source']}")

    with st.expander("📖 本章讲解", expanded=True):
        render_text(lesson["explain"])

    with st.expander("💬 有疑问？点此答疑（基于当前文档）"):
        q = st.text_input("输入你的问题", key=f"tq{idx}")
        if st.button("提问", key=f"tqa{idx}") and q.strip():
            with st.spinner("检索当前文档中…"):
                ans, sources, kind = ask_tutor(q, st.session_state.chapters)
            st.info("💡 答疑：")
            st.info(SOURCE_BADGE.get(kind, ""))
            render_text(ans)
            if sources:
                st.caption("来源：" + ", ".join(sources))
            if st.button("仍未掌握？记入错题本", key=f"tw{idx}"):
                wrong_add("tutor", lesson["title"],
                          {"question": q, "answer": ans, "sources": sources})
                st.toast("已记入错题本")

    quiz = lesson["quiz"]
    if not quiz:
        st.info("本章未生成题目（内容过短或生成失败）。")
        if st.button("下一章 →", key=f"nx{idx}"):
            st.session_state.idx += 1
            st.session_state.q_idx = 0
            _persist()
            st.rerun()
        return

    q_idx = st.session_state.q_idx
    if q_idx >= len(quiz):
        if st.button("✅ 完成本章，下一章 →", key=f"nx{idx}"):
            st.session_state.idx += 1
            st.session_state.q_idx = 0
            _persist()
            st.rerun()
        return

    q = quiz[q_idx]
    st.subheader(f"❓ 第 {q_idx+1}/{len(quiz)} 题")
    render_text(q["question"])

    # 已作答：展示解析 + 深度解析 + 下一题，不再即时跳题，给用户消化空间
    if st.session_state.get("answered") == (idx, q_idx):
        _render_answer_feedback(idx, q_idx, q, lesson)
        if st.button("➡️ 下一题", key=f"nxq{idx}_{q_idx}"):
            st.session_state.answered = None
            st.session_state.q_idx += 1
            _persist()
            st.rerun()
        return

    choice = st.radio("选择：", list(q["options"].keys()),
                      format_func=lambda k: f"{k}. {q['options'][k]}",
                      key=f"opt{idx}_{q_idx}")
    if st.button("提交", key=f"sub{idx}_{q_idx}"):
        correct = grade(choice, q)
        st.session_state.done += 1
        if correct:
            st.session_state.score += 1
            st.session_state.streak += 1
        else:
            st.session_state.streak = 0
            wrong_add("quiz", lesson["title"], q)
        # 记录作答状态与用户所选，供反馈屏使用
        st.session_state[f"_sel{idx}_{q_idx}"] = choice
        st.session_state.answered = (idx, q_idx)
        st.session_state.pop(f"opt{idx}_{q_idx}", None)
        st.rerun()


def _render_answer_feedback(idx: int, q_idx: int, q: dict, lesson: dict):
    """提交后的反馈屏：判分结论 + 文档解析 + 可联网的深度解析。"""
    if grade(st.session_state.get(f"_sel{idx}_{q_idx}", ""), q):
        st.success("✅ 答对！")
    else:
        st.error(f"❌ 正确答案：{q['answer']}")

    # 基于原文的知识点分析（局限文档，离线）
    with st.expander("📑 本题解析（基于文档）", expanded=True):
        st.info(SOURCE_BADGE["doc"])
        if q.get("explanation"):
            render_text(q["explanation"])
        else:
            st.write("（模型未给出解析）")

    # 深度解析：按设置走「文档」或「联网/模型」
    src = settings.get_analysis_source()
    label = "📚 文档深度解析" if src == "doc" else "🌐 联网深度解析"
    if st.button(label, key=f"deep{idx}_{q_idx}"):
        with st.spinner("正在生成深度解析（约 10-20 秒）…"):
            if src == "doc":
                ans, sources, kind = ask_tutor(
                    f"请结合文档，深入分析这道题涉及的知识点与易错点：{q['question']}",
                    st.session_state.chapters)
            else:
                ans, sources, kind = ask_web(f"{lesson['title']}：{q['question']} 相关知识")
        with st.expander(label + " 结果", expanded=True):
            # 醒目标识答案来源：文档 / 联网 / 模型推理，非文档内容必标
            badge = SOURCE_BADGE.get(kind, "")
            if kind == "doc":
                st.info(badge)
            else:
                st.warning("⚠️ " + badge + " —— 以下内容可能超出本文档，仅供参考、请自行核实")
            render_text(ans)
            if sources:
                st.caption("参考来源：" + "；".join(str(s) for s in sources[:5]))


def render_report():
    st.balloons()
    st.title("🎓 学习报告")
    done = st.session_state.done
    score = st.session_state.score
    pct = round(100 * score / done, 1) if done else 0
    st.metric("正确率", f"{pct}%")
    st.write(f"共答题 {done} 道，答对 {score} 道。")
    if pct >= 90:
        st.success("🏆 优秀！已掌握核心内容。")
    elif pct >= 70:
        st.info("👍 良好，建议复习错题。")
    else:
        st.warning("📚 需要加强，去错题本重练吧。")

    wc = wrong_count()
    st.subheader(f"📑 错题本（{wc} 条）")
    if wc:
        for w in all_items()[-10:]:
            if w["kind"] == "quiz":
                st.write(f"· [{w['title']}] {w['item']['question']} → {w['item']['answer']}")
            else:
                st.write(f"· [{w['title']}] 答疑未掌握：{w['item']['question']}")
    else:
        st.success("错题本已清空！")

    if st.button("↩️ 返回书架"):
        st.session_state.pop("chapters", None)
        st.session_state.pop("doc_key", None)
        st.rerun()


def main():
    st.set_page_config(page_title="文档学习机器人", layout="wide")
    st.title("🤖 技术文档学习机器人")

    if "add_fire" not in st.session_state:
        st.session_state.add_fire = settings.get_fire()
    if "answered" not in st.session_state:
        st.session_state.answered = None
    render_settings()

    # 添火开启后，在主区域右侧提示
    if st.session_state.add_fire:
        _, right = st.columns([6, 2])
        with right:
            st.success("🔥 已放入人性")

    # 处理书架选择：重新加载目标文档
    if st.session_state.get("load_shelf_doc"):
        load_shelf_doc(st.session_state.get("target_doc_key", ""))
        st.session_state.load_shelf_doc = False

    if "chapters" in st.session_state and st.session_state.get("doc_key"):
        # 学习中 → 显示章节 + 侧边栏 + 返回书架
        render_sidebar()
        with st.container():
            if st.button("📚 返回书架"):
                _persist()
                st.session_state.pop("chapters", None)
                st.session_state.pop("doc_key", None)
                st.rerun()
        render_lesson()
    else:
        # 主页：书架 + 新建文档
        render_shelf()
        st.divider()
        render_doc_input()


if __name__ == "__main__":
    main()
