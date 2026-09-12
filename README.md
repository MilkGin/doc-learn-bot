# 技术文档学习机器人 · doc-learn-bot

> 把技术文档变成一门可以「学完」的课：**分章讲解 → 即时选择题 → 答疑 → 错题本 → 学习报告**。
> 默认**完全离线运行**（本地 Ollama），数据不出本机；也支持一键切到在线 API。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![LLM](https://img.shields.io/badge/LLM-Local%20First-orange)

---

## 这是什么

不是「文档问答」，是**带教学闭环的文档学习系统**。

上传一份技术文档（比如 PLC 手册、设备说明书、内部 SOP），它会：

1. 按文档的**标题层级**拆成章节
2. 逐章生成**讲解**和**即时选择题**
3. 你在章节内随时**提问**，回答只来自该文档、**可溯源**
4. 答错的题和没解决的疑问自动进**错题本**，跨会话保存
5. 学完给出**报告和得分**，关掉浏览器再打开能**断点续学**

> 面向的场景是**工业现场 / 售后客服的文档培训**——那里通常没有稳定外网，机器配置也有限。

---

## 为什么做成本地优先

| | 本地 Ollama（默认） | 在线 API（可选） |
|---|---|---|
| 数据 | **不出本机** | 上传到第三方 |
| 网络 | 拉完模型后**完全离线可用** | 依赖网络 |
| 成本 | 0 | 按量计费 |
| 硬件 | 需要一块 GPU | 不需要 |

工业现场的文档往往涉及设备参数和工艺细节，**不适合往外传**——这是选择本地优先的根本原因。

---

## 架构

```
文档（md / txt / pdf / docx）
        │
        ├─ ingest.py         固定字符切分（800 字 / 重叠 100）→ ChromaDB 向量库
        │                                        ↑
        │                             embedding.py（nomic-embed-text）
        │
        └─ chapter_split.py  按标题层级拆「章节」（保留语义结构，用于教学）
                    │
                    ├─ course_builder.py   逐章生成「讲解 + 选择题」
                    │                      严格 JSON、绑定原文 source、本地缓存
                    ├─ tutor.py            章节内答疑（切块 → 余弦相似度检索 → 生成）
                    ├─ wrong_book.py       错题本（JSON 持久化）
                    └─ progress.py         进度（doc_key → 断点续学）
                                │
                        app_learn.py（Streamlit 网页）
```

**两种切分策略并存，是有意的**：

- 检索要**均匀粒度** → `ingest.py` 固定字符切分
- 教学要**语义完整** → `chapter_split.py` 按标题层级切分

---

## 功能

| 模块 | 说明 |
|---|---|
| `app_learn.py` | Streamlit 网页原型：**书架**（像微信读书一样列卡片，点击继续学）、上传文档或粘贴文本、章节导航、即时选择题（带连击）、答疑、错题、学习报告 |
| `chapter_split.py` | 按 Markdown 标题层级把文档拆成章节，保留语义结构 |
| `course_builder.py` | 逐章生成讲解与选择题；**严格 JSON 输出**、绑定原文 `source` 便于溯源、结果**本地缓存**避免重复消耗算力；含「添火」深度模式（面向更高阶读者，输出形式化记号 / 定理 / 推导） |
| `tutor.py` | 章节内答疑。两种检索源：基于当前会话已解析的章节（支持刚上传的文档）、或基于 `docs/` 建好的全局向量库。**回答只来自给定片段，不存在即答「文档中未提及」** |
| `wrong_book.py` | 错题本：JSON 持久化，零数据库依赖，支持记录 / 导出 / 随机重练 |
| `progress.py` | 进度：以**文档内容哈希前 10 位**为 key，记录章节、题号、已答、得分、连击、上次学习时间 → 断点续学 |
| `llm_client.py` | 统一的模型调用层。所有生成（讲解 / 出题 / 答疑）都走这里，**改一行配置即可在本地 Ollama 与在线 DeepSeek 之间切换** |
| `embedding.py` | 本地嵌入（默认 `nomic-embed-text`），保证向量化也离线 |
| `ingest.py` | 文档加载与索引。**`.docx` 用内置 `zipfile` 解析 `word/document.xml`，零额外依赖**，并把 Word 标题样式转成 Markdown 层级，让下游章节切分能正确工作 |
| `formula_render.py` | LaTeX → KaTeX 公式渲染；**优先使用本地 `static/katex`**（离线可用），资源缺失时回退 CDN |
| `settings.py` | 界面上直接切换模型后端并写回 `.env`，不用手动改文件 |
| `start.bat` | 一键启动：拉起 Ollama、按需建索引、打开浏览器 |

---

## 快速开始

```bash
# 1. 模型（仅首次需联网，之后完全离线）
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 2. 依赖
pip install -r requirements.txt

# 3. 启动
streamlit run app_learn.py
# 或者直接双击 start.bat（会自动拉起 Ollama 并打开浏览器）
```

浏览器打开终端提示的地址（默认 http://localhost:8501 ）。

**想换文档**：在界面上直接上传 / 粘贴即可；如果用 `docs/` 目录，改完重跑 `python ingest.py` 重建索引。

---

## 配置

复制 `.env.example` 为 `.env`：

| 配置项 | 说明 | 默认 |
|---|---|---|
| `LLM_PROVIDER` | `ollama`（本地离线）/ `deepseek`（在线） | `ollama` |
| `LLM_MODEL` | 生成模型 | `qwen2.5:7b` |
| `EMBED_MODEL` | 嵌入模型 | `nomic-embed-text` |
| `TOP_K` | 每次召回片段数 | `4` |
| `DOCS_DIR` | 文档目录 | `./docs` |
| `CHROMA_DIR` | 向量库目录 | `./chroma_db` |
| `OLLAMA_BASE_URL` | Ollama 地址（也支持 `OLLAMA_HOST` / `OLLAMA_PORT` 环境变量） | `http://localhost:11434` |
| `DEEPSEEK_API_KEY` | 切换到在线后端时必填 | — |

> 机器配置较低可换 `qwen2.5:3b`；嵌入模型可换 `bge-m3`。

---

## 工程上踩过的坑（供参考）

**1. `.docx` 解析：段落边界会丢**

早期实现只抽取 `<w:t>` 里的文本，结果整篇文档被压成**没有换行的一行**，下游章节切分完全失效。
改为按 `</w:p>` 段落边界切分并补换行，同时识别 `w:pStyle` 把 Word 标题样式转成 Markdown 的 `#`，章节切分才正常工作。

**2. 本地模型会「无限挂起」**

Ollama 对超大请求可能长时间不返回，UI 表现就是「卡死」。
解法：`ollama.Client` 显式设 `timeout=300`。

**3. 检索片段叠加会 500**

多段上下文拼起来容易超过 Ollama 默认的 2048 上下文窗口，直接报错。
解法：请求时显式设 `options={"num_ctx": 8192}`。

**4. 题目重复生成浪费算力**

工业现场机器配置低，每次进章节都重新出题不可接受。
解法：生成结果按章节本地缓存，命中即复用；同时保留原文 `source` 供溯源。

---

## Roadmap

- [ ] **文档语义变更检测**：保存章节的向量快照，文档更新后对比新旧向量，只提示「含义发生变化」的章节需要重学（区别于文本 diff——改写措辞但语义不变的章节不打扰用户）
- [ ] 命令行版本（无 GUI 环境 / 批量处理场景）
- [ ] 拆题与试卷导出

---

## 已知限制

- 章节切分依赖文档的标题结构；纯扫描版 PDF 或没有标题层级的文档效果会下降
- 生成质量直接受本地模型能力限制；7B 模型在长文档推理上仍有明显上限
- 当前为原型阶段，未做并发与多用户隔离

---

## 许可证

[Apache License 2.0](./LICENSE) © 2026 MilkGin
