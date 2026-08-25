# 离线技术文档问答机器人（Offline Doc QA）

一个**完全本地运行**的 RAG（检索增强生成）问答机器人：把你的技术文档（Markdown / TXT / PDF）
放进去，用本地大模型（Ollama）进行向量检索 + 生成回答。**数据不出本机，不调用任何云端 API。**

> 📄 许可证：[Apache License 2.0](./LICENSE) © 2026 MilkGin

## 架构流程

```
技术文档 → 切分 → 本地向量化(nomic-embed-text) → ChromaDB
                                              ↓
用户问题 → 向量检索(top-k) → 拼接上下文 → 本地大模型(qwen2.5) → 回答
```

## 1. 环境准备

- 安装 **Python 3.10+**
- 安装 **Ollama**：https://ollama.com （桌面版安装后会自动运行 `ollama serve`）
- 拉取所需模型（首次需联网下载，之后完全离线运行）：

  ```bash
  ollama pull qwen2.5:7b
  ollama pull nomic-embed-text
  ```

  > 机器配置较低可换用 `qwen2.5:3b`；嵌入模型也可换 `bge-m3` 等。改完记得同步修改 `.env`。

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 3. 放入技术文档

把你的技术文档放进 `docs/` 目录（支持 `.md` / `.txt` / `.pdf` / `.docx`，可建子目录分类）。`.docx` 用内置库解析，无需额外依赖。

```bash
mkdir docs
# 把文档复制/拖进 docs/ 即可
```

## 4. 建立索引

```bash
python ingest.py
```

脚本会把文档切分并写入本地向量库 `chroma_db/`。**更换/增删文档后重新运行此命令即可。**

## 5. 开始问答

- **命令行模式：**

  ```bash
  python query.py
  ```

- **网页模式（推荐）：**

  ```bash
  streamlit run app.py
  ```

  浏览器打开终端提示的本地地址（默认 http://localhost:8501 ）即可对话。

## 6. 自定义配置

复制 `.env.example` 为 `.env` 后可调整：

| 配置项            | 说明                              |
| ----------------- | --------------------------------- |
| `LLM_MODEL`       | 生成回答用的大模型                |
| `EMBED_MODEL`     | 向量化用的嵌入模型                |
| `DOCS_DIR`        | 技术文档目录（支持 .md/.txt/.pdf/.docx） |
| `CHROMA_DIR`      | 向量库持久化目录                  |
| `TOP_K`           | 每次检索召回的片段数量            |
| `OLLAMA_BASE_URL` | Ollama 服务地址（默认本机 11434） |

## 常见问题

- **报错 model not found**：先执行 `ollama pull <模型名>` 拉取对应模型。
- **中文回答质量差**：换更大的模型，如 `qwen2.5:14b`。
- **想换文档**：直接修改 `docs/` 后重新运行 `python ingest.py`。
- **完全离线**：模型拉取完成后断网也能正常使用，所有计算在本地完成。

---

# 学习机器人（教学原型）

在「离线问答」之上扩展的**分章节教学 + 即时选择题 + 答疑 + 错题 + 打分**原型，
面向工业界教育 / 售后客服文档培训场景，同样完全离线。

## 模块一览

| 文件 | 作用 |
| ---- | ---- |
| `chapter_split.py` | 按标题层级把文档拆成章节 |
| `course_builder.py` | 每章生成讲解 + 选择题（本地缓存、题目溯源） |
| `learn.py` | 命令行学习主引擎（讲解→答题→打分+错题本） |
| `tutor.py` | 章节内答疑（基于文档 RAG，可溯源） |
| `wrong_book.py` | 跨会话错题本（JSON 持久化） |
| `doc_watch.py` | 文档语义变更检测（向量相似度，非文字 diff） |
| `review.py` | 错题重练（做对即移除） |
| `llm_client.py` | 统一大模型调用封装（Ollama 本地 / DeepSeek 线上切换） |
| `app_learn.py` | **网页原型机（Streamlit）**：章节导航/进度/连击/答疑/错题/更新弹窗/报告 |
| `start.bat` | **一键启动脚本**（自动拉起 Ollama + 打开网页） |

## 一键启动（推荐）

双击 `start.bat` 即可：自动启动 Ollama、建立索引（首次）、打开浏览器进入学习界面。

## 快速开始

```bash
# 1. 准备模型（仅首次需联网，本地 Ollama 后端需要）
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 2. 放入技术文档到 docs/，建索引
python ingest.py

# 3. 启动网页学习原型
streamlit run app_learn.py
```

浏览器打开提示的地址（默认 http://localhost:8501 ）即可：
章节讲解 → 即时选择题（带连击） → 答疑 → 错题自动记录 → 学完出报告。
文档更新后重跑 `python ingest.py`，再次启动会弹「再学习提醒」（仅含义变更的章节）。

## 大模型后端切换（Ollama 本地 / DeepSeek 线上）

编辑 `.env`：

```ini
# ollama  = 本地离线模型（默认，数据不出本机）
# deepseek = DeepSeek 线上 API（需填 DEEPSEEK_API_KEY）
LLM_PROVIDER=ollama

# DeepSeek 线上 API 配置（LLM_PROVIDER=deepseek 时生效）
DEEPSEEK_API_KEY=sk-你的key
DEEPSEEK_MODEL=deepseek-chat
```

- 改 `LLM_PROVIDER=deepseek` 并填入 `DEEPSEEK_API_KEY` 即切到线上 DeepSeek（无需本地 GPU，出题/答疑更智能，但有网络依赖与调用费用）。
- 改回 `LLM_PROVIDER=ollama` 即回本地离线。
- 讲解、出题、答疑、RAG 回答**全部**经 `llm_client.py` 统一封装，切换只改这一处配置。

## 命令行版

```bash
python learn.py      # 学完全程 + 打分 + 错题本
python review.py     # 专门重练错题
python doc_watch.py  # 手动检测文档语义更新
```
