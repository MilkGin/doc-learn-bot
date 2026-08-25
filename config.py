import os

from dotenv import load_dotenv

load_dotenv()

# 自适应 Ollama 服务地址：
# 优先系统环境变量 OLLAMA_HOST，其次 OLLAMA_PORT（形如 127.0.0.1:11435 或 11435），
# 再次 .env 中的 OLLAMA_BASE_URL，最后回退默认 11434。
_env_host = os.getenv("OLLAMA_HOST")
_env_port = os.getenv("OLLAMA_PORT")
if _env_host:
    OLLAMA_BASE_URL = _env_host if _env_host.startswith("http") else "http://" + _env_host
elif _env_port:
    OLLAMA_BASE_URL = (
        "http://" + _env_port if ":" in _env_port else "http://localhost:" + _env_port
    )
else:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# 生成回答用的大模型（Ollama 本地模型名）
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")

# ---- 大模型后端切换：ollama（本地离线） / deepseek（线上 API） ----
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# ---- DeepSeek 线上 API 配置（LLM_PROVIDER=deepseek 时生效） ----
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# 文档向量化用的嵌入模型
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# 技术文档目录
DOCS_DIR = os.getenv("DOCS_DIR", "./docs")

# 向量库持久化目录
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

# 每次检索召回的片段数量
TOP_K = int(os.getenv("TOP_K", "4"))

# ---- Zendesk 客服系统同步 ----
ZENDESK_SUBDOMAIN = os.getenv("ZENDESK_SUBDOMAIN", "")
ZENDESK_EMAIL = os.getenv("ZENDESK_EMAIL", "")
ZENDESK_API_TOKEN = os.getenv("ZENDESK_API_TOKEN", "")
# 命中这些 tag 的工单视为「主管授权·超 SOP 额外奖励」，归入 exception 轨（仅查询用，不进人类文档）
ZENDESK_AUTHORIZED_TAGS = [
    t.strip() for t in os.getenv("ZENDESK_AUTHORIZED_TAGS", "supervisor_exception,approved_bonus").split(",") if t.strip()
]
# 命中这些 tag / 高优先级 视为「难搞客户」
ZENDESK_DIFFICULT_TAGS = [
    t.strip() for t in os.getenv("ZENDESK_DIFFICULT_TAGS", "escalated,repeat_complaint,vip").split(",") if t.strip()
]
# 导出数据存放目录
ZENDESK_EXPORT_DIR = os.getenv("ZENDESK_EXPORT_DIR", "./data")
