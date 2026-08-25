"""设置持久化：读写 .env 中的大模型后端切换配置。

提供界面可调的：
- LLM_PROVIDER: ollama(本地) / deepseek(在线)
- DEEPSEEK_API_KEY: 在线 API Key
- DEEPSEEK_MODEL: 在线模型名
"""

import os
import re

_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")


def _read_env() -> dict:
    data = {}
    if os.path.exists(_ENV_PATH):
        with open(_ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                data[k.strip()] = v.strip()
    return data


def _write_env(data: dict):
    # 保留原文件注释，仅更新/追加指定键
    lines = []
    if os.path.exists(_ENV_PATH):
        with open(_ENV_PATH, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    # 更新已存在的键，记录更新过的键
    updated = set()
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k = s.split("=", 1)[0].strip()
        if k in data:
            lines[i] = f"{k}={data[k]}"
            updated.add(k)
    # 追加未出现过的键
    for k, v in data.items():
        if k not in updated:
            lines.append(f"{k}={v}")
    with open(_ENV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def get_provider():
    return _read_env().get("LLM_PROVIDER", "ollama")


def get_deepseek_key():
    return _read_env().get("DEEPSEEK_API_KEY", "")


def get_deepseek_model():
    return _read_env().get("DEEPSEEK_MODEL", "deepseek-chat")


def get_fire():
    """「添火」深度模式开关，返回 bool。"""
    return _read_env().get("FIRE", "false").lower() == "true"


def set_fire(on: bool):
    data = _read_env()
    data["FIRE"] = "true" if on else "false"
    _write_env(data)


def get_analysis_source():
    """答案解析来源：doc=仅限文档；web=允许联网搜索。"""
    return _read_env().get("ANALYSIS_SOURCE", "doc")


def set_analysis_source(s: str):
    data = _read_env()
    data["ANALYSIS_SOURCE"] = s
    _write_env(data)


def set_provider(provider: str):
    data = _read_env()
    data["LLM_PROVIDER"] = "deepseek" if provider == "deepseek" else "ollama"
    _write_env(data)


def set_deepseek_key(key: str):
    data = _read_env()
    data["DEEPSEEK_API_KEY"] = key
    _write_env(data)


def set_deepseek_model(model: str):
    data = _read_env()
    data["DEEPSEEK_MODEL"] = model
    _write_env(data)
