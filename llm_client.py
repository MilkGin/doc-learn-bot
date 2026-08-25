"""统一的大模型调用封装层：支持「本地 Ollama」与「DeepSeek 线上 API」两种后端。

切换方式：在 .env 设置
    LLM_PROVIDER=ollama      # 本地离线（默认）
    LLM_PROVIDER=deepseek    # DeepSeek 线上 API
    DEEPSEEK_API_KEY=sk-xxx  # 使用 deepseek 时必填

所有生成模块（讲解、出题、答疑、RAG 回答）都应通过这里的 generate() 调用，
这样只需改一个配置即可切换后端，代码无感知。
"""

import os

import ollama
import requests

from config import LLM_MODEL, OLLAMA_BASE_URL, LLM_PROVIDER, DEEPSEEK_API_KEY, DEEPSEEK_MODEL

_DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


def generate(prompt: str, system: str = "") -> str:
    """根据配置的 LLM_PROVIDER 生成文本，返回字符串。

    prompt: 用户侧提示词（含文档片段）
    system: 可选系统提示词（DeepSeek 支持；Ollama 会拼到 prompt 前）
    """
    provider = LLM_PROVIDER
    if provider == "deepseek":
        return _deepseek(prompt, system)
    return _ollama(prompt, system)


def _ollama(prompt: str, system: str) -> str:
    if not system:
        system = "你是一个严谨的技术文档助手，回答要基于给定资料，不要编造。"
    full = f"{system}\n\n{prompt}"
    client = ollama.Client(host=OLLAMA_BASE_URL)
    resp = client.generate(model=LLM_MODEL, prompt=full, stream=False)
    return resp["response"] if isinstance(resp, dict) else resp.response


def _deepseek(prompt: str, system: str) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("使用 DeepSeek 后端需在 .env 配置 DEEPSEEK_API_KEY")
    if not system:
        system = "你是一个严谨的技术文档助手，回答要基于给定资料，不要编造。"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "stream": False,
    }
    resp = requests.post(_DEEPSEEK_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print("当前后端:", LLM_PROVIDER, "| 模型:", LLM_MODEL if LLM_PROVIDER == "ollama" else DEEPSEEK_MODEL)
    print(generate("用一句话介绍 PLC。"))
