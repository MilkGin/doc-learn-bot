"""教学编排层：基于章节内容生成「讲解 + 选择题」，并支持判分。

设计原则（沿用 offline-doc-qa 的离线/不出本机理念）：
- 所有生成均走本地 Ollama，不调用任何云端 API。
- 选择题严格 JSON 输出，并绑定原文 source 以便溯源。
- 题目生成后缓存到本地，避免重复消耗算力（工业现场机器配置低时关键）。
"""

import json
import os
import re

from chapter_split import build_course
from llm_client import generate


# ---------- 「添火」深度模式 ----------

# add_fire=True 时：面向博士研究生，讲解严谨深入（形式化记号/定理/推导），
# 选择题考查细节、推导步骤、反例、定理假设与适用范围，干扰项更具迷惑性。
_FIRE_INSTRUCTION = (
    "学员背景：博士研究生，已具备扎实的理论基础与形式化训练。讲解要严谨深入，可直接使用原文中的"
    "形式化记号、定理与推导，强调证明思路、边界条件、与已有理论的联系及可能的推广；"
    "选择题考查对细节、推导步骤、反例、定理假设与适用范围的精确把握，干扰项需具有迷惑性"
    "（如混淆相似定义、忽略前提条件）。"
)

# add_fire=True（🔥 添火）：科研助手人格，研究生水平、尽可能数学化、实地考察、不啰嗦
_FIRE_INSTRUCTION = (
    "你是我的科研助手，负责在研究生水平上尽可能数学化问题，我的科研特点是尽可能数学化和"
    "尽可能实地考察，不要长篇大论。"
)

# add_fire=False 时（默认）：面向本科低年级，通俗易懂、配生活化类比。
_BASE_INSTRUCTION = (
    "学员背景：本科低年级，刚接触该领域。讲解要通俗易懂，多用生活化类比和直观例子，"
    "避免一上来就堆砌抽象定义；选择题考查对基础概念、术语含义、主要结论的直观理解。"
)


# ---------- 讲解生成 ----------

_EXPLAIN_TMPL = """你是一个工业技术文档讲师。请用有条理的语言讲解下面这段文档内容。
{instruction}
不要编造文档中没有的事实。

【章节标题】{title}
【文档原文】
{content}

请输出讲解（3-6 个要点，中文，简洁）："""


def explain_chapter(title: str, content: str, add_fire: bool = False) -> str:
    instruction = _FIRE_INSTRUCTION if add_fire else _BASE_INSTRUCTION
    prompt = _EXPLAIN_TMPL.format(title=title, content=content, instruction=instruction)
    return generate(prompt)


# ---------- 选择题生成 ----------

_QUIZ_TMPL = """你是一个出题专家。请严格根据下面的【文档原文】出 {n} 道单选题。
要求：
1. 每题只有一个正确选项，其余 3 个为合理干扰项（不要明显错误）。
2. 题干和选项必须基于原文事实，不能凭空捏造。
3. 必须给出正确选项字母和简要解析（引用原文要点）。
4. 只输出 JSON，不要任何额外说明文字。

{instruction}

输出格式（严格 JSON 数组）：
[
  {{
    "question": "题干？",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "answer": "A",
    "explanation": "解析：根据原文……"
  }}
]

【文档原文】
{content}
"""


def _extract_json(text: str):
    """从模型输出里尽量抠出 JSON 数组。"""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def generate_quiz(content: str, n: int = 3, add_fire: bool = False) -> list:
    """生成 n 道选择题，返回校验后的题目列表。"""
    instruction = _FIRE_INSTRUCTION if add_fire else _BASE_INSTRUCTION
    prompt = _QUIZ_TMPL.format(n=n, content=content, instruction=instruction)
    raw = generate(prompt)
    items = _extract_json(raw)
    if not items:
        return []
    valid = []
    for it in items:
        # 校验：必须有题干、4选项、唯一答案
        if not all(k in it for k in ("question", "options", "answer")):
            continue
        if not isinstance(it["options"], dict) or len(it["options"]) != 4:
            continue
        if it["answer"] not in it["options"]:
            continue
        valid.append(it)
    return valid


# ---------- 判分 ----------

def grade(user_answer: str, question: dict) -> bool:
    return user_answer.strip().upper() == question["answer"].strip().upper()


# ---------- 编排主流程（带本地缓存）----------

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "course_cache")


def _cache_path(source: str, title: str, add_fire: bool) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    tag = "fire" if add_fire else "base"
    safe = re.sub(r"[^\w一-鿿]", "_", f"{source}__{title}__{tag}")
    return os.path.join(_CACHE_DIR, safe + ".json")


def build_lesson(title: str, content: str, source: str, n_quiz: int = 3,
                 use_cache: bool = True, add_fire: bool = False):
    """为单章生成「讲解 + 题目」，可选走缓存。

    防护：内容过短时返回空 lesson（quiz=[]），避免把空内容丢给模型导致编造（幻觉）题目。
    """
    if len(content.strip()) < 50:
        return {"title": title, "source": source, "explain": "(章节内容过短，跳过)", "quiz": []}
    cp = _cache_path(source, title, add_fire)
    if use_cache and os.path.exists(cp):
        with open(cp, "r", encoding="utf-8") as f:
            return json.load(f)
    lesson = {
        "title": title,
        "source": source,
        "explain": explain_chapter(title, content, add_fire),
        "quiz": generate_quiz(content, n_quiz, add_fire),
    }
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(lesson, f, ensure_ascii=False, indent=2)
    return lesson


def build_full_course(use_cache: bool = True, n_quiz: int = 3):
    """构建整门课（所有章节的讲解+题目）。"""
    course = build_course()
    lessons = []
    for ch in course:
        lessons.append(build_lesson(ch["title"], ch["content"], ch["source"], n_quiz, use_cache))
    return course, lessons


if __name__ == "__main__":
    course, lessons = build_full_course()
    print(f"课程含 {len(lessons)} 章：")
    for i, ls in enumerate(lessons):
        print(f"\n=== 第 {i+1} 章：{ls['title']} ===")
        print(ls["explain"][:200] + "…")
        print(f"[题目数] {len(ls['quiz'])}")
