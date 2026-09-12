"""doc_watch 的自测脚本。

不依赖 pytest，直接运行：

    python test_doc_watch.py

测试分两层：
1. **比对逻辑**（确定性伪向量）：验证「未变 / 变更 / 新增 / 删除」四类判定正确。
2. **快照读写**（临时文件）：验证跨次调用的身份识别与 JSON 往返。
3. **真实嵌入**（可选）：本地 Ollama 可达时，用真实向量跑一遍端到端；
   不可达时自动跳过，不影响其余用例。
"""

import os
import shutil
import tempfile
import zlib

import doc_watch

DIM = 64


def stub_vec(text: str) -> list[float]:
    """确定性词袋伪向量：共用词越多 → 余弦相似度越高。

    用 crc32 而不是内置 hash()，保证跨进程稳定（避免 PYTHONHASHSEED 影响）。
    """
    vec = [0.0] * DIM
    for tok in str(text).replace("\n", " ").split():
        vec[zlib.crc32(tok.encode("utf-8")) % DIM] += 1.0
    return vec


def mk(title, content, source="demo.md"):
    return {"title": title, "level": 1, "content": content, "source": source}


# ---------- 用例 1：比对逻辑 ----------

def test_compare_classification():
    old = [mk("安装", "安装步骤 依赖 环境"),
           mk("配置", "配置项 说明 默认值"),
           mk("常见问题", "报错 排查 处理")]
    new = [mk("安装", "安装步骤 依赖 环境"),           # 原样
           mk("配置", "配置项 说明 默认值 新增参数"),   # 内容变了
           mk("进阶用法", "高级 技巧 性能优化")]        # 新增；同时"常见问题"被删

    old_snap = [dict(c, vec=stub_vec(doc_watch.chapter_text(c))) for c in old]
    new_vecs = [stub_vec(doc_watch.chapter_text(c)) for c in new]

    rep = doc_watch.compare(old_snap, new, new_vecs)

    assert [c["title"] for c in rep["same"]] == ["安装"], rep["same"]
    assert [c["title"] for c in rep["changed"]] == ["配置"], rep["changed"]
    assert [c["title"] for c in rep["added"]] == ["进阶用法"], rep["added"]
    assert rep["removed"] == ["常见问题"], rep["removed"]
    assert rep["needs_relearn"] == [1, 2], rep["needs_relearn"]
    assert rep["is_update"] is True
    print("  [PASS] 四类判定正确（未变/变更/新增/删除）")


def test_rewritten_chapter_with_same_title():
    """标题没变、内容整段重写 → 应报「变更（同一节被重写）」，而不是「删除 + 新增」。"""
    old = [mk("3. 启动服务", "运行 start.bat 启动服务 默认监听 8501 端口"),
           mk("4. 常见问题", "报错 排查 端口占用 模型缺失 重试")]
    new = [mk("3. 启动服务", "数据备份 迁移 向量库 进度文件 嵌入维度 重新建索引"),
           mk("4. 常见问题", "报错 排查 端口占用 模型缺失 重试")]
    old_snap = [dict(c, vec=stub_vec(doc_watch.chapter_text(c))) for c in old]
    rep = doc_watch.compare(old_snap, new, [stub_vec(doc_watch.chapter_text(c)) for c in new])

    assert [c["title"] for c in rep["changed"]] == ["3. 启动服务"], rep["changed"]
    assert rep["changed"][0].get("reason"), "应带上「同一节被重写」的说明"
    assert rep["added"] == [], f"不应报成新增: {rep['added']}"
    assert rep["removed"] == [], f"不应报成删除: {rep['removed']}"
    print("  [PASS] 同标题内容重写 → 判为变更（而非删除+新增）")


def test_no_change_is_not_update():
    """只调整排版、语义不变 → 不应算作"更新"，避免打扰用户。"""
    old = [mk("安装", "安装步骤 依赖 环境")]
    new = [mk("安装", "安装步骤 依赖 环境")]
    old_snap = [dict(c, vec=stub_vec(doc_watch.chapter_text(c))) for c in old]
    rep = doc_watch.compare(old_snap, new, [stub_vec(doc_watch.chapter_text(c)) for c in new])
    assert rep["is_update"] is False, rep
    assert rep["needs_relearn"] == []
    print("  [PASS] 无变化时不标记为更新")


def test_dimension_mismatch_safe():
    """向量维度不一致（换过嵌入模型）时应安全返回 0，不能抛异常。"""
    assert doc_watch._cosine([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0
    assert doc_watch._cosine([], [1.0]) == 0.0
    assert doc_watch._cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    print("  [PASS] 维度不一致 / 零向量安全降级")


# ---------- 用例 2：快照读写 ----------

def test_snapshot_roundtrip():
    """第一次 check 返回 None（无历史），第二次能比对出变更。"""
    original_path = doc_watch._PATH
    original_embed = doc_watch.embed_chapters
    tmpdir = tempfile.mkdtemp()
    doc_watch._PATH = os.path.join(tmpdir, "snap.json")
    doc_watch.embed_chapters = lambda chapters: [stub_vec(doc_watch.chapter_text(c)) for c in chapters]
    try:
        v1 = [mk("安装", "安装步骤 依赖 环境"), mk("配置", "配置项 说明 默认值")]
        assert doc_watch.check(v1) is None, "首次应无历史可比"
        assert os.path.exists(doc_watch._PATH), "应写入快照"

        v2 = [mk("安装", "安装步骤 依赖 环境"), mk("配置", "配置项 说明 默认值 新增参数")]
        rep = doc_watch.check(v2)
        assert rep is not None, "第二次应能比对"
        assert rep["identity"] == "demo.md", rep["identity"]
        assert [c["title"] for c in rep["changed"]] == ["配置"], rep["changed"]

        # 身份识别不看内容：换一批内容相同的文件名，应命中同一份快照
        v3 = [mk("安装", "完全不同的内容 这里", source="other.md")]
        assert doc_watch.identity_of(v3) == "other.md"
        print("  [PASS] 快照往返 + 身份识别正确")
    finally:
        doc_watch._PATH = original_path
        doc_watch.embed_chapters = original_embed
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_degradation_without_model():
    """向量化不可用时 check() 必须返回 None，不能抛异常打断学习流程。"""
    original = doc_watch.embed_chapters
    doc_watch.embed_chapters = lambda chapters: None
    try:
        assert doc_watch.check([mk("安装", "内容")]) is None
        print("  [PASS] Ollama 不可用时静默降级")
    finally:
        doc_watch.embed_chapters = original


# ---------- 用例 3：真实嵌入（可选） ----------

def test_real_embedding():
    vectors = doc_watch.embed_chapters([
        mk("安装", "先安装 Python 3.10，再执行 pip install -r requirements.txt"),
        mk("配置", "在 .env 中设置 LLM_PROVIDER，本地默认走 Ollama"),
    ])
    if vectors is None:
        print("  [SKIP] Ollama 不可达，跳过真实嵌入测试")
        return
    assert len(vectors) == 2 and len(vectors[0]) > 0
    # 语义相近的两段文本，相似度应明显高于无关文本
    a = doc_watch.embed_chapters([mk("x", "安装 Python 依赖环境")])[0]
    b = doc_watch.embed_chapters([mk("y", "部署 Python 运行环境与依赖")])[0]
    c = doc_watch.embed_chapters([mk("z", "财务报表 季度营收 利润率分析")])[0]
    sim_related = doc_watch._cosine(a, b)
    sim_unrelated = doc_watch._cosine(a, c)
    assert sim_related > sim_unrelated, (sim_related, sim_unrelated)
    print(f"  [PASS] 真实嵌入可用（相关 {sim_related:.3f} > 无关 {sim_unrelated:.3f}）")


def main():
    cases = [
        test_compare_classification,
        test_rewritten_chapter_with_same_title,
        test_no_change_is_not_update,
        test_dimension_mismatch_safe,
        test_snapshot_roundtrip,
        test_degradation_without_model,
        test_real_embedding,
    ]
    failed = 0
    for case in cases:
        try:
            case()
        except AssertionError as e:
            failed += 1
            print(f"  [FAIL] {case.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [ERROR] {case.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(cases) - failed}/{len(cases)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
