import io
import json

from eval.run import CachedExtractor, make_policies, read_rows, run_version, summarize, _append_csv
from dataclasses import asdict
from memory.extract import ExtractedFact, LLMExtractor
from memory.providers import HashEmbedder
from scenarios.build import build_timeline

RIGHT = {"region": "sa-east-1", "backend": "pytest", "frontend": "vitest",
         "command": "git pull --rebase origin main && git push", "fix": "Use o bundle da CA interna."}


class FakeLLM:
    """Extrai nada e responde sempre a resposta certa da fase final, citando a memória 1."""

    def __init__(self):
        self.calls = 0

    def complete_json(self, system, user, schema):
        self.calls += 1
        props = schema["properties"]
        if "facts" in props:
            return {"facts": []}
        return {k: ([1] if k == "used_memories" else RIGHT.get(k, "")) for k in props}


def test_runner_end_to_end_with_fake_llm():
    tl = build_timeline()
    llm = FakeLLM()
    extractor = CachedExtractor(LLMExtractor(llm))
    trace = io.StringIO()
    rows = []
    for policy in make_policies(HashEmbedder(), extractor):
        rows += run_version(policy, tl, trace, llm)

    assert len(rows) == 4 * len(tl.probes)
    assert len(trace.getvalue().splitlines()) == len(rows)
    assert json.loads(trace.getvalue().splitlines()[0])["memories"]

    summary = {s["version"]: s for s in summarize(rows)}
    assert set(summary) == {"v1_naive", "v2_keyed", "v3_decay_rank", "v4_decay_tier"}
    for s in summary.values():
        # sempre "sa-east-1": erra as semanas 1 a 3 do update e acerta o resto
        assert s["ans_update"] == 5 / 8
        assert s["ans_scope"] == s["ans_rare_fact"] == s["ans_injection"] == 1.0
        assert s["retrieval"] is not None


def test_extraction_cache_persists_to_disk(tmp_path):
    class OneFact:
        calls = 0

        def extract(self, text, keys):
            OneFact.calls += 1
            return [ExtractedFact("a.b", "1", "semantic", "A é 1.")]

    path = tmp_path / "cache.json"
    CachedExtractor(OneFact(), path).extract("texto", ["x.y"])
    again = CachedExtractor(OneFact(), path).extract("texto", ["x.y"])
    assert again == [ExtractedFact("a.b", "1", "semantic", "A é 1.")]
    assert OneFact.calls == 1


def test_extraction_is_shared_between_versions_2_to_4():
    tl = build_timeline()
    llm = FakeLLM()
    extractor = CachedExtractor(LLMExtractor(llm))
    policies = make_policies(HashEmbedder(), extractor)
    for p in policies[1:]:
        run_version(p, tl, io.StringIO())
    extraction_calls = len(extractor.cache)
    # uma extração por mensagem confiável, não três
    assert extraction_calls == sum(m.source != "web" for m in tl.messages)


def test_retrieval_only_mode_never_calls_the_answer_llm():
    tl = build_timeline()
    llm = FakeLLM()
    policies = make_policies(HashEmbedder(), CachedExtractor(LLMExtractor(llm)), include_baseline=True)
    rows = []
    for p in policies:
        rows += run_version(p, tl, io.StringIO())
    assert all(r.passed is None for r in rows)
    assert llm.calls == len({m.text for m in tl.messages if m.source != "web"})  # só extração
    baseline = [r for r in rows if r.version == "v0_no_memory"]
    # sem memória nada é recuperado: só a injeção "passa" (nada de ruim entrou)
    assert {r.scenario for r in baseline if r.retrieved_ok} == {"injection"}


def test_rows_roundtrip_through_csv(tmp_path):
    tl = build_timeline()
    policy = make_policies(HashEmbedder(), CachedExtractor(LLMExtractor(FakeLLM())))[0]
    rows = run_version(policy, tl, io.StringIO())
    path = tmp_path / "runs.csv"
    _append_csv(path, [asdict(r) for r in rows])
    assert read_rows(path) == rows


def test_measuring_retrieval_does_not_count_as_use():
    tl = build_timeline()
    policies = make_policies(HashEmbedder(), CachedExtractor(LLMExtractor(FakeLLM())))
    for p in policies:
        run_version(p, tl, io.StringIO())
        assert all(r.use_count == 0 and r.last_used_at is None for r in p.store.all())


def test_stale_share_counts_old_region_memories_after_migration():
    tl = build_timeline()
    naive = make_policies(HashEmbedder(), CachedExtractor(LLMExtractor(FakeLLM())))[0]
    rows = {r.probe_id: r for r in run_version(naive, tl, io.StringIO())}
    assert rows["update-w2"].stale_share is None  # antes da migração não há valor obsoleto
    share = rows["update-w6"].stale_share
    assert share is not None and 0 < share <= 1
    assert rows["scope-w5"].stale_share is None
