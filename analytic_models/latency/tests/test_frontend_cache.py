from analytic_models.latency import frontend


def test_dense_prefill_cache_uses_content_and_compiler_fingerprints(tmp_path, monkeypatch):
    model = tmp_path / "model.json"
    settings = tmp_path / "settings.toml"
    model.write_text("{}")
    settings.write_text("[ANALYTIC]\n")

    calls = []
    sentinel = object()

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(frontend, "_estimate_dense_prefill_uncached", fake)
    monkeypatch.setattr(frontend, "_compiler_source_fingerprint", lambda: "compiler-a")
    frontend.clear_dense_prefill_cache()

    first = frontend.estimate_dense_prefill(
        model,
        settings,
        seq_len=8,
    )
    second = frontend.estimate_dense_prefill(
        model,
        settings,
        seq_len=8,
    )
    assert first is sentinel and second is sentinel
    assert len(calls) == 1
    assert frontend.dense_prefill_cache_info().maxsize == 4
    assert frontend.dense_prefill_cache_info().hits == 1

    settings.write_text("[ANALYTIC]\n# changed\n")
    frontend.estimate_dense_prefill(
        model,
        settings,
        seq_len=8,
    )
    assert len(calls) == 2
