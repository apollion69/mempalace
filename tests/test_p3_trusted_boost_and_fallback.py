"""P3 reconciliation tests — fork search behaviors re-layered onto v3.4.0.

These pin the three behaviors carried from the fork search-cluster onto the
upstream pluggable-backend base (strategy A):
  1. trusted-source authority boost inside _hybrid_rank (rides on top of any
     candidate strategy, incl. upstream's lexical_search union);
  2. boost gated by lexical overlap (authority alone never wins);
  3. BM25 filtered-fallback layering — upstream's unfiltered-retry first, our
     sqlite BM25 only as last resort — and fallback_state exposed on the result.

Written RED against the v3.4.0 clean base (P2), GREEN after the P3 port.
"""
from __future__ import annotations

from mempalace import searcher


def _hit(text, wing="", source_file="", distance=0.5):
    return {
        "text": text,
        "wing": wing,
        "source_file": source_file,
        "metadata": {"wing": wing, "source_file": source_file},
        "distance": distance,
    }


def test_trusted_card_boost_promotes_equal_vector_hit():
    """A trusted-wing fact-card with query overlap must outrank an equal-vector
    non-trusted hit after _hybrid_rank."""
    query = "zabbix maintenance window"
    plain = _hit("zabbix maintenance window notes", wing="general", distance=0.4)
    card = _hit(
        "card_id: x\nzabbix maintenance window procedure",
        wing="infra-facts",
        source_file="infra-zabbix.md",
        distance=0.4,
    )
    ranked = searcher._hybrid_rank([plain, card], query)
    assert ranked[0] is card, "trusted card should rank first"
    assert card.get("authority_boost", 0) > 0


def test_trusted_boost_gated_by_overlap():
    """No query-term overlap → zero authority boost even for a trusted wing."""
    query = "kubernetes pod eviction"
    card = _hit(
        "card_id: y\nunrelated backup retention policy text",
        wing="infra-facts",
        source_file="infra-backup.md",
        distance=0.5,
    )
    searcher._hybrid_rank([card], query)
    assert card.get("authority_boost", 0) == 0


def test_hybrid_rank_sets_authority_boost_key():
    """Every ranked hit carries an authority_boost key (observability)."""
    query = "ansible lock"
    hit = _hit("ansible lock contention", wing="general", distance=0.3)
    searcher._hybrid_rank([hit], query)
    assert "authority_boost" in hit


def test_filtered_fallback_tries_upstream_retry_before_bm25(monkeypatch):
    """_query_drawers_or_bm25_fallback must call upstream's filter-fallback first;
    only if that raises does it reach the sqlite BM25 fallback."""
    calls = []

    def fake_filter_fallback(drawers_col, dkwargs, query, n_results, wing, room):
        calls.append("filter_fallback")
        return {"documents": [["doc"]], "metadatas": [[{}]], "distances": [[0.1]]}

    monkeypatch.setattr(searcher, "_query_drawers_with_filter_fallback", fake_filter_fallback)
    res, fallback, err = searcher._query_drawers_or_bm25_fallback(
        drawers_col=object(),
        query="q",
        palace_path="/nonexistent",
        wing="infra-facts",
        room=None,
    )
    assert calls == ["filter_fallback"]
    assert fallback is None and err is None
    assert res is not None


def test_trusted_wing_bm25_candidates_enter_pool_when_no_wing(monkeypatch):
    """When no wing filter is set, BM25 candidates from each trusted wing must be
    pulled into the rerank pool — otherwise tiny trusted wings (infra-facts: 125
    drawers) are drowned by huge session wings in a global vector/union pool."""
    pulled = []

    def fake_bm25(query, palace_path, wing=None, room=None, n_results=5, _include_internal=False):
        pulled.append(wing)
        if wing == "infra-facts":
            return {"results": [{
                "text": "card_id: z\nzabbix dashboard widget layout",
                "wing": "infra-facts",
                "source_file": "infra-zabbix.md",
                "_source_file_full": "infra-zabbix.md",
                "_chunk_index": 0,
                "metadata": {"wing": "infra-facts", "source_file": "infra-zabbix.md"},
            }]}
        return {"results": []}

    monkeypatch.setattr(searcher, "_bm25_only_via_sqlite", fake_bm25)
    hits = [{"text": "noise", "wing": "cursor-sessions", "distance": 0.3,
             "_source_file_full": "s.jsonl", "_chunk_index": 1}]
    searcher._augment_with_trusted_wing_bm25(
        hits, "zabbix dashboard widget layout", "/p", wing=None, room=None,
        n_results=5, max_distance=0.0,
    )
    assert "infra-facts" in pulled, "each trusted wing must be queried"
    assert any(h.get("wing") == "infra-facts" for h in hits), "trusted card must enter the pool"
    assert any(h.get("distance") is None for h in hits), "BM25-only additions carry distance=None"


def test_trusted_wing_augment_skipped_when_wing_set(monkeypatch):
    """An explicit wing filter means the caller already scoped — no trusted-wing pull."""
    called = []
    monkeypatch.setattr(searcher, "_bm25_only_via_sqlite",
                        lambda *a, **k: called.append(1) or {"results": []})
    hits = [{"text": "x", "wing": "general", "distance": 0.2}]
    searcher._augment_with_trusted_wing_bm25(
        hits, "q", "/p", wing="general", room=None, n_results=5, max_distance=0.0
    )
    assert called == [], "no trusted-wing pull when a wing filter is active"


def test_trusted_wing_augment_skipped_under_max_distance(monkeypatch):
    """A strict vector-distance bound must not be bypassed by distance=None BM25 adds."""
    called = []
    monkeypatch.setattr(searcher, "_bm25_only_via_sqlite",
                        lambda *a, **k: called.append(1) or {"results": []})
    hits = []
    searcher._augment_with_trusted_wing_bm25(
        hits, "q", "/p", wing=None, room=None, n_results=5, max_distance=0.4
    )
    assert called == [], "no BM25-only injection when max_distance > 0"


def test_vector_path_failure_is_logged_even_when_bm25_recovers(monkeypatch, caplog):
    """A vector-path exception must be logged (observability) even when the BM25
    fallback recovers — so a programming error is never fully masked."""
    def boom(*a, **k):
        raise TypeError("renamed kwarg leaked")

    monkeypatch.setattr(searcher, "_query_drawers_with_filter_fallback", boom)
    monkeypatch.setattr(
        searcher,
        "_bm25_fallback_after_filtered_vector_error",
        lambda *a, **k: {"results": [], "fallback_state": "active"},
    )
    with caplog.at_level("WARNING", logger="mempalace_mcp"):
        searcher._query_drawers_or_bm25_fallback(
            drawers_col=object(), query="q", palace_path="/nope", wing="infra-facts"
        )
    assert any("vector" in r.message.lower() for r in caplog.records), caplog.text


def test_double_failure_logs_bm25_error(monkeypatch, caplog):
    """When BM25 ALSO fails, the BM25 error must be logged, not silently dropped."""
    monkeypatch.setattr(
        searcher,
        "_bm25_only_via_sqlite",
        lambda *a, **k: {"error": "sqlite open failed: disk gone"},
    )
    with caplog.at_level("WARNING", logger="mempalace_mcp"):
        out = searcher._bm25_fallback_after_filtered_vector_error(
            RuntimeError("vec boom"), "q", "/nope", wing="infra-facts"
        )
    assert out is None
    assert any("bm25" in r.message.lower() for r in caplog.records), caplog.text


def test_filtered_fallback_reaches_bm25_on_retry_failure(monkeypatch):
    """If upstream's filter-fallback raises, we fall to BM25 and expose fallback state."""
    def boom(*a, **k):
        raise RuntimeError("Error finding id")

    captured = {}

    def fake_bm25(error, query, palace_path, **kw):
        captured["reason"] = "called"
        return {"results": [], "fallback_reason": "vector_query_failed", "vector_error": str(error)}

    monkeypatch.setattr(searcher, "_query_drawers_with_filter_fallback", boom)
    monkeypatch.setattr(searcher, "_bm25_fallback_after_filtered_vector_error", fake_bm25)
    res, fallback, err = searcher._query_drawers_or_bm25_fallback(
        drawers_col=object(),
        query="q",
        palace_path="/nonexistent",
        wing="infra-facts",
        room=None,
    )
    assert res is None
    assert fallback is not None and fallback.get("fallback_reason") == "vector_query_failed"
    assert isinstance(err, RuntimeError)
    assert captured.get("reason") == "called"
