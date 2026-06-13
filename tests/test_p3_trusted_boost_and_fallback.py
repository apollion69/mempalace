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
