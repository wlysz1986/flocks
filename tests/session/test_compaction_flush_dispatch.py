"""Tests for the background memory-flush dispatcher.

Covers the safety guarantees added on top of the naive ``create_task``
backgrounding:

* per-session serialisation (no overlapping daily-file writes for the same
  session);
* hard timeout on the background flush (a stuck provider call cannot pin
  references forever);
* opt-out via ``FLOCKS_COMPACTION_FLUSH_BACKGROUND=0`` (legacy inline mode);
* drain helper for graceful shutdown.

Plus a coverage test for ``summarize_chunked``'s parallel chunk dispatch.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from flocks.session.lifecycle.compaction import compaction as compaction_mod
from flocks.session.lifecycle.compaction import summary as summary_mod
from flocks.session.lifecycle.compaction.compaction import (
    SessionCompaction,
    drain_pending_flush_tasks,
)


@pytest.fixture(autouse=True)
def _reset_flush_state(monkeypatch):
    """Make sure leftover tasks/locks from earlier tests do not leak in."""
    compaction_mod._pending_flush_tasks.clear()
    compaction_mod._session_flush_locks.clear()
    monkeypatch.delenv("FLOCKS_COMPACTION_FLUSH_BACKGROUND", raising=False)
    monkeypatch.delenv("FLOCKS_COMPACTION_FLUSH_TIMEOUT", raising=False)
    yield
    compaction_mod._pending_flush_tasks.clear()
    compaction_mod._session_flush_locks.clear()


def _patch_flush(monkeypatch, fake):
    """Replace ``_flush_memory_to_daily`` with ``fake`` (an async callable)."""
    monkeypatch.setattr(
        SessionCompaction,
        "_flush_memory_to_daily",
        classmethod(lambda cls, **kwargs: fake(**kwargs)),
    )


# ---------------------------------------------------------------------------
# Backgrounding & ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_returns_immediately_in_background_mode(monkeypatch):
    """Dispatch must not wait for the slow flush to finish."""
    flush_started = asyncio.Event()
    flush_release = asyncio.Event()

    async def fake_flush(**kwargs):
        flush_started.set()
        await flush_release.wait()

    _patch_flush(monkeypatch, fake_flush)

    started = time.perf_counter()
    await SessionCompaction._dispatch_memory_flush(
        session_id="sess-A",
        summary_text="summary",
        chat_messages=[],
        model_id="m",
        provider_client=object(),
        ChatMessage=object,
        policy=None,
    )
    elapsed = time.perf_counter() - started

    # Background task should have been scheduled but not awaited.
    assert elapsed < 0.5
    assert len(compaction_mod._pending_flush_tasks) == 1

    await asyncio.wait_for(flush_started.wait(), timeout=1.0)
    flush_release.set()
    await drain_pending_flush_tasks(timeout=2.0)
    assert not compaction_mod._pending_flush_tasks


@pytest.mark.asyncio
async def test_inline_mode_when_env_disables_background(monkeypatch):
    """``FLOCKS_COMPACTION_FLUSH_BACKGROUND=0`` falls back to await."""
    monkeypatch.setenv("FLOCKS_COMPACTION_FLUSH_BACKGROUND", "0")

    calls: list[str] = []

    async def fake_flush(**kwargs):
        calls.append(kwargs["session_id"])

    _patch_flush(monkeypatch, fake_flush)

    await SessionCompaction._dispatch_memory_flush(
        session_id="sess-inline",
        summary_text="x",
        chat_messages=[],
        model_id="m",
        provider_client=object(),
        ChatMessage=object,
        policy=None,
    )

    assert calls == ["sess-inline"]
    assert not compaction_mod._pending_flush_tasks


# ---------------------------------------------------------------------------
# Per-session serialisation (the regression that motivated this fix)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_session_flushes_run_serially(monkeypatch):
    """Two dispatches on the same session must not overlap in the flush body.

    The daily memory file uses non-atomic read-modify-write, so concurrent
    flushes for the same session would silently lose data.
    """
    in_flight = 0
    max_in_flight = 0
    lock_for_counter = asyncio.Lock()

    async def fake_flush(**kwargs):
        nonlocal in_flight, max_in_flight
        async with lock_for_counter:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        async with lock_for_counter:
            in_flight -= 1

    _patch_flush(monkeypatch, fake_flush)

    for _ in range(3):
        await SessionCompaction._dispatch_memory_flush(
            session_id="same-sess",
            summary_text="x",
            chat_messages=[],
            model_id="m",
            provider_client=object(),
            ChatMessage=object,
            policy=None,
        )

    await drain_pending_flush_tasks(timeout=5.0)
    assert max_in_flight == 1, (
        f"per-session serialisation broken: max concurrent flushes = {max_in_flight}"
    )


@pytest.mark.asyncio
async def test_different_sessions_flush_in_parallel(monkeypatch):
    """Cross-session parallelism must be preserved."""
    in_flight = 0
    max_in_flight = 0
    counter_lock = asyncio.Lock()

    async def fake_flush(**kwargs):
        nonlocal in_flight, max_in_flight
        async with counter_lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.1)
        async with counter_lock:
            in_flight -= 1

    _patch_flush(monkeypatch, fake_flush)

    for sid in ("a", "b", "c", "d"):
        await SessionCompaction._dispatch_memory_flush(
            session_id=sid,
            summary_text="x",
            chat_messages=[],
            model_id="m",
            provider_client=object(),
            ChatMessage=object,
            policy=None,
        )

    await drain_pending_flush_tasks(timeout=5.0)
    assert max_in_flight >= 2, (
        f"cross-session parallelism lost: max concurrent flushes = {max_in_flight}"
    )


# ---------------------------------------------------------------------------
# Timeout safety net
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_background_flush_times_out(monkeypatch, caplog):
    """A stuck flush must be cancelled by the wait_for guard."""
    monkeypatch.setenv("FLOCKS_COMPACTION_FLUSH_TIMEOUT", "0.05")

    async def fake_flush(**kwargs):
        await asyncio.sleep(5.0)

    _patch_flush(monkeypatch, fake_flush)

    await SessionCompaction._dispatch_memory_flush(
        session_id="hangs",
        summary_text="x",
        chat_messages=[],
        model_id="m",
        provider_client=object(),
        ChatMessage=object,
        policy=None,
    )

    await drain_pending_flush_tasks(timeout=2.0)
    assert not compaction_mod._pending_flush_tasks


# ---------------------------------------------------------------------------
# drain helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_returns_quickly_when_no_pending_tasks():
    await drain_pending_flush_tasks(timeout=5.0)


# ---------------------------------------------------------------------------
# summarize_chunked: parallel dispatch
# ---------------------------------------------------------------------------


class _StubResponse:
    def __init__(self, text: str):
        self.content = text


class _StubProvider:
    """Records concurrent .chat() invocations and sleeps to simulate latency."""

    def __init__(self, latency: float = 0.1):
        self.latency = latency
        self._in_flight = 0
        self.peak_in_flight = 0
        self._lock = asyncio.Lock()
        self.calls = 0

    async def chat(self, *, model_id, messages, max_tokens):
        async with self._lock:
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
            self.calls += 1
        await asyncio.sleep(self.latency)
        async with self._lock:
            self._in_flight -= 1
        return _StubResponse(
            "## Decisions\n- ok\n## Current Task\nfoo\n"
            "## Open TODOs\n- none\n## Key Files & Identifiers\n- none"
        )


class _StubMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


@pytest.mark.asyncio
async def test_summarize_chunked_runs_chunks_in_parallel(monkeypatch):
    """``summarize_chunked`` must dispatch chunk LLM calls concurrently."""
    monkeypatch.setenv("FLOCKS_COMPACTION_CHUNK_CONCURRENCY", "4")

    # Force chunking: small target_chars + many big messages.
    msgs = [_StubMessage("user", "x" * 200) for _ in range(12)]
    provider = _StubProvider(latency=0.15)

    started = time.perf_counter()
    result = await summary_mod.summarize_chunked(
        chat_messages=msgs,
        prompt_text="prompt",
        target_chars=400,  # ~2 messages per chunk → ~6 chunks
        provider_client=provider,
        model_id="m",
        max_tokens=2000,
        session_id="sess-chunked",
    )
    elapsed = time.perf_counter() - started

    assert result is not None
    # At least 2 chunk calls plus the merge call → > 2 chats.
    assert provider.calls >= 3
    # Parallelism actually happened.
    assert provider.peak_in_flight >= 2, (
        f"chunked summarisation ran serially: peak_in_flight = {provider.peak_in_flight}"
    )
    # Sanity: should be much faster than serial (~6 * 0.15 = 0.9s) + merge.
    assert elapsed < 1.5


@pytest.mark.asyncio
async def test_summarize_chunked_concurrency_limit_respected(monkeypatch):
    """Semaphore caps the number of in-flight chunk calls."""
    monkeypatch.setenv("FLOCKS_COMPACTION_CHUNK_CONCURRENCY", "2")

    msgs = [_StubMessage("user", "x" * 200) for _ in range(12)]
    provider = _StubProvider(latency=0.15)

    await summary_mod.summarize_chunked(
        chat_messages=msgs,
        prompt_text="prompt",
        target_chars=400,
        provider_client=provider,
        model_id="m",
        max_tokens=2000,
        session_id="sess-cap",
    )
    # Final merge call runs alone after gather, so the *peak* during the
    # parallel phase should be ≤ 2.
    assert provider.peak_in_flight <= 2, (
        f"concurrency limit violated: peak_in_flight = {provider.peak_in_flight}"
    )
