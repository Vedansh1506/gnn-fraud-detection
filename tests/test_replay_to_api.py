"""The cloud demo's replay script.

It stands in for the broker + consumer hop on the deployed box, so what matters
is that it replays the same chronological test window the producer does, that
the laundering-start option is honest about what it is, and that it treats
failures the way the real consumer does: retry transient ones, never retry a 4xx.

Pure logic plus an in-process fake transport - no dataset, no network.
"""

from __future__ import annotations

import httpx
import pandas as pd

from scripts.replay_to_api import MAX_ATTEMPTS, post_event, select_window


def _frame(labels: list[int]) -> pd.DataFrame:
    """Rows deliberately out of time order, as the real file is."""
    base = pd.Timestamp("2022-09-09 00:00")
    order = list(reversed(range(len(labels))))
    return pd.DataFrame(
        {
            "Timestamp": [base + pd.Timedelta(minutes=i) for i in order],
            "Is Laundering": [labels[i] for i in order],
            "marker": order,
        }
    )


def test_window_is_chronological_regardless_of_file_order():
    """File order lands in a laundering-dense region and would make the demo
    look absurd - the producer sorts, and this must match it."""
    window = select_window(_frame([0, 0, 0, 0]), limit=None)
    assert window["Timestamp"].is_monotonic_increasing


def test_limit_caps_the_slice_after_ordering():
    window = select_window(_frame([0] * 10), limit=3)
    assert len(window) == 3
    assert list(window["marker"]) == [0, 1, 2]  # the three EARLIEST, not the first in file


def _client_returning(*statuses: int) -> tuple[httpx.Client, list[int]]:
    calls: list[int] = []
    queue = list(statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(queue.pop(0) if queue else statuses[-1])

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def test_success_returns_ok(monkeypatch):
    client, calls = _client_returning(200)
    assert post_event(client, "http://api", "key", {"tx_id": "T1"}) == "ok"
    assert len(calls) == 1


def test_a_4xx_is_never_retried(monkeypatch):
    """A 422 or 401 will never succeed on retry; retrying only hides it."""
    client, calls = _client_returning(422)
    assert post_event(client, "http://api", "key", {"tx_id": "T1"}) == "client_error"
    assert len(calls) == 1


def test_a_transient_5xx_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr("scripts.replay_to_api.time.sleep", lambda _s: None)
    client, calls = _client_returning(503, 503, 200)
    assert post_event(client, "http://api", "key", {"tx_id": "T1"}) == "ok"
    assert len(calls) == 3


def test_persistent_5xx_gives_up_after_the_attempt_limit(monkeypatch):
    monkeypatch.setattr("scripts.replay_to_api.time.sleep", lambda _s: None)
    client, calls = _client_returning(500)
    assert post_event(client, "http://api", "key", {"tx_id": "T1"}) == "server_error"
    assert len(calls) == MAX_ATTEMPTS


def test_the_api_key_header_is_sent(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    post_event(client, "http://api", "secret-key", {"tx_id": "T1"})
    assert seen.get("x-api-key") == "secret-key"
