"""Consumer tests for the retry/failure behaviour, with a stubbed API client.

No broker or scoring service needed - these cover the decisions the consumer
makes when the API misbehaves, which is the part worth pinning down.
"""

import httpx
import pytest

from src.streaming.consumer import SCORE_RETRIES, score_event

EVENT = {
    "tx_id": "T1",
    "sender_account_key": "011_A",
    "receiver_account_key": "020_B",
    "amount_paid": 100.0,
    "payment_currency": "US Dollar",
    "amount_received": 100.0,
    "receiving_currency": "US Dollar",
    "payment_format": "ACH",
    "timestamp": "2022-09-09T02:47:00",
}


def _client(handler) -> httpx.Client:
    return httpx.Client(base_url="http://testserver", transport=httpx.MockTransport(handler))


def test_successful_score_is_returned():
    def handler(request):
        assert request.headers["X-API-Key"]  # service-to-service credential
        return httpx.Response(200, json={"tx_id": "T1", "score": 0.9, "is_flagged": True})

    with _client(handler) as client:
        result = score_event(client, EVENT)

    assert result["is_flagged"] is True


def test_client_error_is_not_retried():
    """A 422 means this event will never be accepted - retrying it just delays
    the rest of the stream."""
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(422, json={"detail": "bad"})

    with _client(handler) as client:
        result = score_event(client, EVENT)

    assert result is None
    assert len(calls) == 1


def test_server_error_is_retried_then_gives_up():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, text="unavailable")

    with _client(handler) as client:
        result = score_event(client, EVENT)

    assert result is None
    assert len(calls) == SCORE_RETRIES


def test_transient_failure_then_success_returns_the_score():
    """A brief API blip should not cost the event - the retry is the point."""
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json={"tx_id": "T1", "score": 0.2, "is_flagged": False})

    with _client(handler) as client:
        result = score_event(client, EVENT)

    assert result["is_flagged"] is False
    assert len(calls) == 2


def test_unreachable_api_returns_none_instead_of_raising():
    """One unscored transaction is logged and skipped; it must not take the
    whole consumer down."""

    def handler(request):
        raise httpx.ConnectError("connection refused")

    with _client(handler) as client:
        result = score_event(client, EVENT)

    assert result is None


@pytest.mark.parametrize("status", [500, 502, 503])
def test_server_errors_are_treated_as_transient(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status)

    with _client(handler) as client:
        score_event(client, EVENT)

    assert len(calls) == SCORE_RETRIES
