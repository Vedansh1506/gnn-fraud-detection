"""API contract tests: the request schema must reject bad input with 422
rather than coerce it (TRD 6 security: "reject rather than sanitize").

These exercise the Pydantic contract directly, so they need neither the trained
artifacts nor a database.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.api.schemas import ScoreRequest

VALID = {
    "tx_id": "T1",
    "sender_account_key": "011_A",
    "receiver_account_key": "020_B",
    "amount_paid": 100.0,
    "payment_currency": "US Dollar",
    "amount_received": 100.0,
    "receiving_currency": "US Dollar",
    "payment_format": "ACH",
    "timestamp": datetime(2022, 9, 1, 10, 0),
}


def test_valid_payload_is_accepted():
    assert ScoreRequest(**VALID).tx_id == "T1"


@pytest.mark.parametrize(
    "field",
    ["tx_id", "sender_account_key", "receiver_account_key", "amount_paid", "timestamp"],
)
def test_missing_required_field_is_rejected(field):
    payload = {k: v for k, v in VALID.items() if k != field}
    with pytest.raises(ValidationError):
        ScoreRequest(**payload)


def test_negative_amount_is_rejected():
    with pytest.raises(ValidationError):
        ScoreRequest(**{**VALID, "amount_paid": -1.0})


def test_empty_account_key_is_rejected():
    with pytest.raises(ValidationError):
        ScoreRequest(**{**VALID, "sender_account_key": ""})


def test_unparseable_timestamp_is_rejected():
    with pytest.raises(ValidationError):
        ScoreRequest(**{**VALID, "timestamp": "not-a-timestamp"})


def test_unknown_field_is_rejected_not_ignored():
    """extra="forbid": a caller sending a field we don't understand is a
    contract mismatch worth surfacing, not something to quietly drop."""
    with pytest.raises(ValidationError):
        ScoreRequest(**{**VALID, "unexpected_field": "surprise"})
