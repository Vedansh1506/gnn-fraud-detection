"""Producer tests - no broker needed, these cover the event contract.

The event shape matters more than it looks: the consumer forwards it straight
to POST /score, so a mismatch here surfaces as a 422 at the API instead of
anything obviously stream-related.
"""

import pandas as pd

from src.api.schemas import ScoreRequest
from src.streaming.producer import to_event


def _row() -> pd.Series:
    return pd.DataFrame(
        {
            "tx_id": ["T1"],
            "sender_account_key": ["011_A"],
            "receiver_account_key": ["020_B"],
            "Timestamp": pd.to_datetime(["2022-09-09 02:47:00"]),
            "Amount Paid": [5845.0],
            "Amount Received": [5845.0],
            "Payment Currency": ["US Dollar"],
            "Receiving Currency": ["Euro"],
            "Payment Format": ["ACH"],
            "Is Laundering": [1],
        }
    ).iloc[0]


def test_event_matches_the_scoring_api_contract():
    """The produced event must validate against the API's own request model -
    that's the contract the consumer relies on."""
    request = ScoreRequest(**to_event(_row()))

    assert request.tx_id == "T1"
    assert request.amount_paid == 5845.0
    assert request.payment_format == "ACH"


def test_event_carries_no_label():
    """Is Laundering is ground truth for evaluation, never an input - shipping
    it to the scorer would be leakage straight into serving."""
    event = to_event(_row())

    assert "Is Laundering" not in event
    assert not any("laundering" in key.lower() for key in event)


def test_event_values_are_json_serialisable_primitives():
    """numpy scalars and pandas Timestamps break json.dumps, and the producer
    serialises every event."""
    import json

    event = to_event(_row())
    round_tripped = json.loads(json.dumps(event))

    assert round_tripped["timestamp"] == "2022-09-09T02:47:00"
    assert isinstance(round_tripped["amount_paid"], float)
    assert isinstance(round_tripped["tx_id"], str)
