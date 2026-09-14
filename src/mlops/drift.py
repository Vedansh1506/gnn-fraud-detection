"""Feature and prediction drift with Evidently (SAD C11, Design Doc 4.4).

**What is compared, and what that does and does not cover.**

Reference = the transaction attributes the model trained on, sampled once into
`artifacts/drift/<model_version>/reference.parquet`.
Current = the same attributes for recently scored transactions, read back out
of the audit log.

Both sides therefore carry the columns the audit log actually stores - amount,
payment format, currency, hour of day - plus the model's own output score. That
covers input drift on the raw transaction attributes and **prediction drift**,
which in practice moves first and is the signal an operator acts on. It does
**not** cover the 64 embedding dimensions or the account aggregates, because
the audit log does not store them; a full-feature check would need the feature
vector persisted per scored transaction. Saying so is the point - a drift panel
that implies more coverage than it has is worse than one that admits its scope.

Nothing here is computed unless there is enough data to compute it. Below
`drift_min_rows` the verdict is "not enough data", never a green light.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from sqlalchemy import select

from src.common.config import get_settings
from src.db.models import AuditLogEntry
from src.db.session import session_scope

logger = logging.getLogger("mlops.drift")

DRIFT_ROOT = Path("artifacts/drift")

# The columns present on both sides. Deliberately the intersection of "what the
# model consumes" and "what the audit log keeps" - anything else could only be
# compared by reconstructing it, which would test our reconstruction, not drift.
NUMERIC_COLUMNS = ["amount_paid", "hour_of_day"]
CATEGORICAL_COLUMNS = ["payment_currency", "payment_format"]
PREDICTION_COLUMN = "score"

DriftState = Literal["ok", "warning", "alert", "not_instrumented"]

# A replay covering a few minutes of transaction time is not a sample of
# production, and comparing it to a full reference produces confident nonsense.
# Measured on a 40,000-event replay: every row landed in the same clock hour, so
# `hour_of_day` had exactly one distinct value and registered as severe drift
# that was purely an artifact of the replay window. Below this span the check
# reports that rather than a verdict.
MIN_CURRENT_SPAN_HOURS = 6.0


@dataclass(frozen=True)
class ColumnDrift:
    column: str
    drifted: bool
    score: float
    test: str


@dataclass(frozen=True)
class DriftReport:
    state: DriftState
    message: str
    checked_at: datetime | None
    reference_rows: int = 0
    current_rows: int = 0
    drifted_columns: int = 0
    total_columns: int = 0
    columns: tuple[ColumnDrift, ...] = ()
    prediction_drifted: bool = False


def reference_path(model_version: str, root: Path | None = None) -> Path:
    # Resolved at call time, not bound as a default: a default argument would
    # capture DRIFT_ROOT at import and quietly ignore any later override.
    return (root or DRIFT_ROOT) / model_version / "reference.parquet"


def build_reference(
    frame: pd.DataFrame, model_version: str, root: Path | None = None, sample: int = 50_000
) -> Path:
    """Persist the training-side reference once, as an artifact.

    Sampled rather than kept whole: the comparison needs a distribution, not
    6.9M rows, and re-deriving it from the raw dataset on every drift check
    would put a multi-minute load on an API request.
    """
    columns = [
        c for c in (*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS, PREDICTION_COLUMN) if c in frame
    ]
    reference = frame[columns]
    if len(reference) > sample:
        # Fixed seed: a reference that shifts between runs would show drift
        # that is entirely our own sampling noise.
        reference = reference.sample(sample, random_state=42)

    path = reference_path(model_version, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(path, index=False)
    logger.info("wrote drift reference %s (%d rows)", path, len(reference))
    return path


def load_reference(model_version: str, root: Path | None = None) -> pd.DataFrame | None:
    path = reference_path(model_version, root)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        logger.warning("unreadable drift reference at %s", path, exc_info=True)
        return None


def load_current(model_version: str, limit: int = 20_000) -> pd.DataFrame:
    """Recently scored transactions, shaped like the reference.

    Restricted to the serving model version: comparing scores produced by two
    different models would report a model change as data drift.
    """
    query = (
        select(
            AuditLogEntry.amount_paid,
            AuditLogEntry.payment_currency,
            AuditLogEntry.payment_format,
            AuditLogEntry.tx_timestamp,
            AuditLogEntry.score,
        )
        .where(
            AuditLogEntry.model_version == model_version,
            # Rows written before the transaction-detail columns existed carry
            # NULLs; including them would read as a collapse in every
            # distribution rather than the absence it is.
            AuditLogEntry.tx_timestamp.is_not(None),
        )
        .order_by(AuditLogEntry.scored_at.desc())
        .limit(limit)
    )

    with session_scope() as session:
        rows = session.execute(query).all()

    frame = pd.DataFrame(
        rows,
        columns=["amount_paid", "payment_currency", "payment_format", "tx_timestamp", "score"],
    )
    if frame.empty:
        return frame

    frame["tx_timestamp"] = pd.to_datetime(frame["tx_timestamp"], utc=True)
    frame["hour_of_day"] = frame["tx_timestamp"].dt.hour
    return frame


# Evidently names each per-column metric like:
#   ValueDrift(column=amount_paid,method=Wasserstein distance (normed),threshold=0.1)
# The column name and threshold are parsed out of that string because the
# structured payload does not expose them separately in this release.
_VALUE_DRIFT = re.compile(r"ValueDrift\(column=(?P<column>[^,)]+)")
_METHOD = re.compile(r"method=(?P<method>[^,)]+(?:\([^)]*\))?)")


def _span_hours(current: pd.DataFrame) -> float | None:
    """How much transaction time the current sample actually covers."""
    if "tx_timestamp" not in current.columns or current.empty:
        return None
    stamps = current["tx_timestamp"].dropna()
    if stamps.empty:
        return None
    return float((stamps.max() - stamps.min()).total_seconds() / 3600.0)


def _comparable_columns(reference: pd.DataFrame, current: pd.DataFrame) -> list[str]:
    """Columns worth comparing on both sides.

    A column with a single distinct value on either side is dropped: drift on a
    constant is not a meaningful verdict, and including it drags the drifted
    share toward whichever answer the constant happens to give.
    """
    candidates = [
        c
        for c in (*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS, PREDICTION_COLUMN)
        if c in reference.columns and c in current.columns
    ]
    return [c for c in candidates if reference[c].nunique() > 1 and current[c].nunique() > 1]


def _evaluate(reference: pd.DataFrame, current: pd.DataFrame) -> list[ColumnDrift]:
    """Run Evidently and flatten its result to per-column verdicts."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    columns = _comparable_columns(reference, current)
    if not columns:
        return []
    definition = DataDefinition(
        numerical_columns=[c for c in (*NUMERIC_COLUMNS, PREDICTION_COLUMN) if c in columns],
        categorical_columns=[c for c in CATEGORICAL_COLUMNS if c in columns],
    )

    reference_ds = Dataset.from_pandas(reference[columns], data_definition=definition)
    current_ds = Dataset.from_pandas(current[columns], data_definition=definition)

    result = Report(metrics=[DataDriftPreset()]).run(
        reference_data=reference_ds, current_data=current_ds
    )
    return _parse_result(json.loads(result.json()))


def _parse_result(payload: dict) -> list[ColumnDrift]:
    """Pull per-column drift out of Evidently's JSON payload.

    Defensive on purpose: Evidently's result shape has changed across releases
    (`.dict()` in this one returns `metric_name: None`, which is why the JSON
    form is used). An unrecognised payload yields an empty list, which the
    caller reports as uncomputable rather than as a confident "no drift".
    """
    found: list[ColumnDrift] = []

    for metric in payload.get("metrics", []):
        name = str(metric.get("metric_name") or "")
        match = _VALUE_DRIFT.search(name)
        if match is None:
            continue

        value = metric.get("value")
        if not isinstance(value, (int, float)):
            continue

        threshold = (metric.get("config") or {}).get("threshold")
        if not isinstance(threshold, (int, float)):
            continue

        method = _METHOD.search(name)
        found.append(
            ColumnDrift(
                column=match.group("column"),
                # Evidently's own convention: at or above the threshold counts
                # as drifted. Applying its threshold rather than inventing one
                # keeps the verdict consistent with its HTML report.
                drifted=float(value) >= float(threshold),
                score=float(value),
                test=method.group("method") if method else "unknown",
            )
        )

    return found


def check_drift(model_version: str | None = None) -> DriftReport:
    """The Ops screen's drift verdict. Never raises - a failed check reports
    itself as uncomputable rather than taking the screen down."""
    settings = get_settings()
    version = model_version or settings.model_version
    now = datetime.now().astimezone()

    reference = load_reference(version)
    if reference is None or reference.empty:
        return DriftReport(
            state="not_instrumented",
            message=(
                f"No drift reference for {version}. Build one with "
                "`uv run python -m src.mlops.build_reference` - until then no "
                "drift check has run and no status is claimed."
            ),
            checked_at=None,
        )

    try:
        current = load_current(version)
    except Exception:
        logger.exception("could not read scored transactions for drift")
        return DriftReport(
            state="not_instrumented",
            message="Could not read scored transactions - the audit database is unavailable.",
            checked_at=None,
        )

    span_hours = _span_hours(current)
    if span_hours is not None and span_hours < MIN_CURRENT_SPAN_HOURS and len(current) > 0:
        return DriftReport(
            state="not_instrumented",
            message=(
                f"The {len(current):,} scored transactions on record span only "
                f"{span_hours:.1f} hours of transaction time - too narrow a slice to "
                "compare against a full reference, and it would report the replay "
                "window itself as drift. Replay a wider time range."
            ),
            checked_at=None,
            reference_rows=len(reference),
            current_rows=len(current),
        )

    if len(current) < settings.drift_min_rows:
        return DriftReport(
            state="not_instrumented",
            message=(
                f"Only {len(current)} scored transactions with full detail on record; "
                f"{settings.drift_min_rows} are needed before a drift verdict means "
                "anything. Replay more traffic."
            ),
            checked_at=None,
            reference_rows=len(reference),
            current_rows=len(current),
        )

    try:
        columns = _evaluate(reference, current)
    except Exception:
        logger.exception("Evidently drift evaluation failed")
        return DriftReport(
            state="not_instrumented",
            message="The drift check could not be computed - see the service logs.",
            checked_at=None,
            reference_rows=len(reference),
            current_rows=len(current),
        )

    if not columns:
        return DriftReport(
            state="not_instrumented",
            message="The drift report returned no readable per-column results.",
            checked_at=None,
            reference_rows=len(reference),
            current_rows=len(current),
        )

    drifted = [c for c in columns if c.drifted]
    share = len(drifted) / len(columns)
    prediction_drifted = any(c.column == PREDICTION_COLUMN and c.drifted for c in columns)

    if share >= settings.drift_alert_share:
        state: DriftState = "alert"
    elif share >= settings.drift_warning_share or prediction_drifted:
        # Prediction drift alone is enough to warrant a look: the score
        # distribution moving means the queue an analyst sees has changed,
        # whatever the inputs did.
        state = "warning"
    else:
        state = "ok"

    return DriftReport(
        state=state,
        message=_describe(state, drifted, columns, len(current), prediction_drifted),
        checked_at=now,
        reference_rows=len(reference),
        current_rows=len(current),
        drifted_columns=len(drifted),
        total_columns=len(columns),
        columns=tuple(columns),
        prediction_drifted=prediction_drifted,
    )


def _describe(
    state: DriftState,
    drifted: list[ColumnDrift],
    columns: list[ColumnDrift],
    current_rows: int,
    prediction_drifted: bool,
) -> str:
    scope = (
        f"Checked {len(columns)} columns over {current_rows:,} recently scored transactions "
        "(transaction attributes and the score; embeddings are not covered)."
    )
    if state == "ok":
        return f"No drift detected. {scope}"

    names = ", ".join(sorted(c.column for c in drifted)) or "none"
    lead = "Significant drift" if state == "alert" else "Some drift"
    prediction_note = (
        " The score distribution itself has shifted, so the queue analysts see has changed."
        if prediction_drifted
        else ""
    )
    return f"{lead} detected in: {names}.{prediction_note} {scope}"
