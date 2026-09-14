"""Close out a retrain request (SAD C11, TRD 4.3).

    uv run python -m src.mlops.close_retrain --latest --model-version gnn_v3
    uv run python -m src.mlops.close_retrain --job-id retrain-abc123 --status failed

`POST /retrain` records a *work order*: the locked compute decision puts GNN
training on Kaggle/Colab, so the API has no GPU and cannot run anything. This
command is the other half - the operator who actually ran the training comes
back and records what happened.

**Why this is a CLI and not an endpoint.** An endpoint called "complete my
retrain" would sit in a service that demonstrably did not perform the retrain.
The person closing the run is the person who ran it, and they are at a
terminal. Keeping it here means nothing in the API implies it trains models.

**What closing a run actually does**, and why it matters more than the status
column: it stamps the analyst feedback that the retrain consumed
(`used_in_training_run`). Until something writes that, "analyst feedback feeds
retraining" is a claim with no record behind it, and the next retrain has no
way to tell new labels from ones already used.

Failed runs deliberately consume nothing - that feedback must stay available
for the next attempt.
"""

from __future__ import annotations

import argparse
import logging

from sqlalchemy import select, update

from src.api.model_registry import ARTIFACTS_ROOT
from src.db.models import Feedback, RetrainRun
from src.db.session import session_scope

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("mlops.close_retrain")

COMPLETED = "completed"
FAILED = "failed"
OPEN_STATUSES = ("requested", "running")


class CloseError(RuntimeError):
    """Something about the request is wrong; nothing was written."""


def _known_version(model_version: str) -> bool:
    """A recorded version must be one that exists on disk with real metrics.

    Guards the failure this whole project keeps guarding against: a database
    row asserting a model that was never trained, which then shows up on the
    Ops screen as fact.
    """
    return (ARTIFACTS_ROOT / model_version / "metrics.json").exists()


def close_run(
    job_id: str | None,
    status: str = COMPLETED,
    model_version: str | None = None,
    notes: str | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Close a retrain run and, when it succeeded, stamp the feedback it used."""
    if status not in (COMPLETED, FAILED):
        raise CloseError(f"status must be '{COMPLETED}' or '{FAILED}', not '{status}'")

    if status == COMPLETED:
        if not model_version:
            raise CloseError(
                "--model-version is required when completing a run: a completed "
                "retrain that names no model has recorded nothing useful."
            )
        if not _known_version(model_version):
            raise CloseError(
                f"No evaluated artifacts for '{model_version}' "
                f"({ARTIFACTS_ROOT / model_version / 'metrics.json'} is missing). "
                "Train and evaluate it before recording it as the result."
            )

    with session_scope() as session:
        run = _select_run(session, job_id)

        if run.status in (COMPLETED, FAILED) and not force:
            raise CloseError(
                f"{run.job_id} is already {run.status}"
                + (f" ({run.resulting_model_version})" if run.resulting_model_version else "")
                + ". Pass --force to overwrite."
            )

        run.status = status
        run.notes = notes
        run.resulting_model_version = model_version if status == COMPLETED else None

        consumed = 0
        if status == COMPLETED:
            # Only rows not already claimed by an earlier retrain. A failed run
            # reaches neither branch, so its feedback stays available.
            consumed = session.execute(
                update(Feedback)
                .where(Feedback.used_in_training_run.is_(None))
                .values(used_in_training_run=run.job_id)
            ).rowcount

        return {
            "job_id": run.job_id,
            "status": run.status,
            "model_version": run.resulting_model_version,
            "feedback_consumed": consumed,
            "requested_by": run.requested_by,
        }


def _select_run(session, job_id: str | None) -> RetrainRun:
    if job_id:
        run = session.scalar(select(RetrainRun).where(RetrainRun.job_id == job_id))
        if run is None:
            raise CloseError(f"No retrain run with job_id '{job_id}'.")
        return run

    run = session.scalar(
        select(RetrainRun)
        .where(RetrainRun.status.in_(OPEN_STATUSES))
        .order_by(RetrainRun.requested_at.desc())
    )
    if run is None:
        raise CloseError("No open retrain runs to close. Request one from the Ops screen first.")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--job-id", help="The run to close.")
    target.add_argument(
        "--latest", action="store_true", help="Close the most recent open run."
    )
    parser.add_argument("--status", default=COMPLETED, choices=[COMPLETED, FAILED])
    parser.add_argument(
        "--model-version", help="Version this retrain produced (required when completing)."
    )
    parser.add_argument("--notes", help="Free text recorded against the run.")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite a run that is already closed."
    )
    args = parser.parse_args()

    try:
        result = close_run(
            job_id=None if args.latest else args.job_id,
            status=args.status,
            model_version=args.model_version,
            notes=args.notes,
            force=args.force,
        )
    except CloseError as error:
        raise SystemExit(f"error: {error}") from error

    print(f"{result['job_id']} -> {result['status']}")
    if result["model_version"]:
        print(f"  resulting model : {result['model_version']}")
    if result["status"] == COMPLETED:
        print(f"  feedback consumed: {result['feedback_consumed']} analyst decision(s)")
    else:
        print("  feedback left unconsumed, so the next attempt can still use it")


if __name__ == "__main__":
    main()
