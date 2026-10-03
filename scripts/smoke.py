"""End-to-end smoke check — run this before any demo.

    uv run python -m scripts.smoke
    uv run python -m scripts.smoke --base-url https://demo.example.com

The PRD requires the system to "run live from a clean state", and the way demos
actually fail is never the model — it is a container that did not come up, a
stale token, or an empty queue nobody noticed until an interviewer was looking.
This walks the real path a demo takes and says plainly which step broke.

It is **read-mostly**: it scores a handful of synthetic transactions through the
real API (which writes audit rows, as it must to prove the path works) and
records no analyst decisions. Synthetic rows use a `smoke-` tx_id prefix that
real data cannot produce, so they are easy to spot and delete.

Exit code is 0 only if every required check passed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import uuid

import httpx

from src.common.config import get_settings

PREFIX = "smoke-"
TIMEOUT = 20.0
# The graph is the slow path: a cold Neo4j page cache over 6.9M edges takes far
# longer on the first query than on every one after it.
GRAPH_TIMEOUT = 90.0

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


class Report:
    """Collects results so one failure doesn't hide the rest - knowing three
    things are broken is more useful than discovering them one run at a time."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))
        marker = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " warn "}[status]
        print(f"[{marker}] {name}" + (f" - {detail}" if detail else ""))

    @property
    def failed(self) -> bool:
        return any(status == FAIL for status, _, _ in self.rows)

    def summary(self) -> None:
        passed = sum(1 for s, _, _ in self.rows if s == PASS)
        warned = sum(1 for s, _, _ in self.rows if s == WARN)
        failed = sum(1 for s, _, _ in self.rows if s == FAIL)
        print()
        print(f"{passed} passed, {warned} warnings, {failed} failed")
        if failed:
            print("\nNOT demo-ready. Fix the FAIL rows above, then re-run.")
        elif warned:
            print("\nDemo-ready, with caveats - read the warnings.")
        else:
            print("\nDemo-ready.")


def check_api(report: Report, base_url: str) -> dict | None:
    try:
        response = httpx.get(f"{base_url}/health", timeout=TIMEOUT)
    except httpx.HTTPError as error:
        report.add(
            FAIL, "API reachable", f"{base_url} - {type(error).__name__}. Is uvicorn running?"
        )
        return None

    if response.status_code != 200:
        report.add(FAIL, "API reachable", f"/health returned {response.status_code}")
        return None

    health = response.json()
    report.add(PASS, "API reachable", base_url)

    components = health["components"]
    for name, ok in components.items():
        # The database being down is survivable for scoring but means no audit
        # trail, which is a product feature - so it warns rather than passes.
        report.add(PASS if ok else WARN, f"component: {name}", "" if ok else "unavailable")

    report.add(
        PASS if health["status"] == "ok" else WARN,
        "service status",
        health["status"],
    )
    report.add(
        PASS,
        "serving versions",
        f"{health['model_version']} / {health['embedding_version']}",
    )
    return health


def check_login(report: Report, base_url: str, username: str, password: str) -> str | None:
    try:
        response = httpx.post(
            f"{base_url}/auth/login",
            json={"username": username, "password": password},
            timeout=TIMEOUT,
        )
    except httpx.HTTPError as error:
        report.add(FAIL, f"login as {username}", type(error).__name__)
        return None

    if response.status_code != 200:
        report.add(
            FAIL,
            f"login as {username}",
            f"{response.status_code} - seed the user with src.db.seed_users",
        )
        return None

    body = response.json()
    report.add(PASS, f"login as {username}", f"role={body['role']}")
    return body["access_token"]


def check_scoring(report: Report, base_url: str, api_key: str) -> None:
    """Scores a synthetic transaction through the real endpoint.

    Uses an account key that cannot exist in the dataset, so the embedding is
    genuinely missing - which also exercises the degraded path rather than only
    the happy one.
    """
    tx_id = f"{PREFIX}{uuid.uuid4().hex[:10]}"
    payload = {
        "tx_id": tx_id,
        "sender_account_key": "999_SMOKETEST_A",
        "receiver_account_key": "999_SMOKETEST_B",
        "amount_paid": 4321.0,
        "payment_currency": "US Dollar",
        "amount_received": 4321.0,
        "receiving_currency": "US Dollar",
        "payment_format": "ACH",
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
    }

    try:
        response = httpx.post(
            f"{base_url}/score", json=payload, headers={"X-API-Key": api_key}, timeout=TIMEOUT
        )
    except httpx.HTTPError as error:
        report.add(FAIL, "score a transaction", type(error).__name__)
        return

    if response.status_code != 200:
        report.add(FAIL, "score a transaction", f"{response.status_code}: {response.text[:120]}")
        return

    body = response.json()
    report.add(
        PASS,
        "score a transaction",
        f"score={body['score']:.3f} latency={body['latency_ms']}ms",
    )

    # The latency NFR is <1s; measured p95 is ~35ms, so anything near the limit
    # means something is wrong rather than merely slow.
    report.add(
        PASS if body["latency_ms"] < 1000 else FAIL,
        "scoring latency < 1s",
        f"{body['latency_ms']}ms",
    )
    report.add(
        PASS if body["explanation"] else FAIL,
        "explanation present",
        f"{len(body['explanation'])} factors",
    )
    # Unknown accounts must degrade, not error - this is the locked rule.
    report.add(
        PASS if body["embedding_version"] == "missing" else WARN,
        "degrades on unknown account",
        f"embedding_version={body['embedding_version']}",
    )


def check_unauthenticated_access_is_refused(report: Report, base_url: str) -> None:
    """A demo that shows an open API is worse than one that shows no API."""
    for path, name in (("/score", "POST /score"), ("/flags", "GET /flags")):
        try:
            response = (
                httpx.post(f"{base_url}{path}", json={}, timeout=TIMEOUT)
                if path == "/score"
                else httpx.get(f"{base_url}{path}", timeout=TIMEOUT)
            )
        except httpx.HTTPError as error:
            report.add(FAIL, f"{name} requires auth", type(error).__name__)
            continue
        report.add(
            PASS if response.status_code in (401, 403) else FAIL,
            f"{name} requires auth",
            f"returned {response.status_code}",
        )


def check_queue(report: Report, base_url: str, token: str) -> None:
    try:
        response = httpx.get(
            f"{base_url}/flags?status=open&limit=200",
            headers={"Authorization": f"Bearer {token}"},
            timeout=TIMEOUT,
        )
    except httpx.HTTPError as error:
        report.add(FAIL, "flag queue loads", type(error).__name__)
        return

    if response.status_code != 200:
        report.add(FAIL, "flag queue loads", str(response.status_code))
        return

    body = response.json()
    flags = body["flags"]
    report.add(PASS, "flag queue loads", f"{body['count']} open, summary present")

    if not flags:
        # Not a failure: an empty queue is a correct state. But it is a bad
        # demo, and this is exactly what the check exists to catch early.
        report.add(
            WARN,
            "queue has flags to show",
            "EMPTY - replay events before demoing (producer first, then consumer)",
        )
        return

    detailed = [f for f in flags if f["amount_paid"] is not None]
    report.add(
        PASS if detailed else WARN,
        "flags carry transaction detail",
        f"{len(detailed)}/{len(flags)} rows show amount and counterparty",
    )

    with_explanation = [f for f in flags if f["explanation"]]
    report.add(
        PASS if with_explanation else FAIL,
        "flags carry explanations",
        f"{len(with_explanation)}/{len(flags)}",
    )
    return


def check_graph(report: Report, base_url: str, token: str, account_key: str | None) -> None:
    if not account_key:
        report.add(WARN, "money-flow graph", "skipped - no flag to inspect")
        return

    import time

    started = time.perf_counter()
    try:
        # A deliberately long timeout. The first 2-hop query after Neo4j starts
        # runs against a cold page cache over 6.9M edges and measured >20s,
        # while subsequent ones take ~0.25s. Reporting that as "Neo4j down" -
        # which an earlier version of this check did - sends you debugging a
        # healthy service.
        response = httpx.get(
            f"{base_url}/graph/{account_key}?hops=2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=GRAPH_TIMEOUT,
        )
    except httpx.TimeoutException:
        report.add(
            WARN,
            "money-flow graph",
            f"no response in {GRAPH_TIMEOUT:.0f}s - Neo4j is probably warming its cache. "
            "Re-run; if it persists, check the container.",
        )
        return
    except httpx.HTTPError as error:
        report.add(WARN, "money-flow graph", f"{type(error).__name__} - is Neo4j running?")
        return
    elapsed = time.perf_counter() - started

    if response.status_code == 503:
        # Documented behaviour: the graph panel degrades alone and scoring is
        # unaffected, so this is a warning rather than a failure.
        report.add(WARN, "money-flow graph", "Neo4j unreachable - detail view will degrade")
        return
    if response.status_code != 200:
        report.add(FAIL, "money-flow graph", str(response.status_code))
        return

    body = response.json()
    report.add(
        PASS,
        "money-flow graph",
        f"{len(body['nodes'])} accounts, {len(body['edges'])} payments in {elapsed:.2f}s",
    )
    # Warm it before an audience sees it: the first click is the slow one.
    report.add(
        PASS if elapsed < 3 else WARN,
        "graph responds quickly",
        f"{elapsed:.2f}s" if elapsed < 3 else f"{elapsed:.1f}s - open a flag once before demoing",
    )


def check_models(report: Report, base_url: str, token: str) -> None:
    try:
        response = httpx.get(
            f"{base_url}/models", headers={"Authorization": f"Bearer {token}"}, timeout=TIMEOUT
        )
    except httpx.HTTPError as error:
        report.add(FAIL, "model metrics load", type(error).__name__)
        return

    if response.status_code != 200:
        report.add(FAIL, "model metrics load", str(response.status_code))
        return

    body = response.json()
    lift = body.get("auprc_lift_pct")
    report.add(
        PASS if lift is not None else FAIL,
        "headline lift available",
        f"{lift:+.1f}%" if lift is not None else "missing - evaluate the models",
    )

    serving = next((v for v in body["versions"] if v["is_serving"]), None)
    if serving is None:
        report.add(FAIL, "serving version has metrics", body["serving_model_version"])
        return
    report.add(PASS, "serving version has metrics", serving["version"])

    if serving.get("alerts_per_day") is None:
        report.add(WARN, "review-load numbers", "absent - re-run evaluate.py to populate")
    else:
        report.add(
            PASS,
            "review-load numbers",
            f"{serving['alerts_per_day']:.0f} alerts/day at "
            f"{serving['precision_at_threshold']:.1%} precision",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=None, help="API base URL (default: from config).")
    parser.add_argument("--username", default="analyst")
    parser.add_argument("--password", default="analyst-demo-pw")
    args = parser.parse_args()

    settings = get_settings()
    base_url = args.base_url or settings.scoring_api_base_url

    print("Smoke check - the path a demo actually walks\n")
    report = Report()

    health = check_api(report, base_url)
    if health is None:
        report.summary()
        return 1

    check_unauthenticated_access_is_refused(report, base_url)
    check_scoring(report, base_url, settings.service_api_key)

    token = check_login(report, base_url, args.username, args.password)
    if token is None:
        report.summary()
        return 1

    check_queue(report, base_url, token)
    check_models(report, base_url, token)

    # Use a real flagged account for the graph check when one exists.
    account_key = None
    try:
        flags = httpx.get(
            f"{base_url}/flags?status=open&limit=1",
            headers={"Authorization": f"Bearer {token}"},
            timeout=TIMEOUT,
        ).json()["flags"]
        account_key = flags[0]["account_key"] if flags else None
    except Exception:
        pass
    check_graph(report, base_url, token, account_key)

    report.summary()
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
