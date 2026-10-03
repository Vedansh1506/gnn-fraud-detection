"""Download the pinned model artifacts from S3.

    uv run python -m scripts.fetch_artifacts --bucket my-bucket
    uv run python -m scripts.fetch_artifacts --bucket my-bucket --upload   # push instead

Used by the container entrypoint at start, and by hand to publish a newly
trained version. Artifacts are git-ignored and ~115 MB, so S3 is how they reach
a deployed box (SAD: "artifacts pulled from S3").

**Only the pinned versions are fetched**, never "the latest" — the same rule the
serving code follows. A bad retrain sitting in the bucket must not be able to
reach a running service just by existing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# Exactly what serving loads. Listed explicitly rather than syncing a prefix, so
# an unexpected extra object in the bucket is never silently pulled onto a
# serving box.
MODEL_FILES = ("model.json", "feature_spec.json", "account_features.parquet", "metrics.json")
EMBEDDING_FILES = ("embeddings.parquet", "metadata.json")

# metrics.json is useful (the Ops screen reads it) but not required to score.
OPTIONAL = {"metrics.json", "metadata.json"}


def _plan(model_version: str, embedding_version: str) -> list[tuple[str, Path]]:
    """(s3 key, local relative path) pairs. Keys mirror the local layout so the
    bucket is browsable and a human can see which version is which."""
    plan = [
        (f"models/{model_version}/{name}", Path("models") / model_version / name)
        for name in MODEL_FILES
    ]
    plan += [
        (f"embeddings/{embedding_version}/{name}", Path("embeddings") / embedding_version / name)
        for name in EMBEDDING_FILES
    ]
    return plan


def download(bucket: str, model_version: str, embedding_version: str, dest: Path) -> int:
    s3 = boto3.client("s3")
    failures = 0

    for key, relative in _plan(model_version, embedding_version):
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            s3.download_file(bucket, key, str(target))
            print(f"  fetched s3://{bucket}/{key} -> {target}")
        except ClientError as error:
            code = error.response["Error"]["Code"]
            if relative.name in OPTIONAL and code in ("404", "NoSuchKey"):
                print(f"  skipped optional {key} (absent)")
                continue
            print(f"  FAILED {key}: {code}", file=sys.stderr)
            failures += 1

    return failures


def upload(bucket: str, model_version: str, embedding_version: str, source: Path) -> int:
    s3 = boto3.client("s3")
    failures = 0

    for key, relative in _plan(model_version, embedding_version):
        local = source / relative
        if not local.exists():
            if relative.name in OPTIONAL:
                print(f"  skipped optional {relative} (not present locally)")
                continue
            print(f"  FAILED {relative}: not found locally", file=sys.stderr)
            failures += 1
            continue
        try:
            s3.upload_file(str(local), bucket, key)
            print(f"  uploaded {local} -> s3://{bucket}/{key}")
        except ClientError as error:
            print(f"  FAILED {key}: {error.response['Error']['Code']}", file=sys.stderr)
            failures += 1

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--model-version", default="gnn_v2")
    parser.add_argument("--embedding-version", default="gnn_emb_v2")
    parser.add_argument(
        "--dest", type=Path, default=Path("artifacts"), help="Local artifacts root."
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Publish local artifacts to S3 instead of downloading them.",
    )
    args = parser.parse_args()

    action = "Uploading" if args.upload else "Fetching"
    print(f"{action} {args.model_version} / {args.embedding_version} (bucket: {args.bucket})")

    runner = upload if args.upload else download
    failures = runner(args.bucket, args.model_version, args.embedding_version, args.dest)

    if failures:
        print(f"{failures} required artifact(s) failed", file=sys.stderr)
        return 1
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
