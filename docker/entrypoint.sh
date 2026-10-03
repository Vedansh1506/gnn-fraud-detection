#!/bin/sh
# Fetch the pinned model artifacts, then start the app.
#
# Artifacts are git-ignored and ~115 MB, so they are not in the image: baking
# them in would make every retrain an image rebuild, and would tie the artifact
# version to the image tag. Instead the container pulls the versions named by
# MODEL_VERSION / EMBEDDING_VERSION at start — the same pinned-version rule the
# serving code follows, applied to delivery.
#
# Set S3_ARTIFACTS_BUCKET to fetch from S3. Leave it unset when artifacts are
# supplied another way (a bind mount locally, or an EBS volume), in which case
# this only verifies they are actually there.
set -eu

ARTIFACTS_DIR="${ARTIFACTS_DIR:-/app/artifacts}"
MODEL_VERSION="${MODEL_VERSION:-gnn_v2}"
EMBEDDING_VERSION="${EMBEDDING_VERSION:-gnn_emb_v2}"

model_dir="${ARTIFACTS_DIR}/models/${MODEL_VERSION}"
embedding_dir="${ARTIFACTS_DIR}/embeddings/${EMBEDDING_VERSION}"

if [ -n "${S3_ARTIFACTS_BUCKET:-}" ]; then
  echo "entrypoint: syncing artifacts from s3://${S3_ARTIFACTS_BUCKET}"
  # python -m, not the AWS CLI: boto3 is already a serving dependency, so this
  # costs nothing extra in the image.
  python -m scripts.fetch_artifacts \
    --bucket "${S3_ARTIFACTS_BUCKET}" \
    --model-version "${MODEL_VERSION}" \
    --embedding-version "${EMBEDDING_VERSION}" \
    --dest "${ARTIFACTS_DIR}"
else
  echo "entrypoint: S3_ARTIFACTS_BUCKET unset - expecting artifacts to be mounted"
fi

# Fail fast and say exactly what is missing. Without this the app starts, loads
# nothing, and every score returns 503 with a far less obvious cause.
for required in "${model_dir}/model.json" "${model_dir}/feature_spec.json" \
                "${model_dir}/account_features.parquet" \
                "${embedding_dir}/embeddings.parquet"; do
  if [ ! -f "${required}" ]; then
    echo "entrypoint: FATAL - missing required artifact: ${required}" >&2
    echo "entrypoint: set S3_ARTIFACTS_BUCKET, or mount ${ARTIFACTS_DIR}" >&2
    exit 1
  fi
done

echo "entrypoint: artifacts present (${MODEL_VERSION} / ${EMBEDDING_VERSION})"
exec "$@"
