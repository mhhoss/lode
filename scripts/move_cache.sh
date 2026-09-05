#!/usr/bin/env bash
# Moves a locally-prepared ONNX embedding model directory into Lode's model
# cache, so OnnxTextEmbeddingAdapter can find it at the expected path.
#
# The adapter expects <model-dir>/onnx/ to contain:
#   model.onnx, tokenizer.json, config.json, tokenizer_config.json
#
# This script does not download or convert models — it only relocates an
# already-prepared model directory into the cache.
#
# Usage: bash scripts/move_cache.sh <source-model-dir> [model-name]

set -euo pipefail

SOURCE_DIR="${1:?Usage: move_cache.sh <source-model-dir> [model-name]}"
MODEL_NAME="${2:-$(basename "${SOURCE_DIR}")}"
CACHE_DIR="${HOME}/.cache/lode/models/${MODEL_NAME}"

echo "Moving ${SOURCE_DIR} -> ${CACHE_DIR} ..."

mkdir -p "$(dirname "${CACHE_DIR}")"
mv "${SOURCE_DIR}" "${CACHE_DIR}"

echo "Done. Model available at: ${CACHE_DIR}"
