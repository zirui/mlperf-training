#!/bin/bash

set -euxo pipefail

export PREPROCESSED_PATH='/data/mlperf_flux1/'

if [ ! -d "$PREPROCESSED_PATH" ]; then
    mkdir -p "$PREPROCESSED_PATH"
fi
curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh > mlc-r2-downloader.sh

urls=(
  "https://training.mlcommons-storage.org/metadata/flux-1-cc12m-preprocessed.uri"
  "https://training.mlcommons-storage.org/metadata/flux-1-coco-preprocessed.uri"
  "https://training.mlcommons-storage.org/metadata/flux-1-empty-encodings.uri"
)

pids=()
for url in "${urls[@]}"; do
  bash mlc-r2-downloader.sh -d "$PREPROCESSED_PATH" "$url" &
  pids+=("$!")
done

# Wait for each and fail fast if any job fails
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "A download (PID $pid) failed." >&2
    exit 1
  fi
done

echo "All downloads completed successfully."

rm mlc-r2-downloader.sh