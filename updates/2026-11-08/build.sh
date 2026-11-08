#!/bin/bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
COMMIT=6ab9bd05cb84b538a6595d634f3e337ac8a80856
if ! docker image inspect ulmus/qwen38-vllm:0.27.1-cu129 >/dev/null 2>&1; then
 docker build -f "$ROOT/docker/Dockerfile.cu129" -t ulmus/qwen38-vllm:0.27.1-cu129 "$ROOT"
fi
if [ ! -d "$HERE/upstream/.git" ]; then git init -q "$HERE/upstream"; fi
git -C "$HERE/upstream" fetch --depth 1 https://github.com/syv-ai/qwen38-27b-rtx3090.git "$COMMIT"
# A fresh directory prevents untracked files from entering the pinned source export.
CONTEXT=$(mktemp -d "${TMPDIR:-/tmp}/ulmus-27b-build.XXXXXXXX")
trap 'rm -rf -- "$CONTEXT"' EXIT
mkdir "$CONTEXT/source"
git -C "$HERE/upstream" archive "$COMMIT" | tar -x -C "$CONTEXT/source"
cp "$HERE/Dockerfile" "$HERE/requirements-cu129.txt" "$CONTEXT/"
docker build -t ulmus/qwen38-vllm:0.32.0-cu129 "$CONTEXT"
