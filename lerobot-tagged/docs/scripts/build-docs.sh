#!/usr/bin/env bash
# build-docs.sh — Run pdoc3 for Python API docs, then build Starlight.
#
# Usage (from lerobot-tagged/docs/):
#   bash scripts/build-docs.sh
#
# Requirements:
#   pip install pdoc3
#   npm install   (in this directory)

set -euo pipefail

DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_SRC="$DOCS_DIR/../python/src/lerobot_tagged"
STATIC_OUT="$DOCS_DIR/public/python-api"

echo "==> Generating Python API docs with pdoc3..."
mkdir -p "$STATIC_OUT"
pdoc3 \
  --html \
  --output-dir "$STATIC_OUT" \
  --force \
  "$PYTHON_SRC"

# pdoc3 nests output under the module name; lift it up so
# /python-api/index.html works directly
if [ -d "$STATIC_OUT/lerobot_tagged" ]; then
  cp -r "$STATIC_OUT/lerobot_tagged/." "$STATIC_OUT/"
  rm -rf "$STATIC_OUT/lerobot_tagged"
fi

echo "==> Building Starlight..."
cd "$DOCS_DIR"
npm run build

echo ""
echo "Done — output in $DOCS_DIR/dist/"
