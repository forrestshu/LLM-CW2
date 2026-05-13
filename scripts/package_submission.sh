#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMISSION_DIR="$ROOT_DIR/submission"

mkdir -p "$SUBMISSION_DIR"
rm -f "$SUBMISSION_DIR/code.zip" "$SUBMISSION_DIR/DTS407TC_A2-report.pdf" "$SUBMISSION_DIR/DTS407TC_A2-ppt.pdf"

cd "$ROOT_DIR"
zip -r "$SUBMISSION_DIR/code.zip" \
  backend \
  frontend \
  evaluation \
  tests \
  scripts \
  README.md \
  pyproject.toml \
  uv.lock \
  package.json \
  package-lock.json \
  vite.config.js \
  tailwind.config.js \
  postcss.config.js \
  .env.example \
  .python-version \
  -x "*/__pycache__/*" \
  -x "*/.DS_Store" \
  -x "evaluation/results/*"

cp "$ROOT_DIR/docs/report/dts407tc_a2_report.pdf" "$SUBMISSION_DIR/DTS407TC_A2-report.pdf"
cp "$ROOT_DIR/docs/presentation/dts407tc_a2_ppt.pdf" "$SUBMISSION_DIR/DTS407TC_A2-ppt.pdf"

echo "Created $SUBMISSION_DIR/code.zip"
