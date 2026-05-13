#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="$ROOT_DIR/docs/report"
PPT_DIR="$ROOT_DIR/docs/presentation"

xelatex -interaction=nonstopmode -halt-on-error -output-directory "$REPORT_DIR" "$REPORT_DIR/dts407tc_a2_report.tex"

pandoc "$PPT_DIR/dts407tc_a2_slides.md" \
  -t beamer \
  -o "$PPT_DIR/dts407tc_a2_ppt.pdf" \
  --pdf-engine=xelatex

echo "Exported report and presentation PDFs."

