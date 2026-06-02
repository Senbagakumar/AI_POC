#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.mplconfig}"

SKIP_INSTALL=0
ONLY_PART=""

usage() {
  cat <<'EOF'
Usage:
  ./run_all.sh
  ./run_all.sh --skip-install
  ./run_all.sh --part PART_3

Options:
  --skip-install     Reuse the existing virtual environment without reinstalling requirements.
  --part PART_NAME   Run only one part. Allowed values: PART_1, PART_2, PART_3, PART_4
  -h, --help         Show this help text.
EOF
}

log() {
  printf '\n==> %s\n' "$*"
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

dataset_exists() {
  [[ -d "$ROOT_DIR/d2c churn data package/d2c churn data package" ]] \
    || [[ -d "$ROOT_DIR/PART_1/data" ]] \
    || [[ -d "$ROOT_DIR/PART_2/data" ]] \
    || [[ -d "$ROOT_DIR/PART_3/data" ]] \
    || [[ -d "$ROOT_DIR/PART_4/data" ]]
}

ensure_file() {
  local path="$1"
  [[ -e "$path" ]] || die "Expected artifact missing: $path"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-install)
      SKIP_INSTALL=1
      ;;
    --part)
      shift
      [[ $# -gt 0 ]] || die "--part requires a value"
      ONLY_PART="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
  shift
done

if [[ -n "$ONLY_PART" ]]; then
  case "$ONLY_PART" in
    PART_1|PART_2|PART_3|PART_4)
      ;;
    *)
      die "Invalid part: $ONLY_PART"
      ;;
  esac
fi

dataset_exists || die "Dataset not found. Keep the package under './d2c churn data package/d2c churn data package' or copy it into a part-local data/ folder."

mkdir -p "$MPLCONFIGDIR"
export MPLCONFIGDIR
export PYTHONUNBUFFERED=1

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  log "Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python"

install_requirements() {
  local part="$1"
  log "Installing dependencies for $part"
  "$PIP" install -r "$ROOT_DIR/$part/requirements.txt"
}

run_part_1() {
  log "Running PART_1"
  (
    cd "$ROOT_DIR/PART_1"
    "$PYTHON" build_part1.py
  )

  ensure_file "$ROOT_DIR/PART_1/eda_audit.ipynb"
  ensure_file "$ROOT_DIR/PART_1/data_quality_report.md"
  ensure_file "$ROOT_DIR/PART_1/business_memo.md"
  ensure_file "$ROOT_DIR/PART_1/charts"

  local chart_count
  chart_count="$(find "$ROOT_DIR/PART_1/charts" -maxdepth 1 -name '*.png' | wc -l | tr -d ' ')"
  [[ "$chart_count" -ge 6 ]] || die "PART_1 expected at least 6 charts, found $chart_count"
}

run_part_2() {
  log "Running PART_2"
  (
    cd "$ROOT_DIR/PART_2"
    "$PYTHON" build_part2.py
  )

  ensure_file "$ROOT_DIR/PART_2/rfm_segmentation.ipynb"
  ensure_file "$ROOT_DIR/PART_2/segments.csv"
  ensure_file "$ROOT_DIR/PART_2/retention_strategy.md"
  ensure_file "$ROOT_DIR/PART_2/manual_review_cases.md"

  grep -q '^customer_id,segment_name,' "$ROOT_DIR/PART_2/segments.csv" \
    || die "PART_2 segments.csv is missing the expected header prefix"
}

run_part_3() {
  log "Running PART_3"
  (
    cd "$ROOT_DIR/PART_3"
    "$PYTHON" train_churn_model.py
  )

  ensure_file "$ROOT_DIR/PART_3/churn_model.ipynb"
  ensure_file "$ROOT_DIR/PART_3/model.pkl"
  ensure_file "$ROOT_DIR/PART_3/metrics.json"
  ensure_file "$ROOT_DIR/PART_3/error_analysis.md"
  ensure_file "$ROOT_DIR/PART_3/model_card.md"
}

run_part_4() {
  log "Building PART_4 model artifact"
  (
    cd "$ROOT_DIR/PART_4"
    "$PYTHON" train_model.py
  )

  ensure_file "$ROOT_DIR/PART_4/model/model.pkl"
  ensure_file "$ROOT_DIR/PART_4/model/model_metadata.json"

  log "Running PART_4 tests"
  (
    cd "$ROOT_DIR/PART_4"
    "$PYTHON" -m pytest tests -q
  )
}

run_part() {
  local part="$1"

  if [[ "$SKIP_INSTALL" -eq 0 ]]; then
    install_requirements "$part"
  fi

  case "$part" in
    PART_1)
      run_part_1
      ;;
    PART_2)
      run_part_2
      ;;
    PART_3)
      run_part_3
      ;;
    PART_4)
      run_part_4
      ;;
    *)
      die "Unsupported part: $part"
      ;;
  esac
}

parts=(PART_1 PART_2 PART_3 PART_4)
if [[ -n "$ONLY_PART" ]]; then
  parts=("$ONLY_PART")
fi

for part in "${parts[@]}"; do
  run_part "$part"
done

log "All requested parts completed successfully."
