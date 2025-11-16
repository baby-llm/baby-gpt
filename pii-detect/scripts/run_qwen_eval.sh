#!/usr/bin/env bash
# Wrapper to execute the Qwen eval script and save results to a log file.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/run_qwen_eval.py"
DEFAULT_LOG_DIR="${PROJECT_ROOT}/logs"
DEFAULT_LOG_FILE="${DEFAULT_LOG_DIR}/qwen_eval_$(date +%Y%m%d_%H%M%S).log"

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "Python eval script not found at ${PY_SCRIPT}" >&2
  exit 1
fi

LOG_FILE="${1:-${DEFAULT_LOG_FILE}}"
shift || true

mkdir -p "$(dirname "${LOG_FILE}")"

python "${PY_SCRIPT}" "$@" | tee "${LOG_FILE}"

echo "Saved evaluation output to ${LOG_FILE}"
