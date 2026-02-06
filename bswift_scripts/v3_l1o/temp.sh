#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash fit-sbm.sh ALL_GRAPHS.txt START_IDX END_IDX SBM DC WAIT TOTAL_SAMPLES B GAMMA [SEED_BASE] [J]
#
# Example:
#   bash fit-sbm.sh all_graphs.txt 0 99 d True 0 5000 12 1.0 1000 8
#
# Notes:
# - J = max parallel jobs (default 4)
# - Assumes this .sh sits next to the Python file; adjust PY path if needed.

ALL_GRAPHS="${1:?all_graphs file}"
START_IDX="${2:?start idx}"
END_IDX="${3:?end idx}"
SBM="${4:?sbm code (a|d|o|h|m)}"
DC="${5:?True|False}"
WAIT="${6:?int}"
TOTAL="${7:?total_samples int}"
B="${8:?B int}"
GAMMA="${9:?gamma float}"
SEED_BASE="${10:-1}"
J="${11:-4}"

# Path to your Python script (adjust if located elsewhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/02a-graph-individual-binary_desc-fit-sbm.py"

export ALL_GRAPHS SBM DC WAIT TOTAL B GAMMA SEED_BASE PY

# Prefer GNU parallel; fallback to xargs if not present.
if command -v parallel >/dev/null 2>&1; then
  seq "${START_IDX}" "${END_IDX}" | parallel -j "${J}" --lb '
    GRAPH_IDX={}
    SEED=$((SEED_BASE + GRAPH_IDX))
    python "$PY" "$ALL_GRAPHS" "$GRAPH_IDX" "$SBM" "$DC" "$WAIT" "$TOTAL" "$B" "$GAMMA" "$SEED"
  '
else
  # Fallback using xargs -P (no extra dependency)
  seq "${START_IDX}" "${END_IDX}" | xargs -I{} -P "${J}" bash -lc '
    GRAPH_IDX="$1"
    SEED=$((SEED_BASE + GRAPH_IDX))
    python "$PY" "$ALL_GRAPHS" "$GRAPH_IDX" "$SBM" "$DC" "$WAIT" "$TOTAL" "$B" "$GAMMA" "$SEED"
  ' _ {}
fi
