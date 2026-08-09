#!/usr/bin/env bash
# run_export_wall_yplus.sh
#
# Scans BASE_DIR for save_stage3a_5min.sim files (the 5-min checkpoint each
# case-setup macro saves), and for each one found runs STAR-CCM+ in batch
# mode with extract_wall_yplus.java to export Wall Y+ (area-average + max)
# per manikin-skin boundary to a CSV named after the case folder. All CSVs
# land in a single yplus_exports/ folder.
#
# Cases with no save_stage3a_5min.sim yet (e.g. full_2nd_ord_30deg_tilted,
# not started as of 2026-08-09) are silently skipped by `find` -- no
# special-casing needed here, they'll pick up automatically once that
# checkpoint exists and this script is re-run.
#
# Adjust BASE_DIR below -- this is a guess based on run_export_temp_csv.sh's
# BASE_DIR (which pointed at a "case_debug_0804" subfolder specifically);
# verify it matches where coarse_full_0805/, fine_full_0805/, etc. actually
# live before running (`ls "$BASE_DIR"` should list them directly).
#
# Same caveat as run_export_temp_csv.sh: the starccm+ launch line below is
# a minimal placeholder -- license server, -power, -np, module load, etc.
# are cluster-specific and not guessed here; verify against run_case.sh.

set -euo pipefail

BASE_DIR="/dss/lxclscratch/08/go34zaw2/StarCCM_Test"
SIM_NAME="save_stage3a_5min.sim"
MACRO="$(cd "$(dirname "$0")" && pwd)/extract_wall_yplus.java"
OUT_DIR="${BASE_DIR}/yplus_exports"

mkdir -p "$OUT_DIR"

find "$BASE_DIR" -type f -name "$SIM_NAME" -print0 | while IFS= read -r -d '' simfile; do
  case_dir="$(dirname "$simfile")"
  case_name="$(basename "$case_dir")"
  out_csv="${OUT_DIR}/${case_name}_yplus.csv"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing case: $case_name"
  echo "  sim file : $simfile"
  echo "  csv out  : $out_csv"

  # NOTE: verify this invocation against run_case.sh -- module load,
  # license flags (-power / -licpath / -podkey), -np etc. may be required
  # on this cluster and are not guessed here.
  STARCCM_YPLUS_CSV_OUT="$out_csv" starccm+ -batch "$MACRO" "$simfile"
done

echo "Done. CSV files are in: $OUT_DIR"
