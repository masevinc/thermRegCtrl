#!/usr/bin/env bash
# run_export_wall_yplus.sh
#
# Scans BASE_DIR (or just one case folder under it, see below) for
# save_stage3a_5min.sim files (the 5-min checkpoint each case-setup macro
# saves), and for each one found runs STAR-CCM+ in batch mode with
# extract_wall_yplus.java to export Wall Y+ (area-average + max) on every
# boundary to a CSV named after the case folder. All CSVs land in a single
# yplus_exports/ folder.
#
# Usage:
#   ./run_export_wall_yplus.sh                  # all cases under BASE_DIR
#   ./run_export_wall_yplus.sh full_1st_ord      # just this one case folder
#
# Cases with no save_stage3a_5min.sim yet (e.g. full_2nd_ord_30deg_tilted,
# not started as of 2026-08-09) are silently skipped by `find` -- no
# special-casing needed here, they'll pick up automatically once that
# checkpoint exists and this script is re-run.
#
# BASE_DIR matches run_export_temp_csv.sh's own BASE_DIR exactly (confirmed
# 2026-08-09 via a directory listing screenshot: full_1st_ord/, full_2nd_ord/,
# full_cleaned_0804/, half_cleaned_0804/, fine_full_0805/, full_deltaT_0p5/,
# full_2nd_ord_30deg_tilted/, extract_temp_csv.java, run_export_temp_csv.sh,
# csv_exports/, logs/ all live directly under case_debug_0804/, not one level
# up under StarCCM_Test/ -- an earlier version of this script guessed wrong).
#
# Same caveat as run_export_temp_csv.sh: the starccm+ launch line below is
# a minimal placeholder -- license server, -power, -np, module load, etc.
# are cluster-specific and not guessed here; verify against run_case.sh.

set -euo pipefail

BASE_DIR="/dss/lxclscratch/08/go34zaw2/StarCCM_Test/case_debug_0804"
SIM_NAME="save_stage3a_5min.sim"
MACRO="$(cd "$(dirname "$0")" && pwd)/extract_wall_yplus.java"
OUT_DIR="${BASE_DIR}/yplus_exports"

# Optional: restrict to a single case folder, e.g. ./run_export_wall_yplus.sh full_1st_ord
CASE_NAME="${1:-}"
if [ -n "$CASE_NAME" ]; then
  SEARCH_DIR="${BASE_DIR}/${CASE_NAME}"
  if [ ! -d "$SEARCH_DIR" ]; then
    echo "ERROR: no such case folder: $SEARCH_DIR" >&2
    exit 1
  fi
else
  SEARCH_DIR="$BASE_DIR"
fi

mkdir -p "$OUT_DIR"

find "$SEARCH_DIR" -type f -name "$SIM_NAME" -print0 | while IFS= read -r -d '' simfile; do
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

if [ -n "$CASE_NAME" ] && [ -z "$(find "$SEARCH_DIR" -type f -name "$SIM_NAME")" ]; then
  echo "No $SIM_NAME found under $SEARCH_DIR -- nothing to process."
fi

echo "Done. CSV files are in: $OUT_DIR"
