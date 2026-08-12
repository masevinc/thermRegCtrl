"""
Compares dummy (thermal manikin test rig) temperature measurements against
CFD simulation results, per body region -- both are air-temperature-family
quantities (test-rig AIR probes vs. CFD sensor monitors), so this is a
direct apples-to-apples comparison, unlike the EQU-vs-JOS3-Tsk case (see
equ_comfort_from_test.py's docstring for why that one instead compares
derived comfort scores, not raw temperatures).

Ported from the original standalone script (same name, used to live in the
master's thesis data root with a hardcoded CASES loop and no --case/
--data-root CLI) into this repo's pipeline so it follows the same
conventions as run_jos3_from_cfd.py / equ_comfort_from_test.py.

Dummy measurement files contain 25 raw zone sensors (Sensor_<1..25>_EQU / _AIR).
The zone numbers correspond to the 1..25 zone numbering shown in the
"3D Comfort - Sensor definition" body diagram.

Two CFD sensor export formats are supported, auto-detected per file
(see detect_sim_format):
  - "legacy16": the original 16-sensor export (one shared "Time" column,
    positional UPPERCASE names like "Sensor_06_CORE_chest"). These don't
    correspond 1:1 to the real test rig's 25 zones -- the test rig's own
    software ("individual setting" table) groups its 25 raw zones into 16
    logical regions to match (some are averages of multiple zones, e.g.
    Thorax = zones 6, 7, 8, 9), see REGION_MAP.
  - "25zone": the newer export (added 2026-08-12, sensor positions updated
    to match the real test rig's own probe layout exactly), 25 sensors
    each with their own "Physical Time (s)" column (verified identical
    across sensors), names already matching the test rig's zone numbering
    1:1 (e.g. "Sensor_14_left_hand" <-> real test zone 14) -- no grouping/
    averaging needed, see load_sim_25zone/region_list_25zone.

If several iterations (v1, v2, ...) exist for a case, the script always
picks the file with the highest v<N> suffix (the most recent iteration).

For each region, the script:
  - computes the test-side average of the mapped zones (T_air) at every
    second of the test,
  - interpolates the sim-side matching sensor onto that same 1 Hz time
    axis,
  - plots both curves on top of each other,
  - computes error metrics (bias, MAE, RMSE, max absolute error) between
    sim and test.

Time alignment: the first valid row in the dummy file (the first row where
no _AIR channel equals the invalid sentinel -1000) is taken as t = 0,
matching the sim start time.

Axes in every plot:
  - X axis: elapsed time since test start [s] (t = 0 is the first valid
    dummy sample, see above)
  - Y axis: temperature [degC]
  - "Test" line: measured T_air, averaged over the region's mapped zones
  - "Sim" line: CFD monitor point for that region, interpolated onto the
    test's time axis

For CFD-only variants with no test rig run of their own (e.g. a
mesh-sensitivity case like coarse_min7), pass --real-case to compare
against a different case's real dummy_measurements data.

Usage:
    python3 compare_dummy_vs_sim.py --case min7
    python3 compare_dummy_vs_sim.py --case coarse_min7 --real-case min7
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INVALID_THRESHOLD = -900.0  # dummy sensors report -1000 when invalid/disconnected

DEFAULT_DATA_ROOT = os.path.expanduser("~/Documents/master_thesis_input_data")

# region_label, [test zone numbers (Sensor_<n>_AIR/EQU)], sim sensor column name fragment
REGION_MAP = [
    ("Scalp",            [1],          "Sensor_01_HEAD_top"),
    ("Face",              [2],          "Sensor_02_HEAD_front"),
    ("Left temple",       [3],          "Sensor_03_HEAD_left"),
    ("Right temple",      [4],          "Sensor_04_HEAD_right"),
    ("Neck",              [5],          "Sensor_05_CORE_upperBack"),
    ("Thorax",            [6, 7, 8, 9], "Sensor_06_CORE_chest"),
    ("Upper arm left",    [10],         "Sensor_07_SHOULDER_left"),
    ("Upper arm right",   [11],         "Sensor_08_SHOULDER_right"),
    ("Left hand",         [12, 14],     "Sensor_09_HAND_left"),
    ("Right hand",        [13, 15],     "Sensor_10_HAND_right"),
    ("Left thigh",        [16, 18],     "Sensor_11_LOWERBODY_leftUpperLeg"),
    ("Right thigh",       [17, 19],     "Sensor_12_LOWERBODY_rightUpperLeg"),
    ("Left shin",         [20, 22],     "Sensor_13_LOWERBODY_leftLowerLeg"),
    ("Right shin",        [21, 23],     "Sensor_14_LOWERBODY_rightLowerLeg"),
    ("Left foot",         [24],         "Sensor_15_LOWERBODY_leftFoot"),
    ("Right foot",        [25],         "Sensor_16_LOWERBODY_rightFoot"),
]

# 25zone-format column parser: raw header before the ":" split looks like
# "Sensor_01_head_scalp Monitor" or, for a couple of sensors, "Sensor_03_
# head_left 2 Monitor" (STAR-CCM+ appended " 2" to dedupe a monitor name
# clash) -- the (?:\s+\d+)? optionally eats that suffix.
NEW_FORMAT_COL_RE = re.compile(r"Sensor_(\d+)_([A-Za-z_]+)(?:\s+\d+)?\s+Monitor", re.IGNORECASE)


def find_dummy_file(data_root: str, case: str) -> str | None:
    matches = sorted(glob.glob(os.path.join(data_root, "dummy_measurements", f"*{case}*.csv")))
    return matches[0] if matches else None


def find_sim_file(data_root: str, case: str) -> str | None:
    """Returns the sim result file for the most recent iteration (highest
    v<N> suffix in the filename). Falls back to plain filename sort if no
    version suffix is present."""
    candidates = [
        p for p in glob.glob(os.path.join(data_root, "sim-results", case, f"*{case}*.csv"))
        if "velocity" not in os.path.basename(p).lower()
    ]
    if not candidates:
        return None

    def version_key(path: str) -> tuple[int, str]:
        match = re.search(r"_v(\d+)", os.path.basename(path))
        return (int(match.group(1)) if match else -1, path)

    return max(candidates, key=version_key)


def load_dummy(path: str, value_kind: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y.%m.%d %H:%M:%S")

    cols = [c for c in df.columns if c.endswith(f"_{value_kind}")]
    df[cols] = df[cols].where(df[cols] > INVALID_THRESHOLD, np.nan)

    valid_mask = df[cols].notna().all(axis=1)
    if not valid_mask.any():
        raise ValueError(f"{path}: no row has all {value_kind} channels valid")
    t0_idx = valid_mask.idxmax()
    t0 = df.loc[t0_idx, "Timestamp"]

    df = df.loc[t0_idx:].reset_index(drop=True)
    df["elapsed_s"] = (df["Timestamp"] - t0).dt.total_seconds()
    return df


def detect_sim_format(path: str) -> str:
    """Peeks at the CSV header only (nrows=0, cheap even on multi-MB
    files). Returns '25zone' or 'legacy16' -- see module docstring."""
    header = pd.read_csv(path, nrows=0).columns
    return "25zone" if any("Physical Time" in c for c in header) else "legacy16"


def load_sim_legacy(path: str) -> pd.DataFrame:
    """Older 16-sensor CFD export: one shared 'Time' column."""
    df = pd.read_csv(path)
    df.columns = [c.split(":")[0].replace("Temperatur - ", "").strip() for c in df.columns]
    return df


def load_sim_25zone(path: str) -> pd.DataFrame:
    """Newer 25-sensor CFD export. Each sensor has its own 'Physical Time
    (s)' column -- verified identical across all 25 on the first real
    export of this format (2026-08-12), so the first one found is used as
    the shared time axis; if a future export ever has per-sensor timing
    drift this would silently use only the first sensor's clock, worth
    re-checking if that's ever suspected. Output columns are named
    'z<NN>_<label>', zone number matching the real test rig's
    Sensor_<NN>_AIR/EQU numbering directly (see region_list_25zone)."""
    raw = pd.read_csv(path)
    time_col = next(c for c in raw.columns if "Physical Time" in c)
    out = pd.DataFrame({"Time": raw[time_col]})
    for c in raw.columns:
        if "Physical Time" in c:
            continue
        m = NEW_FORMAT_COL_RE.search(c)
        if not m:
            print(f"  [!] could not parse a zone number out of column: {c}")
            continue
        zone, label = int(m.group(1)), m.group(2).strip("_").lower()
        out[f"z{zone:02d}_{label}"] = raw[c]
    return out


def region_list_legacy(sim: pd.DataFrame) -> list[tuple[str, list[int], str | None]]:
    """(display label, real-test zones to average, resolved sim column)
    for the 16-region legacy format, resolved via REGION_MAP's fragment
    matching against this specific file's actual column names."""
    result = []
    for label, zones, sim_col_key in REGION_MAP:
        sim_col = next((c for c in sim.columns if sim_col_key in c), None)
        result.append((label, zones, sim_col))
    return result


def region_list_25zone(sim: pd.DataFrame) -> list[tuple[str, list[int], str | None]]:
    """Same shape as region_list_legacy, but 1:1 -- one real-test zone per
    CFD sensor, no averaging, derived directly from load_sim_25zone's own
    column names rather than a hand-maintained map."""
    result = []
    for c in sim.columns:
        if c == "Time":
            continue
        zone_str, _, label = c.partition("_")
        zone = int(zone_str[1:])
        result.append((f"{zone:02d} {label.replace('_', ' ').title()}", [zone], c))
    result.sort(key=lambda r: r[1][0])
    return result


def region_series(dummy: pd.DataFrame, zones: list[int], value_kind: str) -> pd.Series:
    cols = [f"Sensor_{z}_{value_kind}" for z in zones]
    return dummy[cols].mean(axis=1)


def resample_sim(sim: pd.DataFrame, sim_col: str, target_t: np.ndarray) -> np.ndarray:
    sim_t = sim["Time"].to_numpy()
    sim_v = sim[sim_col].to_numpy()
    out = np.interp(target_t, sim_t, sim_v)
    out[(target_t < sim_t.min()) | (target_t > sim_t.max())] = np.nan
    return out


def compare_case(data_root: str, case: str, real_case: str, value_kind: str) -> pd.DataFrame | None:
    dummy_path = find_dummy_file(data_root, real_case)
    sim_path = find_sim_file(data_root, case)

    if dummy_path is None:
        print(f"[{case}] no dummy measurement file found for real-case '{real_case}', skipping.")
        return None
    if sim_path is None:
        print(f"[{case}] no sim result file yet, skipping.")
        return None

    sim_format = detect_sim_format(sim_path)
    print(f"[{case}] dummy: {dummy_path}" + (f"  (real-case override: {real_case})" if real_case != case else ""))
    print(f"[{case}] sim:   {sim_path}  (format: {sim_format})")

    dummy = load_dummy(dummy_path, value_kind)
    if sim_format == "25zone":
        sim = load_sim_25zone(sim_path)
        region_list = region_list_25zone(sim)
    else:
        sim = load_sim_legacy(sim_path)
        region_list = region_list_legacy(sim)

    tag = case if real_case == case else f"{case}_vs_real-{real_case}"
    out_dir = os.path.join(data_root, "comparison_results", tag)
    os.makedirs(out_dir, exist_ok=True)

    target_t = dummy["elapsed_s"].to_numpy()
    n = len(region_list)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig_all, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), sharex=True)
    axes = axes.flatten()

    rows = []
    for i, (label, zones, sim_col) in enumerate(region_list):
        if sim_col is None:
            print(f"  [!] sim column not found for region: {label}")
            continue

        test_series = region_series(dummy, zones, value_kind)
        sim_series = resample_sim(sim, sim_col, target_t)

        err = sim_series - test_series.to_numpy()
        valid = ~np.isnan(err)
        bias = np.nanmean(err) if valid.any() else np.nan
        mae = np.nanmean(np.abs(err)) if valid.any() else np.nan
        rmse = np.sqrt(np.nanmean(err**2)) if valid.any() else np.nan
        max_abs = np.nanmax(np.abs(err)) if valid.any() else np.nan

        rows.append({
            "region": label,
            "test_zones_averaged": ",".join(map(str, zones)),
            "sim_sensor_column": sim_col,
            "bias_sim_minus_test_degC": bias,
            "mae_degC": mae,
            "rmse_degC": rmse,
            "max_abs_error_degC": max_abs,
            "n_compared_seconds": int(valid.sum()),
        })

        # individual per-region plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(target_t, test_series, label=f"Test measurement ({value_kind}, zones {zones})", color="tab:blue")
        ax.plot(target_t, sim_series, label=f"CFD sim ({sim_col})", color="tab:red")
        ax.set_title(f"Case {tag} - {label}: test vs. CFD simulation")
        ax.set_xlabel("Elapsed time since test start [s]")
        ax.set_ylabel("Temperature [degC]")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{i+1:02d}_{label.replace(' ', '_')}.png"), dpi=150)
        plt.close(fig)

        # subplot in the combined overview figure
        ax_grid = axes[i]
        ax_grid.plot(target_t, test_series, label="Test", color="tab:blue", linewidth=1)
        ax_grid.plot(target_t, sim_series, label="Sim", color="tab:red", linewidth=1)
        ax_grid.set_title(label, fontsize=9)
        ax_grid.set_xlabel("Time [s]", fontsize=8)
        ax_grid.set_ylabel("Temp [degC]", fontsize=8)
        ax_grid.grid(alpha=0.3)

    for j in range(len(region_list), len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig_all.legend(handles, labels, loc="upper right")
    fig_all.suptitle(
        f"Case {tag}: test measurement ({value_kind}) vs. CFD simulation, all regions\n"
        "X axis: elapsed time since test start [s]  |  Y axis: temperature [degC]"
    )
    fig_all.tight_layout(rect=(0, 0, 1, 0.94))
    fig_all.savefig(os.path.join(out_dir, "00_overview_all_regions.png"), dpi=150)
    plt.close(fig_all)

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(out_dir, f"error_summary_{tag}.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"[{case}] plots and summary written to -> {out_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default="min7", help="Case name, e.g. min7, min0, pls7, coarse_min7")
    parser.add_argument("--real-case", default=None,
                         help="Dummy-measurement case to compare against, if different from --case "
                              "(e.g. a mesh-sensitivity CFD variant with no test rig run of its own). "
                              "Defaults to --case.")
    parser.add_argument("--value-kind", default="AIR", choices=["AIR", "EQU"],
                         help="Which dummy sensor channel family to compare against the CFD air-temp sensors "
                              "(AIR = air-temperature probes, directly comparable to CFD monitor points; "
                              "EQU = equivalent/heated-probe temperature, a different physical quantity, see "
                              "equ_comfort_from_test.py's docstring)")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()

    compare_case(args.data_root, args.case, args.real_case or args.case, args.value_kind)


if __name__ == "__main__":
    main()
