"""
Classifies real thermal-manikin test measurements (equivalent temperature,
Sensor_<n>_EQU) into ISO 14505-2 Annex D comfort zones (see
iso14505_bands.py), then plots that alongside the JOS-3-driven
sensation/comfort scores from jos3_comfort_from_cfd.py.

This compares two independently derived comfort verdicts for the same
scenario -- one from a standards-based classification of real test
measurements, one from a physiological model (JOS-3 + UCB-Zhang) driven by
the CFD run -- rather than comparing raw temperatures, which are not
directly comparable across these two data sources (see project notes: EQU
and JOS-3's Tsk are different physical quantities).

Run, in order:
    run_jos3_from_cfd.py --case <case>
    jos3_comfort_from_cfd.py --case <case>
    equ_comfort_from_test.py --case <case> --season winter

For CFD-only variants that don't have their own physical test rig run (e.g.
a mesh-sensitivity case like coarse_min7, meant to represent the same
ambient scenario as an existing case), pass --real-case to point at the
real dummy_measurements file to compare against while still reading/writing
this case's own JOS-3 results:
    python3 equ_comfort_from_test.py --case coarse_min7 --real-case min7 --season winter

Usage:
    python3 equ_comfort_from_test.py --case min7 --season winter
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iso14505_bands import BANDS_BY_SEASON, classify, zone_to_sensation_scale

DEFAULT_DATA_ROOT = os.path.expanduser("~/Documents/master_thesis_input_data")
INVALID_THRESHOLD = -900.0
VALUE_KIND = "EQU"

# Test-rig zone(s) -> ISO 14505-2 body region. Multiple test zones average
# into one ISO region where the test rig has finer resolution than the
# standard's chart (e.g. left/right are the same ISO region; temples have
# no ISO row of their own and are treated as "face").
ISO_REGION_MAP: dict[str, list[int]] = {
    "scalp":     [1],
    "face":      [2, 3, 4],          # face + left/right temple
    "chest":     [5, 6, 7, 8, 9],    # neck (no ISO row of its own) + thorax
    "upperArm":  [10, 11],
    "lowerArm":  [10, 11],           # no dedicated forearm sensor, reuse upper arm
    "hand":      [12, 13, 14, 15],
    "thigh":     [16, 17, 18, 19],
    "calf":      [20, 21, 22, 23],
    "foot":      [24, 25],
}
ALL_ZONES = list(range(1, 26))


def find_dummy_file(data_root: str, case: str) -> str:
    matches = sorted(glob.glob(os.path.join(data_root, "dummy_measurements", f"*{case}*.csv")))
    if not matches:
        raise FileNotFoundError(f"No dummy measurement file found for case '{case}'")
    return matches[0]


def find_jos3_comfort_files(data_root: str, case: str) -> tuple[str, str]:
    comfort_dir = os.path.join(data_root, "jos3_results", case, "comfort")
    sensation_path = os.path.join(comfort_dir, "overallSensation.csv")
    comfort_path = os.path.join(comfort_dir, "overallComfort.csv")
    for path in (sensation_path, comfort_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No JOS-3 comfort results found at {path} -- run "
                f"'run_jos3_from_cfd.py --case {case}' then "
                f"'jos3_comfort_from_cfd.py --case {case}' first."
            )
    return sensation_path, comfort_path


def load_dummy(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y.%m.%d %H:%M:%S")

    equ_cols = [f"Sensor_{z}_{VALUE_KIND}" for z in ALL_ZONES]
    df[equ_cols] = df[equ_cols].where(df[equ_cols] > INVALID_THRESHOLD, np.nan)

    valid_mask = df[equ_cols].notna().all(axis=1)
    if not valid_mask.any():
        raise ValueError(f"{path}: no row has all {VALUE_KIND} channels valid")
    t0_idx = valid_mask.idxmax()
    t0 = df.loc[t0_idx, "Timestamp"]

    df = df.loc[t0_idx:].reset_index(drop=True)
    df["elapsed_s"] = (df["Timestamp"] - t0).dt.total_seconds()
    return df


def classify_regions(dummy: pd.DataFrame, season: str) -> pd.DataFrame:
    zones = pd.DataFrame({"elapsed_s": dummy["elapsed_s"]})
    for region, test_zones in ISO_REGION_MAP.items():
        cols = [f"Sensor_{z}_{VALUE_KIND}" for z in test_zones]
        teq = dummy[cols].mean(axis=1)
        zones[region] = teq.apply(lambda v, r=region: classify(v, r, season) if pd.notna(v) else np.nan)

    all_equ_cols = [f"Sensor_{z}_{VALUE_KIND}" for z in ALL_ZONES]
    whole_body_teq = dummy[all_equ_cols].mean(axis=1)
    zones["wholeBody"] = whole_body_teq.apply(lambda v: classify(v, "wholeBody", season) if pd.notna(v) else np.nan)

    region_cols = list(ISO_REGION_MAP.keys()) + ["wholeBody"]
    zones["overall_zone"] = zones[region_cols].mean(axis=1)
    return zones


def resample_to(series_df: pd.DataFrame, value_col: str, target_t: np.ndarray) -> np.ndarray:
    t = series_df["time"].to_numpy() if "time" in series_df.columns else series_df["Time"].to_numpy()
    v = series_df[value_col].to_numpy()
    out = np.interp(target_t, t, v)
    out[(target_t < t.min()) | (target_t > t.max())] = np.nan
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default="min7", help="Case name, e.g. min7, min0, pls7")
    parser.add_argument("--real-case", default=None,
                         help="Dummy-measurement case to compare against, if different from --case "
                              "(e.g. a mesh-sensitivity CFD variant with no test rig run of its own). "
                              "Defaults to --case.")
    parser.add_argument("--season", default="winter", choices=list(BANDS_BY_SEASON.keys()),
                         help="Which ISO 14505-2 comfort chart to classify against")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    real_case = args.real_case or args.case

    dummy_path = find_dummy_file(args.data_root, real_case)
    sensation_path, comfort_path = find_jos3_comfort_files(args.data_root, args.case)
    print(f"[{args.case}] dummy EQU:      {dummy_path}"
          + (f"  (real-case override: {real_case})" if real_case != args.case else ""))
    print(f"[{args.case}] JOS-3 sensation: {sensation_path}")
    print(f"[{args.case}] JOS-3 comfort:   {comfort_path}")

    dummy = load_dummy(dummy_path)
    zones = classify_regions(dummy, args.season)

    jos3_sensation = pd.read_csv(sensation_path)
    jos3_comfort = pd.read_csv(comfort_path)

    target_t = dummy["elapsed_s"].to_numpy()
    jos3_sensation_r = resample_to(jos3_sensation, "overallSensation", target_t)
    jos3_comfort_r = resample_to(jos3_comfort, "overallComfort", target_t)
    equ_scale = zones["overall_zone"].apply(zone_to_sensation_scale).to_numpy()

    out_dir = os.path.join(args.data_root, "jos3_results", args.case, "vs_test_comfort")
    os.makedirs(out_dir, exist_ok=True)
    tag = args.case if real_case == args.case else f"{args.case}_vs_real-{real_case}"

    zones.to_csv(os.path.join(out_dir, f"equ_iso14505_zones_{tag}_{args.season}.csv"), index=False)
    pd.DataFrame({
        "elapsed_s": target_t,
        "equ_overall_zone": zones["overall_zone"],
        "equ_overall_on_sensation_scale": equ_scale,
        "jos3_overall_sensation": jos3_sensation_r,
        "jos3_overall_comfort": jos3_comfort_r,
    }).to_csv(os.path.join(out_dir, f"comparison_{tag}_{args.season}.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    real_label = "Real test" if real_case == args.case else f"Real test ({real_case})"
    ax.plot(target_t / 60, equ_scale, label=f"{real_label} (EQU -> ISO 14505-2 zone, whole body)", color="tab:orange")
    ax.plot(target_t / 60, jos3_sensation_r, label="JOS-3 overall sensation (Berkeley model)", color="tab:red", alpha=0.8)
    ax.plot(target_t / 60, jos3_comfort_r, label="JOS-3 overall comfort (Berkeley model)", color="tab:blue", alpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("-4 (cold/uncomfortable) .. 0 (neutral) .. +4 (hot/uncomfortable)")
    title_case = args.case if real_case == args.case else f"{args.case} (vs. real test {real_case})"
    ax.set_title(
        f"Case {title_case} ({args.season}): real-test ISO 14505-2 comfort zone vs. JOS-3-predicted comfort"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"comparison_{tag}_{args.season}.png"), dpi=150)
    plt.close(fig)

    print(f"[{args.case}] ISO 14505-2 zone classification and JOS-3 comparison written to -> {out_dir}")


if __name__ == "__main__":
    main()
