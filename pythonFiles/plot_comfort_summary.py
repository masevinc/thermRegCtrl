"""
Generates the two summary visualizations from comfort_plots.py using this
repo's pipeline output:
  1. PPD vs. DTS curve, from the JOS-3+Berkeley overall sensation history
     (jos3_comfort_from_cfd.py's overallSensation.csv).
  2. Body-segment equivalent-temperature profile vs. ISO 14505-2 comfort
     zones, comparing JOS-3's predicted operative temperature (To) against
     the real EQU test measurement, at the end of the transient.

Run after run_jos3_from_cfd.py and jos3_comfort_from_cfd.py.

For CFD-only variants with no test rig run of their own (e.g. a
mesh-sensitivity case), pass --real-case to compare against a different
case's real dummy_measurements data (see equ_comfort_from_test.py).

Usage:
    python3 plot_comfort_summary.py --case min7 --season winter
    python3 plot_comfort_summary.py --case coarse_min7 --real-case min7 --season winter
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from comfort_plots import SEGMENT_ORDER, plot_ppd_curve, plot_segment_profile
from equ_comfort_from_test import find_dummy_file, load_dummy

DEFAULT_DATA_ROOT = os.path.expanduser("~/Documents/master_thesis_input_data")

# JOS-3 segment -> dummy test-rig EQU zone(s); same correspondence used
# throughout this pipeline (run_jos3_from_cfd.py's SEGMENT_MAP /
# equ_comfort_from_test.py's ISO_REGION_MAP).
JOS3_TO_TEST_ZONES: dict[str, list[int]] = {
    "Head": [1], "Neck": [5], "Chest": [6, 7, 8, 9], "Back": [6, 7, 8, 9],
    "Pelvis": [16, 17, 18, 19],
    "LShoulder": [10], "LArm": [10], "LHand": [12, 14],
    "RShoulder": [11], "RArm": [11], "RHand": [13, 15],
    "LThigh": [16, 18], "LLeg": [20, 22], "LFoot": [24],
    "RThigh": [17, 19], "RLeg": [21, 23], "RFoot": [25],
}


def jos3_profile_at(jos3_df: pd.DataFrame, time_s: float) -> dict[str, float]:
    idx = (jos3_df["Time"] - time_s).abs().idxmin()
    row = jos3_df.loc[idx]
    return {seg: row[f"To_{seg}"] for seg in SEGMENT_ORDER}


def test_profile_at(dummy: pd.DataFrame, elapsed_s: float, window_s: float = 10.0) -> dict[str, float]:
    mask = (dummy["elapsed_s"] >= elapsed_s - window_s) & (dummy["elapsed_s"] <= elapsed_s + window_s)
    window = dummy.loc[mask]
    profile = {}
    for seg, zones in JOS3_TO_TEST_ZONES.items():
        cols = [f"Sensor_{z}_EQU" for z in zones]
        profile[seg] = window[cols].mean(axis=1).mean()
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default="min7")
    parser.add_argument("--real-case", default=None,
                         help="Dummy-measurement case to compare against, if different from --case. "
                              "Defaults to --case.")
    parser.add_argument("--season", default="winter", choices=["summer", "winter"])
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    real_case = args.real_case or args.case

    case_dir = os.path.join(args.data_root, "jos3_results", args.case)
    jos3_path = os.path.join(case_dir, f"jos3_prediction_{args.case}.csv")
    sensation_path = os.path.join(case_dir, "comfort", "overallSensation.csv")
    for p in (jos3_path, sensation_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found -- run run_jos3_from_cfd.py and jos3_comfort_from_cfd.py first")

    jos3_df = pd.read_csv(jos3_path)
    sensation_df = pd.read_csv(sensation_path)
    dummy = load_dummy(find_dummy_file(args.data_root, real_case))

    out_dir = os.path.join(case_dir, "summary_plots")
    os.makedirs(out_dir, exist_ok=True)
    tag = args.case if real_case == args.case else f"{args.case}_vs_real-{real_case}"
    title_case = args.case if real_case == args.case else f"{args.case} (vs. real test {real_case})"

    # 1) PPD vs DTS: highlight a few points along the transient
    t_end = sensation_df["time"].iloc[-1]
    checkpoints = {"t=2 min": 120, "t=15 min": 900, f"t={int(t_end / 60)} min (end)": t_end}
    highlight = {}
    for label, t in checkpoints.items():
        idx = (sensation_df["time"] - t).abs().idxmin()
        highlight[label] = sensation_df["overallSensation"].iloc[idx]
    plot_ppd_curve(
        os.path.join(out_dir, f"ppd_dts_{tag}.png"),
        highlight=highlight,
        title=f"Case {title_case}: JOS-3-driven overall sensation (DTS) vs. predicted dissatisfaction (PPD)",
    )

    # 2) Body-segment equivalent-temperature profile: JOS-3 (To) vs. real test (EQU), at the end of the transient
    jos3_profile = jos3_profile_at(jos3_df, t_end)
    test_profile = test_profile_at(dummy, t_end)
    real_label = "Real test EQU" if real_case == args.case else f"Real test EQU ({real_case})"
    plot_segment_profile(
        {
            f"JOS-3 predicted (t={int(t_end / 60)} min)": jos3_profile,
            f"{real_label} (t={int(t_end / 60)} min)": test_profile,
        },
        season=args.season,
        out_path=os.path.join(out_dir, f"segment_profile_{tag}.png"),
        title=f"Case {title_case} ({args.season}): body-segment equivalent temperature vs. ISO 14505-2 zones",
    )

    print(f"[{args.case}] summary plots written to -> {out_dir}")


if __name__ == "__main__":
    main()
