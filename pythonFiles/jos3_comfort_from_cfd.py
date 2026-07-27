"""
Converts JOS-3's predicted skin/core temperatures (see run_jos3_from_cfd.py)
into local and overall thermal sensation/comfort scores using the UCB-Zhang
model (BerkeleyModel.py), the same model this repo's postComfort.py already
applies to STAR-CCM+ co-simulation runs.

This is the second half of the one-way CFD -> JOS-3 -> comfort pipeline:
    CFD air temperature -> JOS-3 (run_jos3_from_cfd.py) -> Tsk, Tcr, TskMean
    -> BerkeleyModel (this script) -> local/overall sensation & comfort

Run run_jos3_from_cfd.py --case <case> first.

Note: this script imports BerkeleyModel.py, which must sit in the same
directory (it does not import jos3.py, so it does not need to run from a
directory clear of the repo's jos3.py -- unlike run_jos3_from_cfd.py).

Usage:
    python3 jos3_comfort_from_cfd.py --case min7
    python3 jos3_comfort_from_cfd.py --case min7 --data-root /path/to/master_thesis_input_data
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import BerkeleyModel as ucb

DEFAULT_DATA_ROOT = os.path.expanduser("~/Documents/master_thesis_input_data")

# Berkeley/UCB section -> JOS-3 segment (same correspondence as postComfort.py's
# corrDict, which maps CFD/JOS-3-driven runs onto the 19 Berkeley sections;
# head/face/breathZone all reuse JOS-3's single Head segment, same for the
# left/right forearm <- shoulder proxy already used to drive JOS-3 itself).
BERKELEY_TO_JOS3 = {
    "head": "Head", "face": "Head", "breathZone": "Head",
    "neck": "Neck",
    "chest": "Chest",
    "back": "Back",
    "pelvis": "Pelvis",
    "lUArm": "LShoulder", "rUArm": "RShoulder",
    "lLArm": "LArm", "rLArm": "RArm",
    "lHand": "LHand", "rHand": "RHand",
    "lThigh": "LThigh", "rThigh": "RThigh",
    "lCalf": "LLeg", "rCalf": "RLeg",
    "lFoot": "LFoot", "rFoot": "RFoot",
}
BERKELEY_SECTIONS = list(BERKELEY_TO_JOS3.keys())


def find_jos3_file(data_root: str, case: str) -> str:
    path = os.path.join(data_root, "jos3_results", case, f"jos3_prediction_{case}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No JOS-3 prediction found at {path} -- run "
            f"'run_jos3_from_cfd.py --case {case}' first."
        )
    return path


def run_comfort_model(jos3_df: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    dt = jos3_df["Time"].diff().iloc[-1]  # coupling interval, seconds
    unique_jos3_segments = sorted(set(BERKELEY_TO_JOS3.values()))  # BERKELEY_TO_JOS3 reuses some
    dTsk = jos3_df[[f"Tsk_{seg}" for seg in unique_jos3_segments]].diff() / dt
    dTcr = jos3_df[[f"Tcr_{seg}" for seg in unique_jos3_segments]].diff() / dt

    time, overall_sensation, overall_comfort = [], [], []
    local_sensation_rows, local_comfort_rows = [], []

    # Skip row 0: no valid derivative (diff() is NaN there)
    for i in range(1, len(jos3_df)):
        temp_local = {sec: jos3_df[f"Tsk_{jseg}"].iloc[i] for sec, jseg in BERKELEY_TO_JOS3.items()}
        d_temp_local_dt = {sec: dTsk[f"Tsk_{jseg}"].iloc[i] for sec, jseg in BERKELEY_TO_JOS3.items()}
        d_temp_core_dt = {sec: dTcr[f"Tcr_{jseg}"].iloc[i] for sec, jseg in BERKELEY_TO_JOS3.items()}
        skin_mean = jos3_df["TskMean"].iloc[i]

        model = ucb.BerkeleyModel(temp_local, skin_mean, d_temp_local_dt, d_temp_core_dt)

        time.append(jos3_df["Time"].iloc[i])
        overall_sensation.append(model.OverallSensation())
        overall_comfort.append(model.OverallComfort())
        local_sensation_rows.append(model.LocalSensation())
        local_comfort_rows.append(model.LocalComfort())

    return {
        "time": pd.Series(time, name="Time"),
        "overall_sensation": pd.Series(overall_sensation, name="overallSensation"),
        "overall_comfort": pd.Series(overall_comfort, name="overallComfort"),
        "local_sensation": pd.DataFrame(local_sensation_rows),
        "local_comfort": pd.DataFrame(local_comfort_rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default="min7", help="Case name, e.g. min7, min0, pls7")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Folder containing jos3_results/<case>/")
    args = parser.parse_args()

    jos3_path = find_jos3_file(args.data_root, args.case)
    print(f"[{args.case}] JOS-3 input: {jos3_path}")
    jos3_df = pd.read_csv(jos3_path)

    result = run_comfort_model(jos3_df)

    out_dir = os.path.join(args.data_root, "jos3_results", args.case, "comfort")
    os.makedirs(out_dir, exist_ok=True)

    time = result["time"]
    pd.DataFrame({"time": time, "overallSensation": result["overall_sensation"]}).to_csv(
        os.path.join(out_dir, "overallSensation.csv"), index=False)
    pd.DataFrame({"time": time, "overallComfort": result["overall_comfort"]}).to_csv(
        os.path.join(out_dir, "overallComfort.csv"), index=False)

    local_sensation = result["local_sensation"].copy()
    local_sensation.insert(0, "time", time.to_numpy())
    local_sensation.to_csv(os.path.join(out_dir, "localSensationHistory.csv"), index=False)

    local_comfort = result["local_comfort"].copy()
    local_comfort.insert(0, "time", time.to_numpy())
    local_comfort.to_csv(os.path.join(out_dir, "localComfortHistory.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(time / 60, result["overall_sensation"], label="Overall sensation", color="tab:red", alpha=0.8)
    ax.plot(time / 60, result["overall_comfort"], label="Overall comfort", color="tab:blue", alpha=0.8)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Sensation / Comfort [-4 .. 4]")
    ax.set_title(f"Case {args.case}: JOS-3-driven overall sensation & comfort (UCB-Zhang model)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overallSensationComfort.png"), dpi=150)
    plt.close(fig)

    print(f"[{args.case}] comfort/sensation history and plot written to -> {out_dir}")


if __name__ == "__main__":
    main()
