"""
One-way coupling: drives the JOS-3 thermoregulation model with air
temperatures taken from a (separately run) STAR-CCM+ CFD solution, without
feeding anything back into the CFD run.

This is a validation study, not a live co-simulation like CoSim.py:
  CFD air temperature (per body region, over time)
        -> JOS-3 (this script)
        -> predicted skin temperature (per JOS-3 segment, over time)

The predicted skin temperature can then be compared against the same CFD
run's own sensor data and against physical thermal-manikin measurements
(see compare_dummy_vs_sim.py in the master's thesis data folder), to see
how JOS-3's physiological model responds to that scenario.

Input file format: sim-results/<case>/*<case>*.csv, with 16 monitor columns
named like "Temperatur - Sensor_01_HEAD_top Monitor: ... (C)" -- i.e. the
same CFD sensor export used by compare_dummy_vs_sim.py. Not included in this
repo (kept out of version control, see README below); point --data-root at
wherever that data lives locally.

Assumptions (documented here because none of these have a corresponding
sensor in the CFD export):
  - Relative humidity: constant 50% for all segments and all time steps.
  - Radiant temperature: assumed equal to air temperature (Tr = Ta).
  - Air velocity: constant 0.15 m/s (typical low cabin mixing velocity).
  - Convective/evaporative heat transfer coefficients (JOS-3's internal
    _hc/_rt) are left at their physiological defaults, NOT overridden from
    CFD heat flux -- this CFD run only exports temperature, no heat flux.
  - Clothing insulation per segment follows the same ambient-temperature
    based profile as CoSim.py's clothingReq().
  - Three JOS-3 segments have no directly corresponding CFD sensor and
    reuse the nearest available region as a proxy: Back <- Thorax/chest,
    Pelvis <- average of both thighs, L/RArm (forearm) <- same-side
    shoulder/upper-arm sensor.

Usage:
    python3 run_jos3_from_cfd.py --case min7
    python3 run_jos3_from_cfd.py --case min7 --data-root /path/to/master_thesis_input_data
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import jos3

DEFAULT_DATA_ROOT = os.path.expanduser("~/Documents/master_thesis_input_data")

AMBIENT_RH = 50.0   # %, constant assumption -- no RH sensor in this CFD export
AMBIENT_VA = 0.15   # m/s, constant assumption -- no velocity sensor in this CFD export
COUPLING_DT = 1.0    # s, JOS-3 update interval (matches ThermRegCtrl's default updateTime)
SOAK_MINUTES = 5     # initial soak time before the transient starts

# JOS-3 segment (BODY_NAMES order) <- CFD sensor column fragment(s).
# Segments without a direct sensor reuse the nearest available region (see
# module docstring). Column fragments match compare_dummy_vs_sim.py's
# REGION_MAP so the same 16 raw CFD sensors are used consistently.
SEGMENT_MAP: dict[str, str | list[str]] = {
    "Head":      "Sensor_01_HEAD_top",
    "Neck":      "Sensor_05_CORE_upperBack",
    "Chest":     "Sensor_06_CORE_chest",
    "Back":      "Sensor_06_CORE_chest",       # proxy: no dedicated back sensor
    "Pelvis":    ["Sensor_11_LOWERBODY_leftUpperLeg", "Sensor_12_LOWERBODY_rightUpperLeg"],  # proxy: avg of thighs
    "LShoulder": "Sensor_07_SHOULDER_left",
    "LArm":      "Sensor_07_SHOULDER_left",    # proxy: no dedicated forearm sensor
    "LHand":     "Sensor_09_HAND_left",
    "RShoulder": "Sensor_08_SHOULDER_right",
    "RArm":      "Sensor_08_SHOULDER_right",   # proxy: no dedicated forearm sensor
    "RHand":     "Sensor_10_HAND_right",
    "LThigh":    "Sensor_11_LOWERBODY_leftUpperLeg",
    "LLeg":      "Sensor_13_LOWERBODY_leftLowerLeg",
    "LFoot":     "Sensor_15_LOWERBODY_leftFoot",
    "RThigh":    "Sensor_12_LOWERBODY_rightUpperLeg",
    "RLeg":      "Sensor_14_LOWERBODY_rightLowerLeg",
    "RFoot":     "Sensor_16_LOWERBODY_rightFoot",
}
SECTIONS_JOS3 = list(SEGMENT_MAP.keys())  # must match jos3.matrix.BODY_NAMES order


def clothing_req(ambient_temp_c: float) -> list[float]:
    """Same cold/warm-weather clothing profile as CoSim.py's clothingReq(),
    ordered to match SECTIONS_JOS3 / BODY_NAMES directly."""
    if ambient_temp_c <= -20:
        head, shirt, pant, shoe, hand = 0.0, 1.5, 1.5, 0.2, 0.2
    elif -20 < ambient_temp_c <= -10:
        head, shirt, pant, shoe, hand = 0.0, 1.3, 1.3, 0.15, 0.15
    elif -10 < ambient_temp_c <= 0:
        head, shirt, pant, shoe, hand = 0.0, 0.9, 0.9, 0.1, 0.1
    elif ambient_temp_c > 30:
        head, shirt, pant, shoe, hand = 0.0, 0.25, 0.25, 0.05, 0.0
    else:
        head, shirt, pant, shoe, hand = 0.0, 0.47, 0.4, 0.08, 0.0
    return [
        head, shirt, shirt, shirt, shirt * 0.2 + pant, shirt, shirt, hand,
        shirt, shirt, hand, pant, pant, shoe, pant, pant, shoe,
    ]


def find_sim_file(data_root: str, case: str) -> str:
    candidates = glob.glob(os.path.join(data_root, "sim-results", case, f"*{case}*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CFD sensor CSV found for case '{case}' under "
            f"{os.path.join(data_root, 'sim-results', case)}/"
        )

    def version_key(path: str) -> tuple[int, str]:
        match = re.search(r"_v(\d+)", os.path.basename(path))
        return (int(match.group(1)) if match else -1, path)

    return max(candidates, key=version_key)


def load_cfd(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.split(":")[0].replace("Temperatur - ", "").strip() for c in df.columns]
    return df


def _resolve_column(cfd: pd.DataFrame, fragment: str) -> str:
    match = next((c for c in cfd.columns if fragment in c), None)
    if match is None:
        raise KeyError(f"CFD column not found for fragment '{fragment}'")
    return match


def build_segment_temps(cfd: pd.DataFrame) -> pd.DataFrame:
    """Returns Time + one air-temperature column per JOS-3 segment."""
    out = pd.DataFrame({"Time": cfd["Time"]})
    for segment, fragment in SEGMENT_MAP.items():
        fragments = [fragment] if isinstance(fragment, str) else fragment
        cols = [_resolve_column(cfd, f) for f in fragments]
        out[segment] = cfd[cols].mean(axis=1)
    return out


def run_jos3(segment_temps: pd.DataFrame) -> pd.DataFrame:
    model = jos3.JOS3(height=1.8, weight=75, age=30, ex_output="all")
    model.posture = "sitting"
    model.PAR = 1.0  # metabolic activity level [met], seated/resting

    t0 = segment_temps.iloc[0][SECTIONS_JOS3].to_numpy(dtype=float)
    model.Icl = clothing_req(float(np.mean(t0)))
    model.Va = AMBIENT_VA
    model.RH = AMBIENT_RH
    model.Ta = t0
    model.Tr = t0
    model.simulate(1, SOAK_MINUTES * 60)  # initial soak so the body isn't at an arbitrary start state

    target_t = np.arange(0, segment_temps["Time"].iloc[-1] + COUPLING_DT, COUPLING_DT)
    interp_temps = {
        seg: np.interp(target_t, segment_temps["Time"], segment_temps[seg])
        for seg in SECTIONS_JOS3
    }

    for i in range(len(target_t)):
        ta = np.array([interp_temps[seg][i] for seg in SECTIONS_JOS3])
        model.Ta = ta
        model.Tr = ta  # assumption: no separate radiant sensor available, Tr = Ta
        model.simulate(1, COUPLING_DT)

    history = model.dict_results()
    n_soak_rows = 2  # dict_results() includes the initial construction-time row and the soak-phase row first
    result = pd.DataFrame({"Time": target_t})
    result["TskMean"] = history["TskMean"][n_soak_rows:]
    for seg in SECTIONS_JOS3:
        result[f"Tsk_{seg}"] = history[f"Tsk{seg}"][n_soak_rows:]
        result[f"Tcr_{seg}"] = history[f"Tcr{seg}"][n_soak_rows:]
        result[f"To_{seg}"] = history[f"To{seg}"][n_soak_rows:]  # operative temperature (environmental, not physiological)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default="min7", help="Case name, e.g. min7, min0, pls7")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Folder containing sim-results/<case>/*.csv")
    args = parser.parse_args()

    sim_path = find_sim_file(args.data_root, args.case)
    print(f"[{args.case}] CFD input: {sim_path}")

    cfd = load_cfd(sim_path)
    segment_temps = build_segment_temps(cfd)

    result = run_jos3(segment_temps)

    out_dir = os.path.join(args.data_root, "jos3_results", args.case)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"jos3_prediction_{args.case}.csv")
    result.to_csv(out_path, index=False)
    print(f"[{args.case}] JOS-3 predicted skin temperatures written to -> {out_path}")


if __name__ == "__main__":
    main()
