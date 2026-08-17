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

Input file format: sim-results/<case>/*<case>*.csv, auto-detected the same
way as compare_dummy_vs_sim.py (see detect_sim_format there, imported not
duplicated):
  - "legacy16": 16 monitor columns named like "Temperatur - Sensor_01_
    HEAD_top Monitor: ... (C)", one shared Time column. Uses SEGMENT_MAP
    (fragment matching, several JOS-3 segments proxied from the nearest
    available sensor -- see Assumptions below).
  - "25zone": 25 sensors matching the real test rig's own 1..25 zone
    numbering 1:1 (added 2026-08-12), each with its own "Physical Time
    (s)" column. Uses SEGMENT_MAP_25ZONE (real zone numbers, not sensor
    columns), which -- unlike the legacy map -- has genuine per-side
    forearm data (zones 12/13) instead of proxying LArm/RArm from the
    shoulder/upper-arm sensor. See SEGMENT_MAP_25ZONE's own comment for
    the full zone->segment reasoning (based on the test rig's own
    "Sensor position" zone diagram, shared 2026-08-17).
Not included in this repo (kept out of version control, see README below);
point --data-root at wherever that data lives locally.

Optional velocity input: sim-results/<case>/*velocity*<case>*.csv, auto-
detected the same way as the temperature input (detect_sim_format):
  - "legacy_sparse" (original format, min7/min0/pls7-era): long format
    with columns time_min, sensor_id, sensor_label, velocity_ms (16
    sensors x a handful of timestamps, doesn't need to be dense; values
    are linearly interpolated onto the temperature time grid and held
    constant beyond the last available timestamp). Matched to JOS-3
    segments by sensor_label (anatomical name), NOT sensor_id -- a real
    case (coarse_min7) turned up sensor_id/sensor_label pairs that
    disagree with that era's temperature export's own Sensor_NN
    numbering, so id-based matching would silently pair the wrong
    physical points. See VELOCITY_SEGMENT_MAP.
  - "25zone" (added 2026-08-17, e.g. the tilted/vent30deg_<scenario>
    cases): dense per-second export, SAME structure and SAME zone
    numbering as the 25zone temperature format (just "V - Sensor_NN_
    <label> Monitor" instead of "Temperatur - ..."), confirmed to use
    the identical zone<->label convention as the temperature file for
    the same case (no id/label mismatch this time) -- so it reuses
    SEGMENT_MAP_25ZONE directly, no separate velocity map needed.
When present, either format replaces the constant-Va assumption below
with real per-segment, per-timestep air velocity. When no velocity file
is found for a case, falls back to the constant AMBIENT_VA below.

Assumptions (documented here because none of these have a corresponding
sensor in the CFD export):
  - Relative humidity: constant 50% for all segments and all time steps.
  - Radiant temperature: assumed equal to air temperature (Tr = Ta).
  - Air velocity: constant 0.15 m/s (typical low cabin mixing velocity),
    UNLESS a velocity file is found for the case (see above), in which
    case real per-segment/per-timestep velocity is used instead.
  - Convective/evaporative heat transfer coefficients (JOS-3's internal
    _hc/_rt) are left at their physiological defaults, NOT overridden from
    CFD heat flux -- this CFD run only exports temperature, no heat flux.
  - Clothing insulation: Icl = 0 (nude) for all segments. The physical
    manikin this is validated against (see equ_comfort_from_test.py,
    plot_comfort_summary.py) wears no clothing, and the CFD run's own
    airflow field is solved around that bare geometry -- so a clothed
    JOS-3 body would be inconsistent with both the CFD boundary
    conditions and the real measurement it's compared against.
  - Three JOS-3 segments have no directly corresponding CFD sensor and
    reuse the nearest available region as a proxy: Back <- Thorax/chest,
    Pelvis <- average of both thighs, L/RArm (forearm) <- same-side
    shoulder/upper-arm sensor. The velocity mapping mirrors the same
    proxy choices, except Neck, where the velocity export happens to
    have a genuine neck sensor (head_neck) instead of reusing the
    upper-back proxy used for temperature.

Usage:
    python3 run_jos3_from_cfd.py --case min7
    python3 run_jos3_from_cfd.py --case min7 --data-root /path/to/master_thesis_input_data
    python3 run_jos3_from_cfd.py --case coarse_min7   # picks up velocity file automatically
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import jos3

from compare_dummy_vs_sim import detect_sim_format, load_sim_25zone

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

# JOS-3 segment <- real test rig zone number(s) 1..25 (see the "Sensor
# position" zone diagram Alperen shared 2026-08-17: 1 scalp, 2 face,
# 3 left temple, 4 right temple, 5 neck, 6 thorax middle, 7 thorax left,
# 8 thorax right, 9 stomach, 10 upper arm left, 11 upper arm right,
# 12 lower arm left, 13 lower arm right, 14 left hand, 15 right hand,
# 16 left thigh, 17 right thigh, 18 left thigh side, 19 right thigh side,
# 20 left shin, 21 right shin, 22 left shin side, 23 right shin side,
# 24 left foot, 25 right foot).
#
# Deliberately NOT the same grouping as the test rig's own 25->16
# "individual setting" table (which merges lower-arm+hand into one
# "hand" reading, e.g. zones 12+14 -> "Left hand") -- JOS-3 has separate
# LArm (forearm) and LHand segments, so this map keeps them separate and
# gets a real forearm reading for the first time (zones 12/13), instead
# of the legacy map's proxy (forearm <- shoulder/upper-arm sensor).
# Head averages all 4 head zones (scalp/face/temples) rather than just
# scalp, for a more representative single value now that the resolution
# is there. Chest/Back and Pelvis are still proxies -- the zone diagram
# has no back-facing or pelvis/abdomen sensor at all, same limitation as
# the legacy map, not something this format upgrade can fix.
SEGMENT_MAP_25ZONE: dict[str, list[int]] = {
    "Head":      [1, 2, 3, 4],
    "Neck":      [5],
    "Chest":     [6, 7, 8, 9],
    "Back":      [6, 7, 8, 9],   # proxy: no back-facing sensor in this layout
    "Pelvis":    [16, 17, 18, 19],  # proxy: avg of thighs
    "LShoulder": [10],
    "LArm":      [12],           # real forearm data, unlike SEGMENT_MAP's proxy
    "LHand":     [14],
    "RShoulder": [11],
    "RArm":      [13],           # real forearm data, unlike SEGMENT_MAP's proxy
    "RHand":     [15],
    "LThigh":    [16, 18],
    "LLeg":      [20, 22],
    "LFoot":     [24],
    "RThigh":    [17, 19],
    "RLeg":      [21, 23],
    "RFoot":     [25],
}

# JOS-3 segment <- velocity CSV sensor_label fragment(s). Matched by label,
# not by sensor_id -- see module docstring for why. Mirrors SEGMENT_MAP's
# proxy choices, except Neck (real head_neck sensor available here).
VELOCITY_SEGMENT_MAP: dict[str, str | list[str]] = {
    "Head":      "head_top",
    "Neck":      "head_neck",                       # real sensor, unlike the temp proxy
    "Chest":     "core_chest",
    "Back":      "core_chest",                       # proxy: no dedicated back sensor
    "Pelvis":    ["upperLeg_left", "upperLeg_right"],  # proxy: avg of thighs
    "LShoulder": "shoulder_left",
    "LArm":      "shoulder_left",                    # proxy: no dedicated forearm sensor
    "LHand":     "hand_left",
    "RShoulder": "shoulder_right",
    "RArm":      "shoulder_right",                   # proxy: no dedicated forearm sensor
    "RHand":     "hand_right",
    "LThigh":    "upperLeg_left",
    "LLeg":      "lowerLeg_left",
    "LFoot":     "foot_left",
    "RThigh":    "upperLeg_right",
    "RLeg":      "lowerLeg_right",
    "RFoot":     "foot_right",
}


def find_sim_file(data_root: str, case: str) -> str:
    case_dir = os.path.join(data_root, "sim-results", case)
    candidates = [
        p for p in glob.glob(os.path.join(case_dir, f"*{case}*.csv"))
        if "velocity" not in os.path.basename(p).lower()
    ]
    if not candidates:
        raise FileNotFoundError(f"No CFD sensor CSV found for case '{case}' under {case_dir}/")

    def version_key(path: str) -> tuple[int, str]:
        match = re.search(r"_v(\d+)", os.path.basename(path))
        return (int(match.group(1)) if match else -1, path)

    return max(candidates, key=version_key)


def find_velocity_file(data_root: str, case: str) -> str | None:
    """Optional -- returns None (not an error) when no velocity export
    exists for this case yet, so older cases keep using constant AMBIENT_VA."""
    case_dir = os.path.join(data_root, "sim-results", case)
    candidates = glob.glob(os.path.join(case_dir, f"*velocity*{case}*.csv"))
    if not candidates:
        return None

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
    """Returns Time + one air-temperature column per JOS-3 segment (legacy16 format)."""
    out = pd.DataFrame({"Time": cfd["Time"]})
    for segment, fragment in SEGMENT_MAP.items():
        fragments = [fragment] if isinstance(fragment, str) else fragment
        cols = [_resolve_column(cfd, f) for f in fragments]
        out[segment] = cfd[cols].mean(axis=1)
    return out


def build_segment_temps_25zone(sim25: pd.DataFrame) -> pd.DataFrame:
    """Returns Time + one air-temperature column per JOS-3 segment
    (25zone format, see SEGMENT_MAP_25ZONE). sim25 is load_sim_25zone's
    output: Time + columns named 'z<NN>_<label>'."""
    out = pd.DataFrame({"Time": sim25["Time"]})
    for segment, target_zones in SEGMENT_MAP_25ZONE.items():
        cols = [c for c in sim25.columns if any(c.startswith(f"z{z:02d}_") for z in target_zones)]
        if not cols:
            raise KeyError(f"No 25zone column found for JOS-3 segment '{segment}' (zones {target_zones})")
        out[segment] = sim25[cols].mean(axis=1)
    return out


def load_velocity(path: str) -> pd.DataFrame:
    """Long format: time_min, sensor_id, sensor_label, velocity_ms."""
    df = pd.read_csv(path)
    df["Time"] = df["time_min"] * 60.0  # seconds, to match the temperature CSV's Time column
    return df


def build_segment_velocities(vel: pd.DataFrame) -> pd.DataFrame:
    """Returns Time + one velocity column per JOS-3 segment (sparse -- one
    row per timestamp actually present in the velocity export; interpolation
    onto the simulation's own time grid happens later in run_jos3)."""
    times = sorted(vel["Time"].unique())
    out = pd.DataFrame({"Time": times})
    for segment, fragment in VELOCITY_SEGMENT_MAP.items():
        fragments = [fragment] if isinstance(fragment, str) else fragment
        rows = vel[vel["sensor_label"].isin(fragments)]
        if rows.empty:
            raise KeyError(f"Velocity CSV has no sensor_label matching '{fragments}'")
        out[segment] = rows.groupby("Time")["velocity_ms"].mean().reindex(times).to_numpy()
    return out


def run_jos3(segment_temps: pd.DataFrame, segment_velocities: pd.DataFrame | None = None) -> pd.DataFrame:
    model = jos3.JOS3(height=1.8, weight=75, age=30, ex_output="all")
    model.posture = "sitting"
    model.PAR = 1.0  # metabolic activity level [met], seated/resting

    target_t = np.arange(0, segment_temps["Time"].iloc[-1] + COUPLING_DT, COUPLING_DT)
    interp_temps = {
        seg: np.interp(target_t, segment_temps["Time"], segment_temps[seg])
        for seg in SECTIONS_JOS3
    }
    if segment_velocities is not None:
        # np.interp holds the first/last known value constant outside the
        # velocity export's own time range (e.g. beyond its last timestamp),
        # which is exactly the desired fallback -- see module docstring.
        interp_va = {
            seg: np.interp(target_t, segment_velocities["Time"], segment_velocities[seg])
            for seg in SECTIONS_JOS3
        }
        va0 = np.array([interp_va[seg][0] for seg in SECTIONS_JOS3])
    else:
        interp_va = None
        va0 = np.full(len(SECTIONS_JOS3), AMBIENT_VA)

    t0 = segment_temps.iloc[0][SECTIONS_JOS3].to_numpy(dtype=float)
    model.Icl = [0.0] * len(SECTIONS_JOS3)  # nude, matching the bare test manikin (see module docstring)
    model.Va = va0
    model.RH = AMBIENT_RH
    model.Ta = t0
    model.Tr = t0
    model.simulate(1, SOAK_MINUTES * 60)  # initial soak so the body isn't at an arbitrary start state

    for i in range(len(target_t)):
        ta = np.array([interp_temps[seg][i] for seg in SECTIONS_JOS3])
        model.Ta = ta
        model.Tr = ta  # assumption: no separate radiant sensor available, Tr = Ta
        model.Va = np.array([interp_va[seg][i] for seg in SECTIONS_JOS3]) if interp_va is not None else va0
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
    sim_format = detect_sim_format(sim_path)
    print(f"[{args.case}] CFD input: {sim_path}  (format: {sim_format})")

    if sim_format == "25zone":
        segment_temps = build_segment_temps_25zone(load_sim_25zone(sim_path))
    else:
        segment_temps = build_segment_temps(load_cfd(sim_path))

    vel_path = find_velocity_file(args.data_root, args.case)
    segment_velocities = None
    if vel_path:
        vel_format = detect_sim_format(vel_path)
        print(f"[{args.case}] Velocity input: {vel_path}  (format: {vel_format}, "
              f"overrides constant AMBIENT_VA={AMBIENT_VA})")
        if vel_format == "25zone":
            # Same shape as build_segment_temps_25zone (Time + one column
            # per JOS-3 segment) -- reused as-is, the function doesn't
            # care whether the source quantity is temperature or velocity.
            segment_velocities = build_segment_temps_25zone(load_sim_25zone(vel_path))
        else:
            segment_velocities = build_segment_velocities(load_velocity(vel_path))
    else:
        print(f"[{args.case}] No velocity file found, using constant AMBIENT_VA={AMBIENT_VA} m/s")

    result = run_jos3(segment_temps, segment_velocities)

    out_dir = os.path.join(args.data_root, "jos3_results", args.case)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"jos3_prediction_{args.case}.csv")
    result.to_csv(out_path, index=False)
    print(f"[{args.case}] JOS-3 predicted skin temperatures written to -> {out_path}")


if __name__ == "__main__":
    main()
