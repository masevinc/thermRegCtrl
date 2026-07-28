"""
Post-processing visualizations in the style common in vehicle-cabin
thermal-comfort studies:

  - PPD (Predicted Percentage Dissatisfied) vs. DTS (Dynamic Thermal
    Sensation) curve: Fanger's classic PMV/PPD functional form, applied to
    the Zhang/Berkeley model's "DTS" output (this repo's
    jos3_comfort_from_cfd.py already produces it as overallSensation).

  - Body-segment equivalent-temperature profile: ISO 14505-2 comfort-zone
    boundaries (see iso14505_bands.py) drawn as background reference
    curves across all 17 body segments, with one or more measured/
    predicted per-segment temperature profiles overlaid for comparison.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from iso14505_bands import BANDS_BY_SEASON

# JOS-3 segment, bottom-to-top display order (feet at the bottom, head /
# abdomen at the top -- the conventional manikin-diagram layout).
SEGMENT_ORDER = [
    "RFoot", "LFoot", "RLeg", "LLeg", "RThigh", "LThigh",
    "RHand", "LHand", "RArm", "LArm", "RShoulder", "LShoulder",
    "Back", "Chest", "Neck", "Head", "Pelvis",
]
SEGMENT_DISPLAY_NAMES = {
    "RFoot": "R Foot", "LFoot": "L Foot",
    "RLeg": "R Calf", "LLeg": "L Calf",
    "RThigh": "R Thigh", "LThigh": "L Thigh",
    "RHand": "R Hand", "LHand": "L Hand",
    "RArm": "R Forearm", "LArm": "L Forearm",
    "RShoulder": "R Arm", "LShoulder": "L Arm",
    "Back": "Thorax Back", "Chest": "Thorax Front",
    "Neck": "Face - Neck", "Head": "Head", "Pelvis": "Abdomen",
}
# JOS-3 segment -> ISO 14505-2 body-region category (iso14505_bands.py).
# Neck/Pelvis have no dedicated ISO row; face/wholeBody are the closest
# available proxies (same limitation noted in iso14505_bands.py).
SEGMENT_TO_ISO_REGION = {
    "RFoot": "foot", "LFoot": "foot",
    "RLeg": "calf", "LLeg": "calf",
    "RThigh": "thigh", "LThigh": "thigh",
    "RHand": "hand", "LHand": "hand",
    "RArm": "lowerArm", "LArm": "lowerArm",
    "RShoulder": "upperArm", "LShoulder": "upperArm",
    "Back": "upperBack", "Chest": "chest",
    "Neck": "face", "Head": "scalp", "Pelvis": "wholeBody",
}

ZONE_COLORS = {1: "#1f3fbf", 2: "#4aa3e0", 3: "#2fa63a", 4: "#e08a2f", 5: "#c22a2a"}
ZONE_NAMES = {1: "Too Cold", 2: "Cool", 3: "Neutral", 4: "Warm", 5: "Too Hot"}


def fanger_ppd(sensation):
    """Fanger's PPD formula, applied to a sensation scale such as the
    Zhang/Berkeley model's DTS (overallSensation)."""
    s = np.asarray(sensation, dtype=float)
    return 100 - 95 * np.exp(-0.03353 * s**4 - 0.2179 * s**2)


def plot_ppd_curve(out_path: str, highlight: dict[str, float] | None = None,
                    sensation_range: tuple[float, float] = (-3, 3), title: str = "") -> None:
    """Draws the PPD-vs-DTS curve; `highlight` marks specific scenarios
    (label -> DTS value) as points on the curve."""
    s = np.linspace(sensation_range[0], sensation_range[1], 400)
    ppd = fanger_ppd(s)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    grad = np.vstack([np.linspace(0, 1, 256)] * 2)
    ax.imshow(grad, extent=[s.min(), s.max(), 0, 100], aspect="auto",
              cmap="coolwarm", origin="lower", alpha=0.85, zorder=0)
    ax.fill_between(s, ppd, 100, color="white", zorder=1)
    ax.plot(s, ppd, color="black", linewidth=1.5, zorder=2)

    markers = ["x", "s", "^", "D", "o", "P"]
    if highlight:
        for i, (label, value) in enumerate(highlight.items()):
            ax.scatter([value], [fanger_ppd(value)], marker=markers[i % len(markers)],
                       s=120, edgecolor="black", linewidth=1.2, label=label, zorder=3,
                       color="yellow")

    ax.set_xlim(*sensation_range)
    ax.set_ylim(0, 100)
    ax.set_xlabel("DTS - Dynamic Thermal Sensation")
    ax.set_ylabel("PPD - Predicted Percentage Dissatisfied")
    ticks = np.arange(sensation_range[0], sensation_range[1] + 0.5, 1)
    sensation_labels = {-3: "Too Cold", -2: "Cold", -1: "Slightly\nCold", 0: "Neutral",
                         1: "Slightly\nWarm", 2: "Warm", 3: "Too Hot"}
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}\n{sensation_labels.get(t, '')}" for t in ticks])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{int(v)}%"))
    if title:
        ax.set_title(title, fontsize=11, wrap=True)
    if highlight:
        # place legend on whichever side (left/right of neutral) has fewer highlighted points, to avoid covering data
        mean_sensation = np.mean(list(highlight.values()))
        ax.legend(loc="upper right" if mean_sensation < 0 else "upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_segment_profile(profiles: dict[str, dict[str, float]], season: str, out_path: str,
                          title: str = "") -> None:
    """profiles: {scenario_label: {jos3_segment: temperature}}. Draws the
    ISO 14505-2 comfort-zone boundaries as background reference curves
    across SEGMENT_ORDER, with each scenario's per-segment profile
    overlaid on top."""
    bands = BANDS_BY_SEASON[season]
    y = np.arange(1, len(SEGMENT_ORDER) + 1)

    fig, ax = plt.subplots(figsize=(9, 7))

    for edge_idx in range(4):
        xs = [bands[SEGMENT_TO_ISO_REGION[seg]][edge_idx] for seg in SEGMENT_ORDER]
        ax.plot(xs, y, color=ZONE_COLORS[edge_idx + 1], linewidth=1.8, label=ZONE_NAMES[edge_idx + 1])
    ax.plot([], [], color=ZONE_COLORS[5], linewidth=1.8, label=ZONE_NAMES[5])  # legend entry only

    markers = ["x", "s", "^", "D", "o"]
    for i, (label, seg_values) in enumerate(profiles.items()):
        xs = [seg_values.get(seg, np.nan) for seg in SEGMENT_ORDER]
        ax.plot(xs, y, color="dimgray", linewidth=0.8, alpha=0.7, zorder=1)
        ax.scatter(xs, y, marker=markers[i % len(markers)], s=45, color="black",
                   label=label, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels([str(i) for i in y])
    ax.set_ylim(0.5, len(SEGMENT_ORDER) + 0.5)
    ax2 = ax.secondary_yaxis("right")
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{i}  {SEGMENT_DISPLAY_NAMES[seg]}" for i, seg in zip(y, SEGMENT_ORDER)])
    ax.set_ylabel("Manikin Body Segment ID")
    ax.set_xlabel("Equivalent Homogeneous Temperature (C)")
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper center", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
