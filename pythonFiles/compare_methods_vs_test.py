"""
Mesh/solver-method sensitivity study: overlays several CFD variants of the
same ambient scenario (e.g. coarse vs. fine mesh, k-omega SST vs. base
turbulence model, cleaned vs. raw geometry, different foot-instep meshing)
against ONE real thermal-manikin test run, per body region -- so you can
see at a glance which method setting(s) actually track physical reality,
rather than comparing methods only against each other.

This does not involve JOS-3 at all -- pure CFD air temperature (sim-results)
vs. real test air temperature (dummy_measurements), same REGION_MAP and
loading logic as compare_dummy_vs_sim.py (imported from there, not
duplicated).

Usage:
    python3 compare_methods_vs_test.py --real-case min7
    python3 compare_methods_vs_test.py --real-case min7 --cases coarse_min7,fine_min7,kwsst_min7
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_dummy_vs_sim import (
    DEFAULT_DATA_ROOT,
    DEFAULT_SMOOTH_WINDOW_S,
    REGION_MAP,
    detect_sim_format,
    find_dummy_file,
    find_sim_file,
    infer_dummy_subdir,
    load_dummy,
    load_sim_25zone,
    load_sim_legacy,
    region_list_25zone,
    region_series,
    resample_sim,
    sim_25zone_to_legacy_regions,
)

# Grid resolution is decided per run, not fixed: if every --cases entry is
# in the newer 25-zone format (see compare_dummy_vs_sim.py), the overlay
# is plotted natively on all 25 real-test zones (region_list_25zone) --
# no reason to throw away resolution when nothing needs collapsing. As
# soon as ANY case in the list is the older legacy 16-sensor format, the
# whole overlay falls back to REGION_MAP's 16 grouped regions (the only
# resolution every case can be expressed in), including collapsing any
# 25-zone cases in the mix via sim_25zone_to_legacy_regions.

# Default sensitivity-study cases for the -7C ambient scenario (min7), one
# per CFD method/mesh variant -- see thermRegCtrl memory notes for what each
# one changed (mesh resolution, turbulence model, geometry cleanup, foot
# instep meshing). Override with --cases for a different scenario/case set.
DEFAULT_CASES = [
    "coarse_min7",
    "fine_min7",
    "cleaned_full_min7",
    "half_cleaned_min7",
    "high_instep_min7",
    "low_instep_min7",
    "kwsst_min7",
    "first_order_min7",
    "second_order_min7",
    "tilted_30deg_min7",
    "tilted_30deg_v4_min7",
]

METHOD_COLORS = [
    "tab:red", "tab:green", "tab:purple", "tab:orange",
    "tab:brown", "tab:pink", "tab:olive", "tab:cyan",
    "tab:blue", "gold", "tab:gray", "indigo",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES),
                         help="Comma-separated CFD case names to overlay (each must have its own "
                              "sim-results/<case>/ folder)")
    parser.add_argument("--real-case", default="min7", help="Real dummy_measurements case to compare all methods against")
    parser.add_argument("--tag", default=None,
                         help="Output subfolder suffix: comparison_results/methods_vs_<real-case>[_<tag>]/. "
                              "Use this whenever --cases is a custom subset, so it doesn't overwrite the "
                              "full-sensitivity-study run's output for the same --real-case.")
    parser.add_argument("--value-kind", default="AIR", choices=["AIR", "EQU"])
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--smooth-window", type=float, default=DEFAULT_SMOOTH_WINDOW_S,
                         help="Savitzky-Golay smoothing window for PLOTTED sim curves only, in seconds "
                              f"(default {DEFAULT_SMOOTH_WINDOW_S}s). Error metrics/ranking always use raw data.")
    parser.add_argument("--no-smooth", action="store_true", help="Plot raw (unsmoothed) sim curves.")
    args = parser.parse_args()
    smooth_window_s = None if args.no_smooth else args.smooth_window

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]

    # NOTE: infers ONE dummy_measurements/ subdir (tilted vs default, see
    # infer_dummy_subdir) from the FIRST --cases entry, then uses it for
    # every case in the overlay. Fine when all cases share one vent
    # config (e.g. the full sensitivity study, all historically
    # tilted-vent). Questionable when --cases deliberately mixes
    # vent0deg_* and vent30deg_* (the "ventangle" comparisons) -- those
    # two are now validated against DIFFERENT real tests (default/ vs
    # tilted/ have separate no-tilt/tilted measurements), so overlaying
    # both against a single real-test curve is no longer strictly
    # apples-to-apples now that per-vent real data exists. Flagged, not
    # silently fixed -- ask before splitting these into two single-vent
    # comparisons.
    dummy_path = find_dummy_file(args.data_root, args.real_case, subdir=infer_dummy_subdir(cases[0]))
    if dummy_path is None:
        raise FileNotFoundError(f"No dummy measurement file found for real-case '{args.real_case}'")
    dummy = load_dummy(dummy_path, args.value_kind)
    target_t = dummy["elapsed_s"].to_numpy()

    sims_raw: dict[str, pd.DataFrame] = {}
    formats: dict[str, str] = {}
    for case in cases:
        sim_path = find_sim_file(args.data_root, case)
        if sim_path is None:
            print(f"[!] no sim result found for case '{case}', skipping it")
            continue
        fmt = detect_sim_format(sim_path)
        sims_raw[case] = load_sim_25zone(sim_path) if fmt == "25zone" else load_sim_legacy(sim_path)
        formats[case] = fmt
        print(f"[{case}] sim: {sim_path} (format: {fmt})")
    if not sims_raw:
        raise FileNotFoundError("No sim result files found for any of the requested cases")

    all_25zone = all(fmt == "25zone" for fmt in formats.values())
    if all_25zone:
        sims = sims_raw
        region_list = region_list_25zone(next(iter(sims.values())))
        print(f"All {len(sims)} cases are 25-zone format -- plotting natively on all {len(region_list)} real-test zones.")
    else:
        sims = {
            case: (sim_25zone_to_legacy_regions(sim) if formats[case] == "25zone" else sim)
            for case, sim in sims_raw.items()
        }
        region_list = [(label, zones, sim_col_key) for label, zones, sim_col_key in REGION_MAP]
        if any(fmt == "25zone" for fmt in formats.values()):
            print("Mixed legacy/25-zone cases -- collapsing 25-zone case(s) down to REGION_MAP's 16 regions "
                  "so every case shares one grid.")

    out_name = f"methods_vs_{args.real_case}" + (f"_{args.tag}" if args.tag else "")
    out_dir = os.path.join(args.data_root, "comparison_results", out_name)
    os.makedirs(out_dir, exist_ok=True)

    n = len(region_list)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig_all, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), sharex=True)
    axes = axes.flatten()

    metric_rows = []
    for i, (label, zones, sim_col_key) in enumerate(region_list):
        test_series = region_series(dummy, zones, args.value_kind)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(target_t, test_series, label=f"Real test ({args.value_kind})", color="black", linewidth=2.2, zorder=10)
        ax_grid = axes[i]
        ax_grid.plot(target_t, test_series, color="black", linewidth=1.6, label="Real test", zorder=10)

        for j, (case, sim) in enumerate(sims.items()):
            sim_col = next((c for c in sim.columns if sim_col_key in c), None)
            if sim_col is None:
                continue
            sim_series = resample_sim(sim, sim_col, target_t)  # RAW -- metrics always use this
            color = METHOD_COLORS[j % len(METHOD_COLORS)]

            err = sim_series - test_series.to_numpy()
            valid = ~np.isnan(err)
            metric_rows.append({
                "region": label,
                "method": case,
                "bias_sim_minus_test_degC": np.nanmean(err) if valid.any() else np.nan,
                "mae_degC": np.nanmean(np.abs(err)) if valid.any() else np.nan,
                "rmse_degC": np.sqrt(np.nanmean(err**2)) if valid.any() else np.nan,
                "max_abs_error_degC": np.nanmax(np.abs(err)) if valid.any() else np.nan,
            })

            sim_plot = sim_series if smooth_window_s is None else resample_sim(sim, sim_col, target_t, smooth_window_s)
            ax.plot(target_t, sim_plot, label=case, color=color, alpha=0.85, linewidth=1.2)
            ax_grid.plot(target_t, sim_plot, color=color, alpha=0.85, linewidth=0.9)

        ax.set_title(f"{label}: real test vs. {len(sims)} CFD methods")
        ax.set_xlabel("Elapsed time since test start [s]")
        ax.set_ylabel("Temperature [degC]")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{i+1:02d}_{label.replace(' ', '_')}.png"), dpi=150)
        plt.close(fig)

        ax_grid.set_title(label, fontsize=9)
        ax_grid.set_xlabel("Time [s]", fontsize=8)
        ax_grid.set_ylabel("Temp [degC]", fontsize=8)
        ax_grid.grid(alpha=0.3)

    for j in range(len(region_list), len(axes)):
        axes[j].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig_all.legend(handles, labels, loc="upper right", fontsize=8)
    smooth_note = f" (sim smoothed, Savitzky-Golay {int(smooth_window_s)}s -- metrics use raw data)" if smooth_window_s else ""
    fig_all.suptitle(
        f"Real test ({args.real_case}, {args.value_kind}) vs. {len(sims)} CFD methods, all regions{smooth_note}\n"
        "X axis: elapsed time since test start [s]  |  Y axis: temperature [degC]"
    )
    fig_all.tight_layout(rect=(0, 0, 1, 0.94))
    fig_all.savefig(os.path.join(out_dir, "00_overview_all_regions.png"), dpi=150)
    plt.close(fig_all)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(os.path.join(out_dir, "error_metrics_by_region_and_method.csv"), index=False)

    # Aggregate ranking: mean |bias| and mean RMSE across all 16 regions, per method
    ranking = metrics.groupby("method").agg(
        mean_abs_bias_degC=("bias_sim_minus_test_degC", lambda s: np.mean(np.abs(s))),
        mean_rmse_degC=("rmse_degC", "mean"),
        mean_mae_degC=("mae_degC", "mean"),
    ).sort_values("mean_rmse_degC")
    ranking.to_csv(os.path.join(out_dir, "method_ranking_overall.csv"))

    print()
    print(f"Overall method ranking (best match to real test first, by mean RMSE across all {len(region_list)} regions):")
    print(ranking.to_string())
    print(f"\nPlots and metrics written to -> {out_dir}")


if __name__ == "__main__":
    main()
