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
    REGION_MAP,
    detect_sim_format,
    find_dummy_file,
    find_sim_file,
    load_dummy,
    load_sim_25zone,
    load_sim_legacy,
    region_series,
    resample_sim,
    sim_25zone_to_legacy_regions,
)

# Every case is plotted on REGION_MAP's 16 regions. Legacy-format cases
# (see compare_dummy_vs_sim.py) already have exactly that shape. 25-zone
# cases get collapsed down to the same 16 regions via
# sim_25zone_to_legacy_regions -- re-introduces the averaging the 25-zone
# format was added to avoid, but is the only way to put a 25-zone case on
# the same grid as the older ones for a direct overlay/ranking. Use
# compare_dummy_vs_sim.py directly instead if you want the full-resolution
# 25-zone-native comparison for a single case.

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
    args = parser.parse_args()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]

    dummy_path = find_dummy_file(args.data_root, args.real_case)
    if dummy_path is None:
        raise FileNotFoundError(f"No dummy measurement file found for real-case '{args.real_case}'")
    dummy = load_dummy(dummy_path, args.value_kind)
    target_t = dummy["elapsed_s"].to_numpy()

    sims: dict[str, pd.DataFrame] = {}
    for case in cases:
        sim_path = find_sim_file(args.data_root, case)
        if sim_path is None:
            print(f"[!] no sim result found for case '{case}', skipping it")
            continue
        fmt = detect_sim_format(sim_path)
        sims[case] = (
            sim_25zone_to_legacy_regions(load_sim_25zone(sim_path))
            if fmt == "25zone"
            else load_sim_legacy(sim_path)
        )
        print(f"[{case}] sim: {sim_path} (format: {fmt})")
    if not sims:
        raise FileNotFoundError("No sim result files found for any of the requested cases")

    out_name = f"methods_vs_{args.real_case}" + (f"_{args.tag}" if args.tag else "")
    out_dir = os.path.join(args.data_root, "comparison_results", out_name)
    os.makedirs(out_dir, exist_ok=True)

    n = len(REGION_MAP)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig_all, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), sharex=True)
    axes = axes.flatten()

    metric_rows = []
    for i, (label, zones, sim_col_key) in enumerate(REGION_MAP):
        test_series = region_series(dummy, zones, args.value_kind)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(target_t, test_series, label=f"Real test ({args.value_kind})", color="black", linewidth=2.2, zorder=10)
        ax_grid = axes[i]
        ax_grid.plot(target_t, test_series, color="black", linewidth=1.6, label="Real test", zorder=10)

        for j, (case, sim) in enumerate(sims.items()):
            sim_col = next((c for c in sim.columns if sim_col_key in c), None)
            if sim_col is None:
                continue
            sim_series = resample_sim(sim, sim_col, target_t)
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

            ax.plot(target_t, sim_series, label=case, color=color, alpha=0.85, linewidth=1.2)
            ax_grid.plot(target_t, sim_series, color=color, alpha=0.85, linewidth=0.9)

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

    for j in range(len(REGION_MAP), len(axes)):
        axes[j].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig_all.legend(handles, labels, loc="upper right", fontsize=8)
    fig_all.suptitle(
        f"Real test ({args.real_case}, {args.value_kind}) vs. {len(sims)} CFD methods, all regions\n"
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
    print("Overall method ranking (best match to real test first, by mean RMSE across all 16 regions):")
    print(ranking.to_string())
    print(f"\nPlots and metrics written to -> {out_dir}")


if __name__ == "__main__":
    main()
