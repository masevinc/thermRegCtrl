"""
Combines the per-case Wall Y+ CSVs produced by
starccm_macros/extract_wall_yplus.java (run via run_export_wall_yplus.sh on
the HPC cluster, then downloaded locally) into one table, and -- if
compare_methods_vs_test.py has already been run for the same real-case --
joins it against that script's temperature-accuracy ranking, so you can see
directly whether the methods that matched the real test better also had
better (or worse!) near-wall mesh resolution.

Expects one CSV per case at <data-root>/yplus_exports/<case>_yplus.csv,
columns: boundary,area_avg_yplus,max_yplus (see extract_wall_yplus.java).

Usage:
    python3 summarize_wall_yplus.py
    python3 summarize_wall_yplus.py --real-case min7
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

DEFAULT_DATA_ROOT = os.path.expanduser("~/Documents/master_thesis_input_data")


def load_all(data_root: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_root, "yplus_exports", "*_yplus.csv")))
    if not paths:
        raise FileNotFoundError(
            f"No *_yplus.csv files found under {os.path.join(data_root, 'yplus_exports')}/ -- "
            "run run_export_wall_yplus.sh on the cluster and copy the yplus_exports/ folder here first."
        )

    rows = []
    for path in paths:
        case = os.path.basename(path).removesuffix("_yplus.csv")
        df = pd.read_csv(path)
        df.insert(0, "case", case)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real-case", default=None,
                         help="If given, joins against comparison_results/methods_vs_<real-case>/"
                              "method_ranking_overall.csv (see compare_methods_vs_test.py)")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()

    detail = load_all(args.data_root)
    detail_path = os.path.join(args.data_root, "yplus_exports", "wall_yplus_all_cases.csv")
    detail.to_csv(detail_path, index=False)

    per_case = detail.groupby("case").agg(
        mean_area_avg_yplus=("area_avg_yplus", "mean"),
        worst_area_avg_yplus=("area_avg_yplus", "max"),
        overall_max_yplus=("max_yplus", "max"),
        n_boundaries=("boundary", "count"),
    ).sort_values("overall_max_yplus")

    if args.real_case:
        ranking_path = os.path.join(
            args.data_root, "comparison_results", f"methods_vs_{args.real_case}", "method_ranking_overall.csv"
        )
        if os.path.exists(ranking_path):
            ranking = pd.read_csv(ranking_path, index_col="method")
            per_case = per_case.join(ranking, how="left")
        else:
            print(f"[!] {ranking_path} not found -- run compare_methods_vs_test.py --real-case {args.real_case} first "
                  "to get the temperature-accuracy columns alongside Y+.")

    summary_path = os.path.join(args.data_root, "yplus_exports", "wall_yplus_summary_by_case.csv")
    per_case.to_csv(summary_path)

    print(per_case.to_string())
    print(f"\nPer-boundary detail -> {detail_path}")
    print(f"Per-case summary    -> {summary_path}")


if __name__ == "__main__":
    main()
