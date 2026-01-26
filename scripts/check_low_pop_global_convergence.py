"""
Check whether low-pop (precinct, demo) cells are closer to global avg or uniform 1/n.
Compares mean absolute distance to global average vs uniform across output CSVs.
"""
import argparse
import os

import numpy as np
import pandas as pd

DEMOGRAPHICS = ["Wht", "His", "Blk", "Asn", "Oth"]
VOTE_TYPES = ["D", "R", "O", "N"]
EPSILON = 1e-10


def resolve_index(df):
    if "AFFGEOID" in df.columns:
        df = df.set_index("AFFGEOID")
    elif "Unnamed: 0" in df.columns:
        df = df.set_index("Unnamed: 0")
    df.index = df.index.astype(str)
    return df


def find_prob_columns(df):
    cols = {}
    for vote in VOTE_TYPES:
        for demo in DEMOGRAPHICS:
            candidates = [
                f"{vote}_{demo}_prob",
                f"{vote}_{demo}",
                f"{vote}_{demo}_p",
            ]
            match = next((c for c in candidates if c in df.columns), None)
            if match is None:
                return None
            cols[(vote, demo)] = match
    return cols


def build_prob_tensor(df, prob_cols):
    num_precincts = len(df)
    num_demos = len(DEMOGRAPHICS)
    num_votes = len(VOTE_TYPES)
    p = np.zeros((num_precincts, num_demos, num_votes), dtype=float)
    for demo_idx, demo in enumerate(DEMOGRAPHICS):
        for vote_idx, vote in enumerate(VOTE_TYPES):
            p[:, demo_idx, vote_idx] = df[prob_cols[(vote, demo)]].values.astype(float)
    return p


def compute_low_pop_report(df, prob_cols, low_pop_threshold):
    demo_cols = [f"cvap_{demo}" for demo in DEMOGRAPHICS]
    if not all(col in df.columns for col in demo_cols):
        return None, "missing cvap columns"

    D = np.array([df[col].values.astype(float) for col in demo_cols]).T
    p = build_prob_tensor(df, prob_cols)

    low_pop_mask = D < low_pop_threshold
    if not np.any(low_pop_mask):
        return None, "no low-pop cells"

    uniform = np.full((len(VOTE_TYPES),), 1.0 / len(VOTE_TYPES))

    global_avg = np.zeros((len(DEMOGRAPHICS), len(VOTE_TYPES)), dtype=float)
    for demo_idx in range(len(DEMOGRAPHICS)):
        demo_pop = D[:, demo_idx]
        total_pop = demo_pop.sum()
        if total_pop > 0:
            global_avg[demo_idx, :] = np.average(p[:, demo_idx, :], axis=0, weights=demo_pop)
        else:
            global_avg[demo_idx, :] = uniform

    low_pop_indices = np.argwhere(low_pop_mask)
    dist_global = []
    dist_uniform = []
    for precinct_idx, demo_idx in low_pop_indices:
        p_vec = p[precinct_idx, demo_idx, :]
        dist_global.append(np.mean(np.abs(p_vec - global_avg[demo_idx, :])))
        dist_uniform.append(np.mean(np.abs(p_vec - uniform)))

    dist_global = np.array(dist_global)
    dist_uniform = np.array(dist_uniform)
    closer_global = np.mean(dist_global < dist_uniform) if len(dist_global) else 0.0

    return {
        "low_pop_cells": int(len(low_pop_indices)),
        "mean_abs_diff_global": float(np.mean(dist_global)),
        "mean_abs_diff_uniform": float(np.mean(dist_uniform)),
        "pct_closer_global": float(closer_global),
    }, None


def iter_csvs(output_dir, files):
    if files:
        return files
    if not output_dir:
        return []
    return [
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.lower().endswith(".csv")
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Check if low-pop cells converge toward global avg or uniform."
    )
    parser.add_argument("--output-dir", default="output", help="Directory with output CSVs")
    parser.add_argument("--files", nargs="*", help="Specific CSV files to check")
    parser.add_argument("--low-pop-threshold", type=float, default=25.0)
    parser.add_argument("--summary-out", default="output/low_pop_global_check.csv")
    args = parser.parse_args()

    results = []
    skipped = []
    for csv_path in sorted(iter_csvs(args.output_dir, args.files)):
        try:
            df = pd.read_csv(csv_path, dtype={"AFFGEOID": str})
        except Exception as exc:
            skipped.append((os.path.basename(csv_path), f"read_error: {exc}"))
            continue

        df = resolve_index(df)
        prob_cols = find_prob_columns(df)
        if prob_cols is None:
            skipped.append((os.path.basename(csv_path), "missing prob columns"))
            continue

        report, reason = compute_low_pop_report(df, prob_cols, args.low_pop_threshold)
        if report is None:
            skipped.append((os.path.basename(csv_path), reason))
            continue

        report["file"] = os.path.basename(csv_path)
        report["low_pop_threshold"] = args.low_pop_threshold
        results.append(report)

    summary_df = pd.DataFrame(results).sort_values("mean_abs_diff_global")
    outdir = os.path.dirname(args.summary_out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    summary_df.to_csv(args.summary_out, index=False)

    print("Summary written to:", args.summary_out)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    if skipped:
        print("\nSkipped files:")
        for name, reason in skipped:
            print(f"- {name}: {reason}")


if __name__ == "__main__":
    main()
