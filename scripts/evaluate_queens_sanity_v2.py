"""
Evaluate Queens-only EI results with sanity checks and metrics.
"""
import argparse
import ast
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DEMOGRAPHICS = ["Wht", "His", "Blk", "Asn", "Oth"]
VOTE_TYPES = ["D", "R", "O", "N"]
EPSILON = 1e-12


def format_affgeoid(geoid):
    return f"1500000US{geoid}"


def parse_neighbors(value):
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def load_main_data(main_data_csv, county_fips):
    df = pd.read_csv(main_data_csv, dtype={"AFFGEOID": str})
    df = df[df["AFFGEOID"].str.startswith(format_affgeoid(county_fips), na=False)].copy()
    if df.empty:
        raise SystemExit(f"No rows found for county {county_fips}.")

    column_mapping = {
        "cvap_est_White Alone": "cvap_Wht",
        "cvap_est_Hispanic or Latino": "cvap_His",
        "cvap_est_Black or African American Alone": "cvap_Blk",
        "cvap_est_Asian Alone": "cvap_Asn",
        "cvap_est_American Indian or Alaska Native Alone": "cvap_aian",
        "cvap_est_Native Hawaiian or Other Pacific Islander Alone": "cvap_nhpi",
        "cvap_est_Mixed": "cvap_sor",
    }
    df.rename(columns=column_mapping, inplace=True)

    if "cvap_Oth" not in df.columns:
        df["cvap_Oth"] = df.get("cvap_aian", 0) + df.get("cvap_nhpi", 0) + df.get("cvap_sor", 0)
    if "cvap_total" not in df.columns:
        cvap_cols = [f"cvap_{demo}" for demo in DEMOGRAPHICS]
        df["cvap_total"] = df[cvap_cols].sum(axis=1)

    if "votes_D" not in df.columns:
        df["votes_D"] = df["D_Votes_2020"]
        df["votes_R"] = df["R_Votes_2020"]
        df["votes_O"] = df["O_Votes_2020"]
    if "votes_N" not in df.columns:
        total_votes = df["votes_D"] + df["votes_R"] + df["votes_O"]
        df["votes_N"] = (df["cvap_total"] - total_votes).clip(lower=0)

    df.set_index("AFFGEOID", inplace=True)
    return df


def build_adjacency(df):
    nodes = list(df.index)
    node_idx = {node: i for i, node in enumerate(nodes)}
    neighbors_list = df.get("neighbors")
    if neighbors_list is None:
        raise SystemExit("Missing neighbors column in main data CSV.")

    rows = []
    cols = []
    for node, raw_neighbors in neighbors_list.items():
        i = node_idx[node]
        neighbors = parse_neighbors(raw_neighbors)
        for nbr in neighbors:
            if nbr in node_idx:
                rows.append(i)
                cols.append(node_idx[nbr])
    data = np.ones(len(rows), dtype=float)
    adj = csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)))
    return adj, nodes


def preprocess_votes(df):
    vote_cols = [f"votes_{v}" for v in VOTE_TYPES]
    demo_cols = [f"cvap_{d}" for d in DEMOGRAPHICS]
    V = np.array([df[col].values.astype(float) for col in vote_cols]).T
    D = np.array([df[col].values.astype(float) for col in demo_cols]).T

    vote_totals = V.sum(axis=1)
    demo_totals = D.sum(axis=1)
    scaling_factors = demo_totals / (vote_totals + EPSILON)
    V_scaled = V * scaling_factors[:, np.newaxis]
    V_scaled = np.maximum(V_scaled, EPSILON)

    D = np.maximum(D, 1.0)
    vote_totals_scaled = V_scaled.sum(axis=1)
    renorm_factors = demo_totals / (vote_totals_scaled + EPSILON)
    V_scaled = V_scaled * renorm_factors[:, np.newaxis]

    return V_scaled, D


def extract_probabilities(results_df):
    num_precincts = len(results_df)
    num_demos = len(DEMOGRAPHICS)
    num_votes = len(VOTE_TYPES)
    p = np.zeros((num_precincts, num_demos, num_votes), dtype=float)
    for d_idx, demo in enumerate(DEMOGRAPHICS):
        for v_idx, vote in enumerate(VOTE_TYPES):
            col = f"{vote}_{demo}_prob"
            if col not in results_df.columns:
                raise SystemExit(f"Missing column {col} in results CSV.")
            p[:, d_idx, v_idx] = results_df[col].values.astype(float)
    p = np.maximum(p, EPSILON)
    p = p / (p.sum(axis=2, keepdims=True) + EPSILON)
    return p


def compute_margin(p):
    d = p[:, :, 0]
    r = p[:, :, 1]
    o = p[:, :, 2]
    denom = d + r + o + EPSILON
    return (d - r) / denom


def morans_i(values, adj_matrix):
    n = len(values)
    w_sum = adj_matrix.sum()
    if w_sum == 0:
        return 0.0
    z = values - np.mean(values)
    denom = np.sum(z ** 2)
    if denom == 0:
        return 0.0
    num = z @ (adj_matrix @ z)
    return (n / w_sum) * (num / denom)


def boundary_heuristic(neighbor_margins):
    if len(neighbor_margins) < 4:
        return False
    low = np.sum(neighbor_margins <= -0.2)
    high = np.sum(neighbor_margins >= 0.2)
    return (low >= 2) and (high >= 2)


def outlier_check(margins, adj_matrix):
    num_precincts, num_demos = margins.shape
    outliers = []
    for i in range(num_precincts):
        neighbors = adj_matrix[i].nonzero()[1]
        if neighbors.size == 0:
            continue
        neighbor_margins = margins[neighbors]
        for d in range(num_demos):
            vals = neighbor_margins[:, d]
            mean_val = np.mean(vals)
            if boundary_heuristic(vals):
                continue
            if abs(margins[i, d] - mean_val) > 0.50:
                outliers.append((i, d))
    return outliers


def variance_check(margins):
    return margins.var(axis=0)


def spatial_global_check(margins, adj_matrix, global_margin):
    spatial = []
    global_corr = []
    for d in range(margins.shape[1]):
        vals = margins[:, d]
        spatial.append(morans_i(vals, adj_matrix))
        if np.std(vals) < EPSILON or np.std(global_margin) < EPSILON:
            corr = 0.0
        else:
            corr = np.corrcoef(vals, global_margin)[0, 1]
        global_corr.append(corr)
    return np.array(spatial), np.array(global_corr)


def evaluate_results(main_df, adj_matrix, results_csv, baseline_p=None):
    results_df = pd.read_csv(results_csv, dtype={"AFFGEOID": str})
    results_df = results_df[results_df["AFFGEOID"].str.startswith(format_affgeoid("36081"), na=False)].copy()
    results_df.set_index("AFFGEOID", inplace=True)
    results_df = results_df.loc[main_df.index]

    p = extract_probabilities(results_df)
    V_scaled, D = preprocess_votes(main_df)

    U = np.einsum("ijk,ij->ik", p, D)
    mae_uv = float(np.mean(np.abs(U - V_scaled)))

    margins = compute_margin(p)

    baseline_mae = None
    if baseline_p is not None:
        baseline_margins = compute_margin(baseline_p)
        baseline_mae = float(np.mean(np.abs(margins - baseline_margins)))

    outliers = outlier_check(margins, adj_matrix)
    variances = variance_check(margins)

    global_margin = (main_df["votes_D"] - main_df["votes_R"]) / (
        main_df["votes_D"] + main_df["votes_R"] + main_df["votes_O"] + EPSILON
    )
    spatial, global_corr = spatial_global_check(margins, adj_matrix, global_margin.values)

    sanity_outliers = "FAIL" if outliers else "PASS"
    sanity_variance = "FAIL" if np.any(variances <= 0.05) else "PASS"
    sanity_spatial = "FAIL" if np.any((spatial < 0.1) & (global_corr > 0.9)) else "PASS"

    return {
        "mae_uv": mae_uv,
        "mae_vs_baseline": baseline_mae,
        "sanity_outliers": sanity_outliers,
        "sanity_variance": sanity_variance,
        "sanity_spatial": sanity_spatial,
        "spatial_autocorr": spatial,
        "global_corr": global_corr,
        "outliers_count": len(outliers),
        "variances": variances,
    }


def write_metrics(output_path, metrics_list):
    timestamp = datetime.now().isoformat(timespec="seconds")
    lines = []
    lines.append("=== BASELINE AND PAST METHODS (Queens) ===")
    lines.append(f"Generated: {timestamp}")
    lines.append("")
    for entry in metrics_list:
        lines.append(f"--- {entry['name']} ---")
        lines.append(f"File: {entry['path']}")
        lines.append(f"MAE_U_V={entry['metrics']['mae_uv']:.6f}")
        if entry["metrics"]["mae_vs_baseline"] is not None:
            lines.append(f"MAE_vs_baseline={entry['metrics']['mae_vs_baseline']:.6f}")
        lines.append(
            "Sanity: OUTLIERS={outliers}, VARIANCE={variance}, SPATIAL_GLOBAL={spatial}".format(
                outliers=entry["metrics"]["sanity_outliers"],
                variance=entry["metrics"]["sanity_variance"],
                spatial=entry["metrics"]["sanity_spatial"],
            )
        )
        spatial = entry["metrics"]["spatial_autocorr"]
        global_corr = entry["metrics"]["global_corr"]
        lines.append(f"Spatial autocorr (per demo): {np.round(spatial, 4).tolist()}")
        lines.append(f"Global corr (per demo): {np.round(global_corr, 4).tolist()}")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Queens EI results with sanity checks.")
    parser.add_argument("main_data_csv", help="Main data CSV (must include neighbors column).")
    parser.add_argument("--county-fips", default="36081")
    parser.add_argument("--baseline", required=True, help="Baseline results CSV for MAE comparison.")
    parser.add_argument("--results", nargs="+", required=True, help="Result CSVs to evaluate.")
    parser.add_argument("--output", default="output/baseline_and_past_metrics.txt", help="Output summary file.")
    args = parser.parse_args()

    main_df = load_main_data(args.main_data_csv, args.county_fips)
    adj_matrix, nodes = build_adjacency(main_df)
    main_df = main_df.loc[nodes]

    baseline_df = pd.read_csv(args.baseline, dtype={"AFFGEOID": str})
    baseline_df = baseline_df[baseline_df["AFFGEOID"].str.startswith(format_affgeoid(args.county_fips), na=False)].copy()
    baseline_df.set_index("AFFGEOID", inplace=True)
    baseline_df = baseline_df.loc[main_df.index]
    baseline_p = extract_probabilities(baseline_df)

    metrics_list = []
    for path in args.results:
        name = os.path.basename(path)
        metrics = evaluate_results(main_df, adj_matrix, path, baseline_p=baseline_p)
        metrics_list.append({"name": name, "path": path, "metrics": metrics})

    write_metrics(args.output, metrics_list)
    print(f"Wrote metrics to {args.output}")


if __name__ == "__main__":
    main()
