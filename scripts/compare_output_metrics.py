"""
Compare output CSVs on:
- avg results error (MAE% on vote totals)
- neighbor error (mean abs prob diff vs neighbor avg)
- global error (mean abs prob diff vs global avg)
"""
import argparse
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DEMOGRAPHICS = ["Wht", "His", "Blk", "Asn", "Oth"]
VOTE_TYPES = ["D", "R", "O", "N"]
EPSILON = 1e-10


def preprocess_data_mle(df):
    vote_cols = [f"votes_{vtype}" for vtype in VOTE_TYPES]
    V = np.array([df[col].values.astype(float) for col in vote_cols]).T

    demo_cols = [f"cvap_{demo}" for demo in DEMOGRAPHICS]
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


def compute_metrics(p, D, V, adj_matrix):
    U = np.einsum("ijk,ij->ik", p, D)
    V_sum = V.sum(axis=1)
    abs_err = np.abs(U - V).sum(axis=1)
    valid = V_sum > 0
    if np.any(valid):
        mae_percent = np.average(abs_err[valid] / (V_sum[valid] + EPSILON), weights=V_sum[valid])
    else:
        mae_percent = 0.0

    neighbor_diffs = []
    neighbor_weights = []
    for demo_idx in range(len(DEMOGRAPHICS)):
        p_demo = p[:, demo_idx, :]
        D_demo = D[:, demo_idx]
        neighbor_sum = adj_matrix @ (p_demo * D_demo[:, np.newaxis])
        neighbor_pop = adj_matrix @ D_demo
        neighbor_avg = neighbor_sum / (neighbor_pop[:, np.newaxis] + EPSILON)
        diff = np.abs(p_demo - neighbor_avg)
        valid_mask = neighbor_pop > 0
        if np.any(valid_mask):
            neighbor_diffs.append(diff[valid_mask])
            weights = np.repeat(D_demo[valid_mask][:, np.newaxis], len(VOTE_TYPES), axis=1)
            neighbor_weights.append(weights)
    if neighbor_diffs:
        diffs = np.concatenate(neighbor_diffs, axis=0)
        weights = np.concatenate(neighbor_weights, axis=0)
        neighbor_diff = np.average(diffs.ravel(), weights=weights.ravel())
    else:
        neighbor_diff = 0.0

    global_diffs = []
    global_weights = []
    for demo_idx in range(len(DEMOGRAPHICS)):
        p_demo = p[:, demo_idx, :]
        D_demo = D[:, demo_idx]
        total_pop = D_demo.sum()
        if total_pop <= 0:
            continue
        avg = np.average(p_demo, axis=0, weights=D_demo)
        diff = np.abs(p_demo - avg)
        global_diffs.append(diff)
        weights = np.repeat(D_demo[:, np.newaxis], len(VOTE_TYPES), axis=1)
        global_weights.append(weights)
    if global_diffs:
        diffs = np.concatenate(global_diffs, axis=0)
        weights = np.concatenate(global_weights, axis=0)
        global_diff = np.average(diffs.ravel(), weights=weights.ravel())
    else:
        global_diff = 0.0

    return mae_percent, neighbor_diff, global_diff


def main():
    parser = argparse.ArgumentParser(description="Compare output CSVs on MAE/neighbor/global errors")
    parser.add_argument("--output-dir", default="output", help="Directory with output CSVs")
    parser.add_argument(
        "--graph-file",
        default="blockgroups_graph_correct.gpickle",
        help="Graph file for neighbor calculations",
    )
    parser.add_argument(
        "--summary-out",
        default="output/output_metrics_summary.csv",
        help="Path for summary CSV",
    )
    args = parser.parse_args()

    with open(args.graph_file, "rb") as f:
        G = pickle.load(f)

    output_dir = args.output_dir
    csv_files = [
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.lower().endswith(".csv")
    ]

    results = []
    skipped = []
    for csv_path in sorted(csv_files):
        try:
            df = pd.read_csv(csv_path, dtype={"AFFGEOID": str})
        except Exception as exc:
            skipped.append((os.path.basename(csv_path), f"read_error: {exc}"))
            continue
        df = resolve_index(df)

        required_votes = [f"votes_{v}" for v in VOTE_TYPES]
        required_cvap = [f"cvap_{d}" for d in DEMOGRAPHICS]
        if not all(col in df.columns for col in required_votes + required_cvap):
            skipped.append((os.path.basename(csv_path), "missing vote/cvap columns"))
            continue

        prob_cols = find_prob_columns(df)
        if prob_cols is None:
            skipped.append((os.path.basename(csv_path), "missing prob columns"))
            continue

        graph_nodes = {f"1500000US{node}" for node in G.nodes()}
        df_nodes = set(df.index)
        node_list = sorted(list(graph_nodes.intersection(df_nodes)))
        if not node_list:
            skipped.append((os.path.basename(csv_path), "no graph/node overlap"))
            continue
        df = df.loc[node_list]

        adj_matrix = nx.to_scipy_sparse_array(
            G, nodelist=[geoid.replace("1500000US", "") for geoid in node_list], format="csr"
        )
        if not isinstance(adj_matrix, csr_matrix):
            adj_matrix = csr_matrix(adj_matrix)

        V, D = preprocess_data_mle(df)
        p = build_prob_tensor(df, prob_cols)
        mae, neighbor, global_diff = compute_metrics(p, D, V, adj_matrix)

        results.append(
            {
                "file": os.path.basename(csv_path),
                "n_rows": len(df),
                "mae_percent": mae,
                "neighbor_diff": neighbor,
                "global_diff": global_diff,
            }
        )

    summary_df = pd.DataFrame(results).sort_values("mae_percent")
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
