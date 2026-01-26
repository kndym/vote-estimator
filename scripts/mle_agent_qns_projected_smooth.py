"""
Projected-gradient EI with neighbor averaging (Queens only).

Method:
  - Initialize p with flat Dirichlet (Uniform->normalize) per (precinct, demo).
  - Minimize ||U - V||^2 in probability space with projected steps.
  - Apply demographic-wise neighbor averaging for spatial smoothing.
  - For low-pop (precinct, demo), blend with flat prior noise each iteration.
"""
import argparse
import ast
import os
import time

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


def prepare_data(main_data_csv, county_fips):
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


def build_adjacency_from_neighbors(df):
    nodes = list(df.index)
    node_idx = {node: i for i, node in enumerate(nodes)}
    neighbors_list = df.get("neighbors")
    if neighbors_list is None:
        raise SystemExit("Missing neighbors column in input data.")

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


def preprocess_data(df):
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


def initialize_probabilities(num_precincts, num_demos, num_vote_types, rng):
    total_samples = num_precincts * num_demos
    gammas = rng.gamma(1.0, 1.0, size=(total_samples, num_vote_types))
    p_flat = gammas / (gammas.sum(axis=1, keepdims=True) + EPSILON)
    p = p_flat.reshape(num_precincts, num_demos, num_vote_types)
    return p


def compute_U(p, D):
    return np.einsum("ijk,ij->ik", p, D)


def neighbor_average(p, D, adj_matrix):
    num_precincts, num_demos, num_vote_types = p.shape
    p_smoothed = np.copy(p)
    for demo_idx in range(num_demos):
        p_demo = p[:, demo_idx, :]
        D_demo = D[:, demo_idx]
        neighbor_sum = adj_matrix @ (p_demo * D_demo[:, np.newaxis])
        neighbor_pop = adj_matrix @ D_demo
        neighbor_avg = neighbor_sum / (neighbor_pop[:, np.newaxis] + EPSILON)
        has_neighbors = neighbor_pop > 0
        if np.any(has_neighbors):
            p_smoothed[has_neighbors, demo_idx, :] = neighbor_avg[has_neighbors]
    return p_smoothed


def projected_gradient_descent(
    p,
    D,
    V,
    adj_matrix,
    rng,
    iterations,
    learning_rate,
    spatial_alpha,
    low_pop_threshold,
    low_pop_blend,
):
    num_precincts, num_demos, num_vote_types = p.shape
    D_totals = D.sum(axis=1)

    for iteration in range(1, iterations + 1):
        U = compute_U(p, D)
        residual = U - V
        grad_p = 2.0 * residual[:, np.newaxis, :] * D[:, :, np.newaxis]
        grad_p = grad_p / (D_totals[:, np.newaxis, np.newaxis] + EPSILON)

        p = p - learning_rate * grad_p
        p = np.maximum(p, EPSILON)
        p = p / (p.sum(axis=2, keepdims=True) + EPSILON)

        if spatial_alpha > 0:
            p_neighbor = neighbor_average(p, D, adj_matrix)
            p = (1.0 - spatial_alpha) * p + spatial_alpha * p_neighbor
            p = np.maximum(p, EPSILON)
            p = p / (p.sum(axis=2, keepdims=True) + EPSILON)

        if low_pop_blend > 0:
            low_pop_mask = D < low_pop_threshold
            if np.any(low_pop_mask):
                n_low = int(np.sum(low_pop_mask))
                noise = rng.dirichlet(np.ones(num_vote_types), size=n_low)
                flat_indices = np.argwhere(low_pop_mask)
                for idx, (i, d) in enumerate(flat_indices):
                    p[i, d, :] = (1.0 - low_pop_blend) * p[i, d, :] + low_pop_blend * noise[idx]

        if iteration % 10 == 0 or iteration == 1:
            mae = np.mean(np.abs(U - V))
            print(f"Iteration {iteration}: MAE(U,V)={mae:.6f}")

    return p


def main():
    parser = argparse.ArgumentParser(description="Projected-gradient EI with neighbor averaging (Queens only).")
    parser.add_argument("main_data_csv", help="Main data CSV (must include AFFGEOID, votes, cvap, neighbors).")
    parser.add_argument("--county-fips", default="36081", help="County FIPS (default: Queens = 36081).")
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--spatial-alpha", type=float, default=0.2, help="Neighbor averaging weight.")
    parser.add_argument("--low-pop-threshold", type=float, default=25.0)
    parser.add_argument("--low-pop-blend", type=float, default=0.15, help="Blend weight for flat prior noise.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="output/mle_agent_qns_projected_smooth.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    df = prepare_data(args.main_data_csv, args.county_fips)
    adj_matrix, nodes = build_adjacency_from_neighbors(df)
    df = df.loc[nodes]

    V, D = preprocess_data(df)
    num_precincts, num_demos = D.shape
    print(f"Queens data shape: {num_precincts} precincts, {num_demos} demographics")

    p_init = initialize_probabilities(num_precincts, num_demos, len(VOTE_TYPES), rng)

    start = time.time()
    p_final = projected_gradient_descent(
        p_init,
        D,
        V,
        adj_matrix,
        rng=rng,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        spatial_alpha=args.spatial_alpha,
        low_pop_threshold=args.low_pop_threshold,
        low_pop_blend=args.low_pop_blend,
    )
    runtime = time.time() - start

    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    result_df = df.copy()
    for demo_idx, demo in enumerate(DEMOGRAPHICS):
        for vote_idx, vote_type in enumerate(VOTE_TYPES):
            result_df[f"{vote_type}_{demo}_prob"] = p_final[:, demo_idx, vote_idx]

    result_df.reset_index().to_csv(args.output, index=False)
    print(f"Saved results to {args.output}")
    print(f"Runtime seconds: {runtime:.2f}")


if __name__ == "__main__":
    main()
