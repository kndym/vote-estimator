import argparse
import os
import time
import pickle
import numpy as np
import pandas as pd
import networkx as nx

EPSILON = 1e-12
DEMOGRAPHICS = ["Wht", "His", "Blk", "Asn", "Oth"]
VOTE_TYPES = ["D", "R", "O", "N"]


def format_affgeoid(geoid):
    return f"1500000US{geoid}"


def prepare_data(main_data_csv, graph_file, queens_geoid=None):
    df = pd.read_csv(main_data_csv, dtype={"AFFGEOID": str})
    df["geoid_part"] = df["AFFGEOID"].str.replace("1500000US", "", regex=False)
    if queens_geoid:
        queens_mask = df["geoid_part"].str.startswith(queens_geoid)
        df = df.loc[queens_mask]
    df = df.drop(columns=["geoid_part"])
    df.set_index("AFFGEOID", inplace=True)

    column_mapping = {
        "cvap_est_White Alone": "cvap_Wht",
        "cvap_est_Hispanic or Latino": "cvap_His",
        "cvap_est_Black or African American Alone": "cvap_Blk",
        "cvap_est_Asian Alone": "cvap_Asn",
        "cvap_est_American Indian or Alaska Native Alone": "cvap_aian",
        "cvap_est_Native Hawaiian or Other Pacific Islander Alone": "cvap_nhpi",
        "cvap_est_Mixed": "cvap_sor",
    }
    df = df.rename(columns=column_mapping)

    df["cvap_Oth"] = df["cvap_aian"] + df["cvap_nhpi"] + df["cvap_sor"]
    cvap_cols = ["cvap_Wht", "cvap_His", "cvap_Blk", "cvap_Asn", "cvap_Oth"]
    df["cvap_total"] = df[cvap_cols].sum(axis=1)

    df["votes_D"] = df["D_Votes_2020"]
    df["votes_R"] = df["R_Votes_2020"]
    df["votes_O"] = df["O_Votes_2020"]
    total_votes = df["votes_D"] + df["votes_R"] + df["votes_O"]
    df["votes_N"] = (df["cvap_total"] - total_votes).clip(lower=0)

    with open(graph_file, "rb") as f:
        G = pickle.load(f)

    graph_nodes = {format_affgeoid(node) for node in G.nodes()}
    node_list = sorted(graph_nodes.intersection(df.index))
    if len(node_list) != len(df):
        df = df.loc[node_list]

    adj_matrix = nx.to_scipy_sparse_array(
        G,
        nodelist=[geoid.replace("1500000US", "") for geoid in node_list],
        format="csr",
    )
    return df, adj_matrix


def preprocess_counts(df):
    vote_cols = [f"votes_{vtype}" for vtype in VOTE_TYPES]
    demo_cols = [f"cvap_{demo}" for demo in DEMOGRAPHICS]

    V = np.array([df[col].values.astype(float) for col in vote_cols]).T
    D = np.array([df[col].values.astype(float) for col in demo_cols]).T

    vote_totals = V.sum(axis=1)
    demo_totals = D.sum(axis=1)
    scaling_factors = demo_totals / (vote_totals + EPSILON)
    V_scaled = V * scaling_factors[:, np.newaxis]

    V_scaled = np.maximum(V_scaled, EPSILON)
    D = D + EPSILON
    D = np.maximum(D, 1.0)

    vote_totals_scaled = V_scaled.sum(axis=1)
    renorm_factors = demo_totals / (vote_totals_scaled + EPSILON)
    V_scaled = V_scaled * renorm_factors[:, np.newaxis]
    return V_scaled, D


def initialize_flat_dirichlet(num_precincts, num_demos, num_vote_types, rng):
    samples = rng.uniform(0.0, 1.0, size=(num_precincts, num_demos, num_vote_types))
    samples_sum = samples.sum(axis=2, keepdims=True) + EPSILON
    return samples / samples_sum


def compute_U(p, D):
    return np.einsum("ijk,ij->ik", p, D)


def adam_update(params, grads, m, v, step, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * (grads ** 2)
    m_hat = m / (1 - beta1 ** step)
    v_hat = v / (1 - beta2 ** step)
    params = params - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
    return params, m, v


def run_optimization(
    p,
    D,
    V,
    adj_matrix,
    iterations,
    learning_rate,
    spatial_weight,
    entropy_weight,
    noise_scale,
    low_pop_scale,
    rng,
):
    num_precincts, num_demos, num_vote_types = p.shape
    degrees = np.array(adj_matrix.sum(axis=1)).reshape(-1)
    uniform = np.full((num_vote_types,), 1.0 / num_vote_types)

    m = np.zeros_like(p)
    v = np.zeros_like(p)

    for step in range(1, iterations + 1):
        U = compute_U(p, D)
        residual = U - V
        grad_data = 2.0 * residual[:, np.newaxis, :] * D[:, :, np.newaxis]

        grad_spatial = np.zeros_like(p)
        for demo_idx in range(num_demos):
            p_demo = p[:, demo_idx, :]
            D_demo = D[:, demo_idx]
            neighbor_sum = adj_matrix @ (p_demo * D_demo[:, np.newaxis])
            neighbor_pop = adj_matrix @ D_demo
            neighbor_avg = neighbor_sum / (neighbor_pop[:, np.newaxis] + EPSILON)
            diff = p_demo - neighbor_avg
            grad_spatial[:, demo_idx, :] = 2.0 * diff

        grad_entropy = np.log(p + EPSILON) + 1.0
        grads = grad_data + spatial_weight * grad_spatial + entropy_weight * grad_entropy

        if noise_scale > 0:
            low_pop_weight = np.exp(-D / (low_pop_scale + EPSILON))
            noise = rng.normal(0.0, noise_scale, size=p.shape)
            grads = grads + noise * low_pop_weight[:, :, np.newaxis]

        p, m, v = adam_update(p, grads, m, v, step, learning_rate)

        p = np.clip(p, EPSILON, None)
        p = p / (p.sum(axis=2, keepdims=True) + EPSILON)

        low_pop_weight = np.exp(-D / (low_pop_scale + EPSILON))
        neighbor_avg = np.zeros_like(p)
        for demo_idx in range(num_demos):
            p_demo = p[:, demo_idx, :]
            D_demo = D[:, demo_idx]
            neighbor_sum = adj_matrix @ (p_demo * D_demo[:, np.newaxis])
            neighbor_pop = adj_matrix @ D_demo
            neighbor_avg[:, demo_idx, :] = neighbor_sum / (neighbor_pop[:, np.newaxis] + EPSILON)
        p = (1 - low_pop_weight[:, :, np.newaxis]) * p + low_pop_weight[:, :, np.newaxis] * neighbor_avg

        if step % 50 == 0 or step == 1:
            data_loss = np.mean(np.abs(residual))
            print(f"Step {step}: mean_abs_residual={data_loss:.4f}")

    return p


def main():
    parser = argparse.ArgumentParser(description="Laplacian + entropy regularized EI.")
    parser.add_argument("main_data_csv", help="Path to the main data CSV.")
    parser.add_argument("graph_file", help="Path to the graph gpickle file.")
    parser.add_argument("--output", default="output/mle_agent_kv_lapent_results.csv")
    parser.add_argument("--queens-geoid", default=None, help="Optional county GEOID prefix (e.g., 36081).")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--spatial-weight", type=float, default=0.4)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--noise-scale", type=float, default=0.005)
    parser.add_argument("--low-pop-scale", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    start = time.time()

    df, adj_matrix = prepare_data(args.main_data_csv, args.graph_file, queens_geoid=args.queens_geoid)
    if df.empty:
        raise ValueError("No Queens rows found after filtering.")

    V, D = preprocess_counts(df)
    num_precincts = len(df)
    p = initialize_flat_dirichlet(num_precincts, len(DEMOGRAPHICS), len(VOTE_TYPES), rng)

    p_final = run_optimization(
        p,
        D,
        V,
        adj_matrix,
        args.iterations,
        args.learning_rate,
        args.spatial_weight,
        args.entropy_weight,
        args.noise_scale,
        args.low_pop_scale,
        rng,
    )

    result_df = df.copy()
    for demo_idx, demo in enumerate(DEMOGRAPHICS):
        for vote_idx, vote_type in enumerate(VOTE_TYPES):
            result_df[f"{vote_type}_{demo}_prob"] = p_final[:, demo_idx, vote_idx]

    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    result_df.reset_index().to_csv(args.output, index=False)

    runtime = time.time() - start
    print(f"Saved results to {args.output}")
    print(f"Runtime (sec): {runtime:.2f}")


if __name__ == "__main__":
    main()
