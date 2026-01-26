"""
Agent method: L2 data fit with symmetric neighbor KL smoothing in logit space.
Queens-only run, flat-Dirichlet init (Uniform -> normalize), Adam optimizer.
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

DEMOGRAPHICS = ["Wht", "His", "Blk", "Asn", "Oth"]
VOTE_TYPES = ["D", "R", "O", "N"]
EPSILON = 1e-10


def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(np.clip(x_shifted, -500, 500))
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + EPSILON)


def logits_to_probs(logits):
    return softmax(logits, axis=-1)


def prepare_data_step(main_data_csv, graph_file, county_fips="36081"):
    print(f"Loading main data from {main_data_csv}...")
    df = pd.read_csv(main_data_csv, dtype={"AFFGEOID": str})
    df.set_index("AFFGEOID", inplace=True)

    queens_prefix = f"1500000US{county_fips}"
    df = df[df.index.str.startswith(queens_prefix, na=False)]
    print(f"Filtered to Queens (GEOID {county_fips}): {len(df)} rows.")

    print(f"Loading graph from {graph_file}...")
    with open(graph_file, "rb") as f:
        G = pickle.load(f)

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

    df["cvap_Oth"] = df["cvap_aian"] + df["cvap_nhpi"] + df["cvap_sor"]
    cvap_cols = ["cvap_Wht", "cvap_His", "cvap_Blk", "cvap_Asn", "cvap_Oth"]
    df["cvap_total"] = df[cvap_cols].sum(axis=1)

    df["votes_D"] = df["D_Votes_2020"]
    df["votes_R"] = df["R_Votes_2020"]
    df["votes_O"] = df["O_Votes_2020"]
    total_votes = df["votes_D"] + df["votes_R"] + df["votes_O"]
    df["votes_N"] = df["cvap_total"] - total_votes
    df["votes_N"] = df["votes_N"].clip(lower=0)

    return df, G


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


def initialize_logits(num_precincts, num_demos, num_vote_types, rng):
    total_samples = num_precincts * num_demos
    raw = rng.random(size=(total_samples, num_vote_types))
    p_flat = raw / (raw.sum(axis=1, keepdims=True) + EPSILON)
    p = p_flat.reshape(num_precincts, num_demos, num_vote_types)

    p_safe = np.maximum(p, EPSILON)
    log_p = np.log(p_safe)
    logits = log_p - log_p.max(axis=2, keepdims=True)

    return logits


def compute_U_from_logits(logits, D):
    p = logits_to_probs(logits)
    U = np.einsum("ijk,ij->ik", p, D)
    return U


def compute_loss_and_gradient(logits_flat, D, V, adj_matrix, spatial_weight):
    num_precincts, num_demos = D.shape
    num_vote_types = len(VOTE_TYPES)

    logits = logits_flat.reshape(num_precincts, num_demos, num_vote_types)
    p = logits_to_probs(logits)

    U = np.einsum("ijk,ij->ik", p, D)
    data_residual = U - V
    data_loss = np.sum(data_residual ** 2)
    grad_p_data = 2 * np.einsum("ik,ij->ijk", data_residual, D)

    degrees = np.asarray(adj_matrix.sum(axis=1)).reshape(-1)
    spatial_loss = 0.0
    grad_p_spatial = np.zeros_like(p)

    for demo_idx in range(num_demos):
        p_demo = p[:, demo_idx, :]
        log_p_demo = np.log(np.maximum(p_demo, EPSILON))
        neighbor_logsum = adj_matrix @ log_p_demo

        entropy_term = (p_demo * log_p_demo).sum(axis=1)
        spatial_loss += np.sum(degrees * entropy_term - (p_demo * neighbor_logsum).sum(axis=1))

        grad_p_spatial[:, demo_idx, :] = (
            degrees[:, np.newaxis] * (log_p_demo + 1.0) - neighbor_logsum
        )

    total_loss = data_loss + spatial_weight * spatial_loss
    grad_p = grad_p_data + spatial_weight * grad_p_spatial

    grad_p_times_p = np.sum(grad_p * p, axis=2, keepdims=True)
    grad_logits = grad_p * p - p * grad_p_times_p
    grad_logits_flat = grad_logits.flatten()

    return total_loss, grad_logits_flat


def adam_update(gradients, m, v, iteration, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m_new = beta1 * m + (1 - beta1) * gradients
    v_new = beta2 * v + (1 - beta2) * (gradients ** 2)
    m_hat = m_new / (1 - beta1 ** iteration)
    v_hat = v_new / (1 - beta2 ** iteration)
    update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return update, m_new, v_new


def optimize_kl_spatial(
    D,
    V,
    adj_matrix,
    spatial_weight=0.2,
    max_iterations=200,
    learning_rate=0.01,
    rng=None,
    low_pop_threshold=10.0,
    last_n_avg=20,
    prior_blend=False,
):
    if rng is None:
        rng = np.random.default_rng(42)

    num_precincts, num_demos = D.shape
    num_vote_types = len(VOTE_TYPES)

    low_pop_mask = (D < low_pop_threshold)
    n_low_pop = int(np.sum(low_pop_mask))
    use_low_pop_logic = (last_n_avg > 0) and (n_low_pop > 0)

    logits = initialize_logits(num_precincts, num_demos, num_vote_types, rng)
    logits_flat = logits.flatten()

    m = np.zeros_like(logits_flat)
    v = np.zeros_like(logits_flat)

    buf = [None] * last_n_avg if use_low_pop_logic else None

    print("\nStarting KL-spatial optimization in logit space...")
    print(f"Spatial weight: {spatial_weight}, Learning rate: {learning_rate}, Optimizer: Adam")
    if use_low_pop_logic:
        print(f"Low-pop: threshold={low_pop_threshold}, last_n_avg={last_n_avg}, prior_blend={prior_blend}, low-pop cells={n_low_pop}")

    for iteration in tqdm(range(1, max_iterations + 1), desc="Optimization"):
        loss, grad = compute_loss_and_gradient(logits_flat, D, V, adj_matrix, spatial_weight)

        grad_norm = np.linalg.norm(grad)
        if grad_norm > 100.0:
            grad = grad * (100.0 / (grad_norm + EPSILON))

        update, m, v = adam_update(grad, m, v, iteration, learning_rate)
        logits_flat = logits_flat - update

        if use_low_pop_logic:
            logits = logits_flat.reshape(num_precincts, num_demos, num_vote_types)
            p = logits_to_probs(logits)
            slot = (iteration - 1) % last_n_avg
            buf[slot] = p.copy()

        if iteration % 10 == 0 or iteration == 1:
            logits = logits_flat.reshape(num_precincts, num_demos, num_vote_types)
            U = compute_U_from_logits(logits, D)
            U_total = U.sum(axis=0)
            V_total = V.sum(axis=0)
            U_props = U_total / (U_total.sum() + EPSILON)
            V_props = V_total / (V_total.sum() + EPSILON)
            mae = np.mean(np.abs(U_props - V_props))
            print(f"  Iteration {iteration}: loss={loss:.2f}, MAE={mae:.4f}")
            print(f"    U: D={U_props[0]:.4f}, R={U_props[1]:.4f}, O={U_props[2]:.4f}, N={U_props[3]:.4f}")
            print(f"    V: D={V_props[0]:.4f}, R={V_props[1]:.4f}, O={V_props[2]:.4f}, N={V_props[3]:.4f}")

    logits = logits_flat.reshape(num_precincts, num_demos, num_vote_types)
    p = logits_to_probs(logits)

    if not use_low_pop_logic:
        return p, logits

    prior = np.mean(p, axis=0)
    p_out = np.copy(p)
    valid_buf = [b for b in buf if b is not None]
    n_stored = len(valid_buf)

    print(f"  Low-pop post-process: averaged over last {n_stored} iters for {n_low_pop} cells")
    for i in range(num_precincts):
        for demo_idx in range(num_demos):
            if not low_pop_mask[i, demo_idx]:
                continue
            p_avg = np.mean([b[i, demo_idx, :] for b in valid_buf], axis=0)
            if prior_blend:
                alpha = np.clip(1.0 - (D[i, demo_idx] / (low_pop_threshold + EPSILON)), 0.0, 1.0)
                p_out[i, demo_idx, :] = (1.0 - alpha) * p_avg + alpha * prior[demo_idx, :]
            else:
                p_out[i, demo_idx, :] = p_avg

    return p_out, logits


def main():
    parser = argparse.ArgumentParser(description="KL-spatial L2 optimization in logit space (Queens only)")
    parser.add_argument("main_data_csv", help="Path to main data CSV")
    parser.add_argument("graph_file", help="Path to graph gpickle (GEOID-based)")
    parser.add_argument("--spatial-weight", type=float, default=0.2, help="Weight for KL neighbor smoothing")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--iterations", type=int, default=200, help="Number of iterations")
    parser.add_argument("--output", default="output/mle_agent_kl_spatial_queens.csv", help="Output CSV file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--low-pop-threshold", type=float, default=10.0, help="Low-pop threshold on D")
    parser.add_argument("--last-n-avg", type=int, default=20, help="Low-pop output = mean of last N iters; 0 disables")
    parser.add_argument("--prior-blend", action="store_true", help="Blend low-pop avg with demographic prior")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    df, G = prepare_data_step(args.main_data_csv, args.graph_file)
    df.sort_index(inplace=True)

    graph_nodes = {f"1500000US{node}" for node in G.nodes()}
    df_nodes = set(df.index)
    node_list = sorted(list(graph_nodes.intersection(df_nodes)))
    if not node_list:
        raise SystemExit(
            "No block groups overlap between data and graph. "
            "Use a GEOID-based graph that matches your data."
        )

    df = df.loc[node_list]

    adj_matrix = nx.to_scipy_sparse_array(
        G, nodelist=[geoid.replace("1500000US", "") for geoid in node_list], format="csr"
    )

    V, D = preprocess_data_mle(df)
    print(f"\nData shape: {len(df)} precincts, {len(DEMOGRAPHICS)} demographics, {len(VOTE_TYPES)} vote types")

    start_time = time.time()
    p_final, logits_final = optimize_kl_spatial(
        D,
        V,
        adj_matrix,
        spatial_weight=args.spatial_weight,
        max_iterations=args.iterations,
        learning_rate=args.learning_rate,
        rng=rng,
        low_pop_threshold=args.low_pop_threshold,
        last_n_avg=args.last_n_avg,
        prior_blend=args.prior_blend,
    )
    runtime_sec = time.time() - start_time
    print(f"Runtime (sec): {runtime_sec:.2f}")

    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    print(f"\nSaving results to {args.output}...")
    result_df = df.copy()

    for demo_idx, demo in enumerate(DEMOGRAPHICS):
        for vote_idx, vote_type in enumerate(VOTE_TYPES):
            result_df[f"{vote_type}_{demo}_prob"] = p_final[:, demo_idx, vote_idx]

    result_df.to_csv(args.output, index=True)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
