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
EPSILON = 1e-12


def format_affgeoid(geoid):
    return f"1500000US{geoid}"


def prepare_data_step(main_data_csv, graph_file):
    df = pd.read_csv(main_data_csv, dtype={"AFFGEOID": str})
    df.set_index("AFFGEOID", inplace=True)

    # Filter to NY state (AFFGEOID starts with 1500000US36)
    df = df[df.index.str.startswith("1500000US36", na=False)]

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


def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(np.clip(x_shifted, -500, 500))
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + EPSILON)


def logits_to_probs(logits):
    return softmax(logits, axis=-1)


def initialize_probabilities(num_precincts, num_demos, num_vote_types, rng):
    total_samples = num_precincts * num_demos
    gammas = rng.gamma(1.0, 1.0, size=(total_samples, num_vote_types))
    p_flat = gammas / (gammas.sum(axis=1, keepdims=True) + EPSILON)
    p = p_flat.reshape(num_precincts, num_demos, num_vote_types)
    logits = np.log(np.maximum(p, EPSILON))
    logits = logits - logits.max(axis=2, keepdims=True)
    return p, logits


def compute_U_from_probs(p, D):
    return np.einsum("ijk,ij->ik", p, D)


def compute_loss_and_gradient_data_only(logits_flat, D, V):
    num_precincts, num_demos = D.shape
    num_vote_types = len(VOTE_TYPES)
    logits = logits_flat.reshape(num_precincts, num_demos, num_vote_types)
    p = logits_to_probs(logits)

    U = np.einsum("ijk,ij->ik", p, D)
    residual = U - V
    data_loss = np.sum(residual ** 2)

    grad_p = 2 * np.einsum("ik,ij->ijk", residual, D)
    grad_p_times_p = np.sum(grad_p * p, axis=2, keepdims=True)
    grad_logits = grad_p * p - p * grad_p_times_p

    return data_loss, grad_logits.flatten()


def adam_update(gradients, m, v, iteration, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m_new = beta1 * m + (1 - beta1) * gradients
    v_new = beta2 * v + (1 - beta2) * (gradients ** 2)
    m_hat = m_new / (1 - beta1 ** iteration)
    v_hat = v_new / (1 - beta2 ** iteration)
    update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return update, m_new, v_new


def compute_neighbor_avg(p, D, adj_matrix):
    num_precincts, num_demos, num_vote_types = p.shape
    neighbor_avg = np.zeros_like(p)
    for demo_idx in range(num_demos):
        p_demo = p[:, demo_idx, :]
        D_demo = D[:, demo_idx]
        neighbor_sum = adj_matrix @ (p_demo * D_demo[:, np.newaxis])
        neighbor_pop = adj_matrix @ D_demo
        neighbor_avg[:, demo_idx, :] = neighbor_sum / (neighbor_pop[:, np.newaxis] + EPSILON)
    return neighbor_avg


def apply_low_pop_noise(p, D, low_pop_threshold, noise_weight, rng):
    if noise_weight <= 0.0:
        return p
    low_pop_mask = D < low_pop_threshold
    if not np.any(low_pop_mask):
        return p
    num_vote_types = p.shape[2]
    rows, cols = np.where(low_pop_mask)
    noise = rng.gamma(1.0, 1.0, size=(len(rows), num_vote_types))
    noise = noise / (noise.sum(axis=1, keepdims=True) + EPSILON)
    for idx, (i, j) in enumerate(zip(rows, cols)):
        p[i, j, :] = (1.0 - noise_weight) * p[i, j, :] + noise_weight * noise[idx]
    return p


def run_optimization(
    p_init,
    logits_init,
    D,
    V,
    adj_matrix,
    iterations,
    learning_rate,
    smoothing_rate,
    noise_every,
    low_pop_threshold,
    noise_weight,
    rng,
):
    num_precincts, num_demos, num_vote_types = p_init.shape
    logits_flat = logits_init.flatten()
    m = np.zeros_like(logits_flat)
    v = np.zeros_like(logits_flat)
    p = p_init.copy()

    for iteration in tqdm(range(1, iterations + 1), desc="Optimization"):
        loss, grad = compute_loss_and_gradient_data_only(logits_flat, D, V)
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 100.0:
            grad = grad * (100.0 / (grad_norm + EPSILON))

        update, m, v = adam_update(grad, m, v, iteration, learning_rate)
        logits_flat = logits_flat - update

        logits = logits_flat.reshape(num_precincts, num_demos, num_vote_types)
        p = logits_to_probs(logits)

        if smoothing_rate > 0.0:
            neighbor_avg = compute_neighbor_avg(p, D, adj_matrix)
            p = (1.0 - smoothing_rate) * p + smoothing_rate * neighbor_avg

        if noise_every > 0 and iteration % noise_every == 0:
            p = apply_low_pop_noise(p, D, low_pop_threshold, noise_weight, rng)

        p = p / (p.sum(axis=2, keepdims=True) + EPSILON)
        logits = np.log(np.maximum(p, EPSILON))
        logits = logits - logits.max(axis=2, keepdims=True)
        logits_flat = logits.flatten()

        if iteration % 10 == 0 or iteration == 1:
            U = compute_U_from_probs(p, D)
            mae = float(np.mean(np.abs(U - V)))
            print(f"  Iteration {iteration}: loss={loss:.2f}, MAE(U,V)={mae:.4f}")

    return p


def main():
    parser = argparse.ArgumentParser(description="Neighbor-blend L2 optimization (Queens-only).")
    parser.add_argument("main_data_csv", help="Path to main data CSV")
    parser.add_argument("graph_file", help="Path to graph gpickle (GEOID-based)")
    parser.add_argument("--output", default="output/mle_agent_kv_neighbor_blend.csv", help="Output CSV path")
    parser.add_argument("--iterations", type=int, default=200, help="Number of iterations")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Adam learning rate")
    parser.add_argument("--smoothing-rate", type=float, default=0.2, help="Neighbor blend rate per iteration")
    parser.add_argument("--noise-every", type=int, default=5, help="Apply low-pop noise every N iterations (0 to disable)")
    parser.add_argument("--low-pop-threshold", type=float, default=10.0, help="Low-pop threshold on D")
    parser.add_argument("--noise-weight", type=float, default=0.15, help="Noise blend weight for low-pop cells")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    df, G = prepare_data_step(args.main_data_csv, args.graph_file)

    # Queens only
    df = df[df.index.str.startswith("1500000US36081", na=False)]
    df.sort_index(inplace=True)

    graph_nodes = {f"1500000US{node}" for node in G.nodes()}
    df_nodes = set(df.index)
    node_list = sorted(list(graph_nodes.intersection(df_nodes)))
    if not node_list:
        raise SystemExit("No Queens nodes overlap between data and graph.")
    df = df.loc[node_list]

    adj_matrix = nx.to_scipy_sparse_array(
        G, nodelist=[geoid.replace("1500000US", "") for geoid in node_list], format="csr"
    )

    V, D = preprocess_data_mle(df)
    num_precincts = len(df)
    num_demos = len(DEMOGRAPHICS)
    num_vote_types = len(VOTE_TYPES)

    print(f"Data shape: {num_precincts} precincts, {num_demos} demographics, {num_vote_types} vote types")

    p_init, logits_init = initialize_probabilities(num_precincts, num_demos, num_vote_types, rng)

    start = time.time()
    p_final = run_optimization(
        p_init,
        logits_init,
        D,
        V,
        adj_matrix,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        smoothing_rate=args.smoothing_rate,
        noise_every=args.noise_every,
        low_pop_threshold=args.low_pop_threshold,
        noise_weight=args.noise_weight,
        rng=rng,
    )
    runtime = time.time() - start
    print(f"Runtime: {runtime:.2f} sec")

    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    result_df = df.copy()
    for demo_idx, demo in enumerate(DEMOGRAPHICS):
        for vote_idx, vote_type in enumerate(VOTE_TYPES):
            result_df[f"{vote_type}_{demo}_prob"] = p_final[:, demo_idx, vote_idx]

    result_df.reset_index().to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
