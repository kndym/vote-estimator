"""
Population-weighted L2 logit with entropy prior (Queens only).

Method:
  - Data fit: population-weighted L2 between U and V
  - Spatial: population-weighted neighbor smoothing
  - Entropy: low-pop cells pushed toward higher-entropy (flatter) probabilities
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

DEMOGRAPHICS = ['Wht', 'His', 'Blk', 'Asn', 'Oth']
VOTE_TYPES = ['D', 'R', 'O', 'N']
EPSILON = 1e-10


def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(np.clip(x_shifted, -500, 500))
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + EPSILON)


def logits_to_probs(logits):
    return softmax(logits, axis=-1)


def prepare_data_step(main_data_csv, graph_file):
    print(f"Loading main data from {main_data_csv}...")
    df = pd.read_csv(main_data_csv, dtype={'AFFGEOID': str})
    df.set_index('AFFGEOID', inplace=True)

    # Queens only (GEOID 36081)
    df = df[df.index.str.startswith('1500000US36081', na=False)]
    print(f"Filtered to Queens: {len(df)} rows.")

    print(f"Loading graph from {graph_file}...")
    with open(graph_file, 'rb') as f:
        G = pickle.load(f)

    column_mapping = {
        'cvap_est_White Alone': 'cvap_Wht',
        'cvap_est_Hispanic or Latino': 'cvap_His',
        'cvap_est_Black or African American Alone': 'cvap_Blk',
        'cvap_est_Asian Alone': 'cvap_Asn',
        'cvap_est_American Indian or Alaska Native Alone': 'cvap_aian',
        'cvap_est_Native Hawaiian or Other Pacific Islander Alone': 'cvap_nhpi',
        'cvap_est_Mixed': 'cvap_sor'
    }
    df.rename(columns=column_mapping, inplace=True)

    df['cvap_Oth'] = df['cvap_aian'] + df['cvap_nhpi'] + df['cvap_sor']
    cvap_cols = ['cvap_Wht', 'cvap_His', 'cvap_Blk', 'cvap_Asn', 'cvap_Oth']
    df['cvap_total'] = df[cvap_cols].sum(axis=1)

    df['votes_D'] = df['D_Votes_2020']
    df['votes_R'] = df['R_Votes_2020']
    df['votes_O'] = df['O_Votes_2020']
    total_votes = df['votes_D'] + df['votes_R'] + df['votes_O']
    df['votes_N'] = (df['cvap_total'] - total_votes).clip(lower=0)

    return df, G


def preprocess_data_mle(df):
    vote_cols = [f'votes_{vtype}' for vtype in VOTE_TYPES]
    V = np.array([df[col].values.astype(float) for col in vote_cols]).T

    demo_cols = [f'cvap_{demo}' for demo in DEMOGRAPHICS]
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

    s = demo_totals
    return V_scaled, D, s


def initialize_logits(num_precincts, num_demos, num_vote_types, rng):
    total_samples = num_precincts * num_demos
    gammas = rng.gamma(1.0, 1.0, size=(total_samples, num_vote_types))
    p_flat = gammas / (gammas.sum(axis=1, keepdims=True) + EPSILON)
    p = p_flat.reshape(num_precincts, num_demos, num_vote_types)

    p_safe = np.maximum(p, EPSILON)
    log_p = np.log(p_safe)
    logits = log_p - log_p.max(axis=2, keepdims=True)
    return logits


def compute_loss_and_gradient(
    logits_flat,
    D,
    V,
    adj_matrix,
    spatial_weight,
    entropy_weight,
    low_pop_threshold,
    precinct_weights,
):
    num_precincts, num_demos = D.shape
    num_vote_types = len(VOTE_TYPES)

    logits = logits_flat.reshape(num_precincts, num_demos, num_vote_types)
    p = logits_to_probs(logits)

    U = np.einsum('ijk,ij->ik', p, D)
    data_residual = U - V
    data_loss = np.sum((data_residual ** 2) * precinct_weights[:, np.newaxis])

    weighted_residual = data_residual * precinct_weights[:, np.newaxis]
    grad_p_data = 2 * np.einsum('ik,ij->ijk', weighted_residual, D)

    spatial_loss = 0.0
    entropy_loss = 0.0
    grad_p_spatial = np.zeros_like(p)
    grad_p_entropy = np.zeros_like(p)

    for demo_idx in range(num_demos):
        p_demo = p[:, demo_idx, :]
        D_demo = D[:, demo_idx]

        neighbor_sum = adj_matrix @ (p_demo * D_demo[:, np.newaxis])
        neighbor_pop = adj_matrix @ D_demo
        neighbor_avg = neighbor_sum / (neighbor_pop[:, np.newaxis] + EPSILON)

        spatial_pop_weight = D_demo / (D_demo + low_pop_threshold)
        diff = p_demo - neighbor_avg
        spatial_loss += np.sum(spatial_pop_weight[:, np.newaxis] * (diff ** 2))

        grad_diff = 2 * diff * spatial_pop_weight[:, np.newaxis]
        grad_p_spatial[:, demo_idx, :] += grad_diff

        for vote_idx in range(num_vote_types):
            neighbor_grad = adj_matrix.T @ (grad_diff[:, vote_idx] / (neighbor_pop + EPSILON))
            grad_p_spatial[:, demo_idx, vote_idx] -= neighbor_grad * D_demo

        entropy_pop_weight = low_pop_threshold / (D_demo + low_pop_threshold)
        entropy_term = p_demo * np.log(p_demo + EPSILON)
        entropy_loss += entropy_weight * np.sum(entropy_pop_weight[:, np.newaxis] * entropy_term)
        grad_p_entropy[:, demo_idx, :] = entropy_weight * entropy_pop_weight[:, np.newaxis] * (1.0 + np.log(p_demo + EPSILON))
    total_loss = data_loss + spatial_weight * spatial_loss + entropy_loss
    grad_p = grad_p_data + spatial_weight * grad_p_spatial + grad_p_entropy

    grad_p_times_p = np.sum(grad_p * p, axis=2, keepdims=True)
    grad_logits = grad_p * p - p * grad_p_times_p

    return total_loss, grad_logits.flatten()


def adam_update(gradients, m, v, iteration, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m_new = beta1 * m + (1 - beta1) * gradients
    v_new = beta2 * v + (1 - beta2) * (gradients ** 2)
    m_hat = m_new / (1 - beta1 ** iteration)
    v_hat = v_new / (1 - beta2 ** iteration)
    update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return update, m_new, v_new


def optimize_pwle(
    D,
    V,
    adj_matrix,
    spatial_weight,
    entropy_weight,
    learning_rate,
    max_iterations,
    low_pop_threshold,
    last_n_avg,
    rng,
):
    num_precincts, num_demos = D.shape
    num_vote_types = len(VOTE_TYPES)

    logits = initialize_logits(num_precincts, num_demos, num_vote_types, rng)
    logits_flat = logits.flatten()

    m = np.zeros_like(logits_flat)
    v = np.zeros_like(logits_flat)

    precinct_weights = np.sqrt(D.sum(axis=1) + EPSILON)
    precinct_weights = precinct_weights / (precinct_weights.mean() + EPSILON)

    low_pop_mask = D < low_pop_threshold
    use_low_pop_logic = last_n_avg > 0 and np.any(low_pop_mask)
    buf = [None] * last_n_avg if use_low_pop_logic else None

    print("\nStarting PWLE optimization...")
    print(f"Spatial weight: {spatial_weight}, Entropy weight: {entropy_weight}, LR: {learning_rate}")
    if use_low_pop_logic:
        print(f"Low-pop: threshold={low_pop_threshold}, last_n_avg={last_n_avg}")

    for iteration in tqdm(range(1, max_iterations + 1), desc="Optimization"):
        loss, grad = compute_loss_and_gradient(
            logits_flat,
            D,
            V,
            adj_matrix,
            spatial_weight,
            entropy_weight,
            low_pop_threshold,
            precinct_weights,
        )

        grad_norm = np.linalg.norm(grad)
        if grad_norm > 100.0:
            grad = grad * (100.0 / (grad_norm + EPSILON))

        update, m, v = adam_update(grad, m, v, iteration, learning_rate)
        logits_flat = logits_flat - update

        if use_low_pop_logic:
            logits = logits_flat.reshape(num_precincts, num_demos, num_vote_types)
            p = logits_to_probs(logits)
            buf[(iteration - 1) % last_n_avg] = p.copy()

        if iteration % 10 == 0 or iteration == 1:
            logits = logits_flat.reshape(num_precincts, num_demos, num_vote_types)
            p = logits_to_probs(logits)
            U = np.einsum('ijk,ij->ik', p, D)
            U_total = U.sum(axis=0)
            V_total = V.sum(axis=0)
            U_props = U_total / (U_total.sum() + EPSILON)
            V_props = V_total / (V_total.sum() + EPSILON)
            mae = np.mean(np.abs(U_props - V_props))
            print(f"  Iteration {iteration}: loss={loss:.2f}, MAE={mae:.4f}")

    logits = logits_flat.reshape(num_precincts, num_demos, num_vote_types)
    p = logits_to_probs(logits)

    if not use_low_pop_logic:
        return p

    valid_buf = [b for b in buf if b is not None]
    if not valid_buf:
        return p

    p_out = np.copy(p)
    for i in range(num_precincts):
        for demo_idx in range(num_demos):
            if not low_pop_mask[i, demo_idx]:
                continue
            p_out[i, demo_idx, :] = np.mean([b[i, demo_idx, :] for b in valid_buf], axis=0)

    return p_out


def main():
    parser = argparse.ArgumentParser(description="Population-weighted L2 logit + entropy prior (Queens only).")
    parser.add_argument("main_data_csv", help="Path to main data CSV")
    parser.add_argument("graph_file", help="Path to graph gpickle (GEOID-based)")
    parser.add_argument("--output", default="output/mle_pwle_queens.csv", help="Output CSV path")
    parser.add_argument("--iterations", type=int, default=150, help="Number of iterations")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--spatial-weight", type=float, default=0.2, help="Spatial smoothing weight")
    parser.add_argument("--entropy-weight", type=float, default=0.05, help="Low-pop entropy weight")
    parser.add_argument("--low-pop-threshold", type=float, default=10.0, help="Low-pop threshold (cvap)")
    parser.add_argument("--last-n-avg", type=int, default=20, help="Low-pop: mean p over last N iters (0=off)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    df, G = prepare_data_step(args.main_data_csv, args.graph_file)
    df.sort_index(inplace=True)

    graph_nodes = {f"1500000US{node}" for node in G.nodes()}
    df_nodes = set(df.index)
    node_list = sorted(list(graph_nodes.intersection(df_nodes)))
    if not node_list:
        raise SystemExit("No block groups overlap between data and graph.")
    df = df.loc[node_list]

    adj_matrix = nx.to_scipy_sparse_array(
        G, nodelist=[geoid.replace("1500000US", "") for geoid in node_list], format="csr"
    )

    V, D, _ = preprocess_data_mle(df)
    print(f"\nData shape: {len(df)} precincts, {len(DEMOGRAPHICS)} demos, {len(VOTE_TYPES)} vote types")

    p_final = optimize_pwle(
        D,
        V,
        adj_matrix,
        spatial_weight=args.spatial_weight,
        entropy_weight=args.entropy_weight,
        learning_rate=args.learning_rate,
        max_iterations=args.iterations,
        low_pop_threshold=args.low_pop_threshold,
        last_n_avg=args.last_n_avg,
        rng=rng,
    )

    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    result_df = df.copy()
    for demo_idx, demo in enumerate(DEMOGRAPHICS):
        for vote_idx, vote_type in enumerate(VOTE_TYPES):
            result_df[f'{vote_type}_{demo}_prob'] = p_final[:, demo_idx, vote_idx]

    result_df.to_csv(args.output, index=True)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
