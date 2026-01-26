"""
Staged logit-space optimization with guardrails:
1) Precinct accuracy (vote totals MAE%) < threshold
2) Neighbor demo diff < threshold (while keeping stage 1 satisfied)
3) Global avg diff < threshold (while keeping stages 1-2 satisfied)
"""
import argparse
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
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


def prepare_data_step(main_data_csv, graph_file):
    print(f"Loading main data from {main_data_csv}...")
    df = pd.read_csv(main_data_csv, dtype={"AFFGEOID": str})
    df.set_index("AFFGEOID", inplace=True)

    df = df[df.index.str.startswith("1500000US36", na=False)]
    print(f"Filtered to New York state: {len(df)} rows.")

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
    gammas = rng.gamma(1.0, 1.0, size=(total_samples, num_vote_types))
    p_flat = gammas / (gammas.sum(axis=1, keepdims=True) + EPSILON)
    p = p_flat.reshape(num_precincts, num_demos, num_vote_types)

    p_safe = np.maximum(p, EPSILON)
    log_p = np.log(p_safe)
    logits = log_p - log_p.max(axis=2, keepdims=True)
    return logits


def compute_U(p, D):
    return np.einsum("ijk,ij->ik", p, D)


def compute_metrics_from_logits(logits, D, V, adj_matrix):
    p = logits_to_probs(logits)
    U = compute_U(p, D)

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
        weights = D_demo[:, np.newaxis]
        valid_mask = neighbor_pop > 0
        if np.any(valid_mask):
            neighbor_diffs.append(diff[valid_mask])
            neighbor_weights.append(weights[valid_mask])
    if neighbor_diffs:
        diffs = np.concatenate(neighbor_diffs, axis=0)
        weights = np.concatenate(neighbor_weights, axis=0)
        if weights.shape[1] == 1 and diffs.shape[1] > 1:
            weights = np.repeat(weights, diffs.shape[1], axis=1)
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
        global_weights.append(D_demo[:, np.newaxis])
    if global_diffs:
        diffs = np.concatenate(global_diffs, axis=0)
        weights = np.concatenate(global_weights, axis=0)
        if weights.shape[1] == 1 and diffs.shape[1] > 1:
            weights = np.repeat(weights, diffs.shape[1], axis=1)
        global_diff = np.average(diffs.ravel(), weights=weights.ravel())
    else:
        global_diff = 0.0

    return mae_percent, neighbor_diff, global_diff


def compute_loss_and_gradient(logits_flat, D, V, adj_matrix, neighbor_weight, global_weight):
    num_precincts, num_demos = D.shape
    num_vote_types = len(VOTE_TYPES)

    logits = logits_flat.reshape(num_precincts, num_demos, num_vote_types)
    p = logits_to_probs(logits)

    U = compute_U(p, D)
    data_residual = U - V
    data_loss = np.sum(data_residual ** 2)
    grad_p_data = 2 * np.einsum("ik,ij->ijk", data_residual, D)

    spatial_loss = 0.0
    grad_p_spatial = np.zeros_like(p)
    for demo_idx in range(num_demos):
        p_demo = p[:, demo_idx, :]
        D_demo = D[:, demo_idx]
        neighbor_sum = adj_matrix @ (p_demo * D_demo[:, np.newaxis])
        neighbor_pop = adj_matrix @ D_demo
        neighbor_avg = neighbor_sum / (neighbor_pop[:, np.newaxis] + EPSILON)
        diff = p_demo - neighbor_avg
        spatial_loss += np.sum(diff ** 2)
        grad_diff = 2 * diff
        grad_p_spatial[:, demo_idx, :] = grad_diff
        for vote_idx in range(num_vote_types):
            neighbor_grad = adj_matrix.T @ (grad_diff[:, vote_idx] / (neighbor_pop + EPSILON))
            grad_p_spatial[:, demo_idx, vote_idx] -= neighbor_grad * D_demo

    global_loss = 0.0
    grad_p_global = np.zeros_like(p)
    for demo_idx in range(num_demos):
        p_demo = p[:, demo_idx, :]
        D_demo = D[:, demo_idx]
        total_pop = D_demo.sum()
        if total_pop <= 0:
            continue
        avg = np.average(p_demo, axis=0, weights=D_demo)
        diff = p_demo - avg
        global_loss += np.sum((diff ** 2) * D_demo[:, np.newaxis])
        grad_p_global[:, demo_idx, :] = 2 * diff * D_demo[:, np.newaxis]

    total_loss = data_loss + neighbor_weight * spatial_loss + global_weight * global_loss
    grad_p = grad_p_data + neighbor_weight * grad_p_spatial + global_weight * grad_p_global

    grad_p_times_p = np.sum(grad_p * p, axis=2, keepdims=True)
    grad_logits = grad_p * p - p * grad_p_times_p
    return total_loss, grad_logits.flatten()


def adam_update(gradients, m, v, iteration, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m_new = beta1 * m + (1 - beta1) * gradients
    v_new = beta2 * v + (1 - beta2) * (gradients ** 2)
    m_hat = m_new / (1 - beta1 ** iteration)
    v_hat = v_new / (1 - beta2 ** iteration)
    update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return update, m_new, v_new


def run_stage(
    stage,
    logits_flat,
    D,
    V,
    adj_matrix,
    config,
    thresholds,
):
    max_iterations = config["max_iterations"]
    learning_rate = config["learning_rate"]
    grad_clip_norm = config["grad_clip_norm"]
    backtrack_factor = config["backtrack_factor"]
    max_rejects = config["max_rejects"]
    stage_patience = config["stage_patience"]

    neighbor_weight = config["neighbor_weight"] if stage >= 2 else 0.0
    global_weight = config["global_weight"] if stage >= 3 else 0.0

    m = np.zeros_like(logits_flat)
    v = np.zeros_like(logits_flat)

    best_logits = logits_flat.copy()
    best_metrics = None
    ok_count = 0
    rejects = 0

    for iteration in tqdm(range(1, max_iterations + 1), desc=f"Stage {stage}"):
        loss, grad = compute_loss_and_gradient(
            logits_flat, D, V, adj_matrix, neighbor_weight, global_weight
        )
        grad_norm = np.linalg.norm(grad)
        if grad_norm > grad_clip_norm:
            grad = grad * (grad_clip_norm / (grad_norm + EPSILON))

        update, m, v = adam_update(grad, m, v, iteration, learning_rate)
        candidate = logits_flat - update
        candidate_logits = candidate.reshape((-1,))
        num_precincts, num_demos = D.shape
        candidate_reshaped = candidate_logits.reshape(
            num_precincts, num_demos, len(VOTE_TYPES)
        )
        mae_percent, neighbor_diff, global_diff = compute_metrics_from_logits(
            candidate_reshaped, D, V, adj_matrix
        )

        violates = False
        if stage >= 2 and mae_percent > thresholds["mae"]:
            violates = True
        if stage >= 3 and neighbor_diff > thresholds["neighbor"]:
            violates = True

        if violates:
            rejects += 1
            learning_rate *= backtrack_factor
            if rejects >= max_rejects:
                break
            continue

        logits_flat = candidate_logits

        metrics = (mae_percent, neighbor_diff, global_diff)
        if best_metrics is None:
            best_logits = logits_flat.copy()
            best_metrics = metrics
        else:
            metric_idx = stage - 1
            if metrics[metric_idx] < best_metrics[metric_idx]:
                best_logits = logits_flat.copy()
                best_metrics = metrics

        stage_ok = False
        if stage == 1 and mae_percent <= thresholds["mae"]:
            stage_ok = True
        if stage == 2 and mae_percent <= thresholds["mae"] and neighbor_diff <= thresholds["neighbor"]:
            stage_ok = True
        if stage == 3 and (
            mae_percent <= thresholds["mae"]
            and neighbor_diff <= thresholds["neighbor"]
            and global_diff <= thresholds["global"]
        ):
            stage_ok = True

        ok_count = ok_count + 1 if stage_ok else 0
        if ok_count >= stage_patience:
            return logits_flat, metrics, True

    return best_logits, best_metrics, False


def optimize_stage_cost(D, V, adj_matrix, rng, config, thresholds):
    num_precincts, num_demos = D.shape
    num_vote_types = len(VOTE_TYPES)
    logits = initialize_logits(num_precincts, num_demos, num_vote_types, rng)
    logits_flat = logits.flatten()

    stage_metrics = {}
    for stage in (1, 2, 3):
        logits_flat, metrics, converged = run_stage(
            stage,
            logits_flat,
            D,
            V,
            adj_matrix,
            config,
            thresholds,
        )
        stage_metrics[stage] = {
            "metrics": metrics,
            "converged": converged,
        }
        if not converged:
            break

    logits = logits_flat.reshape(num_precincts, num_demos, num_vote_types)
    p_final = logits_to_probs(logits)
    return p_final, stage_metrics


CONFIGS = [
    {
        "name": "base",
        "learning_rate": 0.01,
        "max_iterations": 200,
        "neighbor_weight": 0.1,
        "global_weight": 0.05,
        "grad_clip_norm": 100.0,
        "backtrack_factor": 0.5,
        "max_rejects": 25,
        "stage_patience": 5,
    },
    {
        "name": "aggressive",
        "learning_rate": 0.02,
        "max_iterations": 250,
        "neighbor_weight": 0.15,
        "global_weight": 0.08,
        "grad_clip_norm": 100.0,
        "backtrack_factor": 0.5,
        "max_rejects": 25,
        "stage_patience": 5,
    },
    {
        "name": "conservative",
        "learning_rate": 0.005,
        "max_iterations": 300,
        "neighbor_weight": 0.08,
        "global_weight": 0.04,
        "grad_clip_norm": 80.0,
        "backtrack_factor": 0.6,
        "max_rejects": 30,
        "stage_patience": 8,
    },
    {
        "name": "neighbor_heavy",
        "learning_rate": 0.01,
        "max_iterations": 250,
        "neighbor_weight": 0.2,
        "global_weight": 0.03,
        "grad_clip_norm": 100.0,
        "backtrack_factor": 0.5,
        "max_rejects": 25,
        "stage_patience": 6,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Staged optimization in logit space with guardrails")
    parser.add_argument("main_data_csv", help="Path to main data CSV")
    parser.add_argument("graph_file", help="Path to graph gpickle (GEOID-based)")
    parser.add_argument("--output-dir", default="output", help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", help="Run a single config by name")
    parser.add_argument("--run-all", action="store_true", help="Run all configs")
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
    if not isinstance(adj_matrix, csr_matrix):
        adj_matrix = csr_matrix(adj_matrix)

    V, D = preprocess_data_mle(df)
    thresholds = {"mae": 0.05, "neighbor": 0.05, "global": 0.05}

    if args.config:
        configs = [cfg for cfg in CONFIGS if cfg["name"] == args.config]
        if not configs:
            raise SystemExit(f"Unknown config '{args.config}'. Options: {[c['name'] for c in CONFIGS]}")
    elif args.run_all or args.config is None:
        configs = CONFIGS
    else:
        configs = [CONFIGS[0]]

    os.makedirs(args.output_dir, exist_ok=True)
    for cfg in configs:
        print("\n" + "=" * 70)
        print(f"Running config: {cfg['name']}")
        p_final, stage_metrics = optimize_stage_cost(D, V, adj_matrix, rng, cfg, thresholds)

        for stage in sorted(stage_metrics):
            metrics = stage_metrics[stage]["metrics"]
            converged = stage_metrics[stage]["converged"]
            if metrics is None:
                print(f"Stage {stage}: no metrics (terminated early)")
                continue
            mae, neighbor, global_diff = metrics
            print(
                f"Stage {stage} converged={converged} "
                f"mae={mae:.4f} neighbor={neighbor:.4f} global={global_diff:.4f}"
            )

        result_df = df.copy()
        for demo_idx, demo in enumerate(DEMOGRAPHICS):
            for vote_idx, vote_type in enumerate(VOTE_TYPES):
                result_df[f"{vote_type}_{demo}_prob"] = p_final[:, demo_idx, vote_idx]

        out_path = os.path.join(args.output_dir, f"stage_cost_{cfg['name']}.csv")
        result_df.to_csv(out_path, index=True)
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
