"""
MLE using L2 optimization in logit space with analytical gradients.
Minimizes: ||U - V||^2 + spatial_weight * spatial_smoothing_term
Uses analytical gradient computation for efficiency.
"""
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix
from tqdm import tqdm
import argparse
import pickle

# Constants
DEMOGRAPHICS = ['Wht', 'His', 'Blk', 'Asn', 'Oth']
VOTE_TYPES = ['D', 'R', 'O', 'N']
EPSILON = 1e-10

def softmax(x, axis=-1):
    """Softmax function for converting logits to probabilities."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(np.clip(x_shifted, -500, 500))
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + EPSILON)

def logits_to_probs(logits):
    """Convert logits to probability vectors (each row sums to 1)."""
    return softmax(logits, axis=-1)

def prepare_data_step(main_data_csv, graph_file):
    """Prepare data - simplified version from main_mle.py"""
    print(f"Loading main data from {main_data_csv}...")
    df = pd.read_csv(main_data_csv, dtype={'AFFGEOID': str})
    df.set_index('AFFGEOID', inplace=True)
    
    # Filter to Queens, NY (FIPS 36081)
    df = df[df.index.str.contains('36081', na=False)]
    print(f"Filtered to Queens, NY (county FIPS 36081): {len(df)} rows.")
    
    print(f"Loading graph from {graph_file}...")
    with open(graph_file, 'rb') as f:
        G = pickle.load(f)
    
    # Calculate vote shares
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
    df['votes_N'] = df['cvap_total'] - total_votes
    df['votes_N'] = df['votes_N'].clip(lower=0)
    
    return df, G

def preprocess_data_mle(df):
    """Preprocess data: scale for consistent population."""
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
    """Initialize logits from flat Dirichlet probabilities."""
    total_samples = num_precincts * num_demos
    gammas = rng.gamma(1.0, 1.0, size=(total_samples, num_vote_types))
    p_flat = gammas / (gammas.sum(axis=1, keepdims=True) + EPSILON)
    p = p_flat.reshape(num_precincts, num_demos, num_vote_types)
    
    # Convert to logits (inverse softmax)
    p_safe = np.maximum(p, EPSILON)
    log_p = np.log(p_safe)
    logits = log_p - log_p.max(axis=2, keepdims=True)
    
    return logits

def compute_U_from_logits(logits, D):
    """Compute U from logits."""
    p = logits_to_probs(logits)
    U = np.einsum('ijk,ij->ik', p, D)
    return U

def compute_spatial_loss(logits, D, adj_matrix):
    """Compute spatial smoothing loss: sum of squared differences between neighbors."""
    p = logits_to_probs(logits)
    num_precincts, num_demos, num_vote_types = p.shape
    
    spatial_loss = 0.0
    for demo_idx in range(num_demos):
        p_demo = p[:, demo_idx, :]  # (num_precincts, num_vote_types)
        D_demo = D[:, demo_idx]  # (num_precincts,)
        
        # Compute weighted average of neighbors for each precinct
        neighbor_sum = adj_matrix @ (p_demo * D_demo[:, np.newaxis])
        neighbor_pop = adj_matrix @ D_demo
        neighbor_avg = neighbor_sum / (neighbor_pop[:, np.newaxis] + EPSILON)
        
        # L2 loss: squared difference from neighbors
        diff = p_demo - neighbor_avg
        spatial_loss += np.sum(diff ** 2)
    
    return spatial_loss

def compute_loss_and_gradient(logits_flat, D, V, adj_matrix, spatial_weight):
    """
    Compute loss and gradient analytically.
    
    Loss: ||U - V||^2 + spatial_weight * spatial_smoothing
    
    Gradient uses chain rule through softmax.
    """
    num_precincts, num_demos = D.shape
    num_vote_types = len(VOTE_TYPES)
    
    logits = logits_flat.reshape(num_precincts, num_demos, num_vote_types)
    p = logits_to_probs(logits)
    
    # Compute U
    U = np.einsum('ijk,ij->ik', p, D)  # (num_precincts, num_vote_types)
    
    # Data fitting loss: ||U - V||^2
    data_residual = U - V  # (num_precincts, num_vote_types)
    data_loss = np.sum(data_residual ** 2)
    
    # Gradient of data loss w.r.t. p
    grad_p_data = 2 * np.einsum('ik,ij->ijk', data_residual, D)  # (num_precincts, num_demos, num_vote_types)
    
    # Spatial smoothing loss and gradient
    spatial_loss = 0.0
    grad_p_spatial = np.zeros_like(p)
    
    for demo_idx in range(num_demos):
        p_demo = p[:, demo_idx, :]  # (num_precincts, num_vote_types)
        D_demo = D[:, demo_idx]  # (num_precincts,)
        
        # Neighbor averages: for each precinct i, average over neighbors j
        # neighbor_avg[i] = sum_j(adj[i,j] * p[j] * D[j]) / sum_j(adj[i,j] * D[j])
        neighbor_sum = adj_matrix @ (p_demo * D_demo[:, np.newaxis])  # (num_precincts, num_vote_types)
        neighbor_pop = adj_matrix @ D_demo  # (num_precincts,)
        neighbor_avg = neighbor_sum / (neighbor_pop[:, np.newaxis] + EPSILON)  # (num_precincts, num_vote_types)
        
        # Spatial loss: sum over precincts of (p[i] - neighbor_avg[i])^2
        diff = p_demo - neighbor_avg  # (num_precincts, num_vote_types)
        spatial_loss += np.sum(diff ** 2)
        
        # Gradient: d/dp sum_i (p[i] - neighbor_avg[i])^2
        # = 2 * sum_i (p[i] - neighbor_avg[i]) * (I - d(neighbor_avg[i])/dp)
        # For each precinct i: grad comes from p[i] itself and from neighbors j that include i
        grad_diff = 2 * diff  # (num_precincts, num_vote_types)
        
        # Direct term: gradient from p[i] itself
        grad_p_spatial[:, demo_idx, :] = grad_diff
        
        # Backprop through neighbor_avg: if j is a neighbor of i, then p[j] affects neighbor_avg[i]
        # d(neighbor_avg[i])/dp[j] = adj[i,j] * D[j] / sum_k(adj[i,k] * D[k])
        # So gradient at p[j] gets contribution from all neighbors i where adj[i,j] = 1
        for vote_idx in range(num_vote_types):
            # For each neighbor i of j, subtract contribution
            # adj_matrix.T gives us neighbors j of each precinct i
            neighbor_grad = adj_matrix.T @ (grad_diff[:, vote_idx] / (neighbor_pop + EPSILON))
            grad_p_spatial[:, demo_idx, vote_idx] -= neighbor_grad * D_demo
    
    total_loss = data_loss + spatial_weight * spatial_loss
    grad_p = grad_p_data + spatial_weight * grad_p_spatial
    
    # Convert gradient from p to logits using softmax Jacobian
    # If p = softmax(logits), then for each row: dp/dlogits = diag(p) - p @ p^T
    # grad_logits = p * grad_p - p * sum(p * grad_p)
    grad_p_times_p = np.sum(grad_p * p, axis=2, keepdims=True)  # (num_precincts, num_demos, 1)
    grad_logits = grad_p * p - p * grad_p_times_p
    
    grad_logits_flat = grad_logits.flatten()
    
    return total_loss, grad_logits_flat

def adam_update(gradients, m, v, iteration, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """Adam optimizer update step."""
    m_new = beta1 * m + (1 - beta1) * gradients
    v_new = beta2 * v + (1 - beta2) * (gradients ** 2)
    m_hat = m_new / (1 - beta1 ** iteration)
    v_hat = v_new / (1 - beta2 ** iteration)
    update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return update, m_new, v_new

def optimize_l2_logit(D, V, adj_matrix, spatial_weight=0.1, max_iterations=100, learning_rate=0.01, rng=None, use_adam=True):
    """Optimize using gradient descent with L2 loss in logit space."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    num_precincts, num_demos = D.shape
    num_vote_types = len(VOTE_TYPES)
    
    # Initialize
    logits = initialize_logits(num_precincts, num_demos, num_vote_types, rng)
    logits_flat = logits.flatten()
    
    # Initialize Adam state
    m = np.zeros_like(logits_flat)
    v = np.zeros_like(logits_flat)
    
    history = []
    
    print(f"\nStarting L2 optimization in logit space...")
    print(f"Spatial weight: {spatial_weight}, Learning rate: {learning_rate}, Optimizer: {'Adam' if use_adam else 'SGD'}")
    
    for iteration in tqdm(range(1, max_iterations + 1), desc="Optimization"):
        # Compute loss and gradient
        loss, grad = compute_loss_and_gradient(logits_flat, D, V, adj_matrix, spatial_weight)
        
        # Gradient clipping
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 100.0:
            grad = grad * (100.0 / (grad_norm + EPSILON))
        
        # Update step
        if use_adam:
            update, m, v = adam_update(grad, m, v, iteration, learning_rate)
            logits_flat = logits_flat - update
        else:
            logits_flat = logits_flat - learning_rate * grad
        
        # Store history
        if iteration % 10 == 0 or iteration == 1:
            logits = logits_flat.reshape(num_precincts, num_demos, num_vote_types)
            U = compute_U_from_logits(logits, D)
            
            U_total = U.sum(axis=0)
            V_total = V.sum(axis=0)
            U_props = U_total / (U_total.sum() + EPSILON)
            V_props = V_total / (V_total.sum() + EPSILON)
            
            mae = np.mean(np.abs(U_props - V_props))
            
            history.append({
                'iteration': iteration,
                'loss': loss,
                'mae': mae,
                'U_props': U_props.copy(),
                'V_props': V_props.copy()
            })
            
            print(f"  Iteration {iteration}: loss={loss:.2f}, MAE={mae:.4f}")
            print(f"    U: D={U_props[0]:.4f}, R={U_props[1]:.4f}, O={U_props[2]:.4f}, N={U_props[3]:.4f}")
            print(f"    V: D={V_props[0]:.4f}, R={V_props[1]:.4f}, O={V_props[2]:.4f}, N={V_props[3]:.4f}")
    
    logits = logits_flat.reshape(num_precincts, num_demos, num_vote_types)
    p = logits_to_probs(logits)
    
    return p, logits, history

def main():
    parser = argparse.ArgumentParser(description="L2 optimization in logit space for MLE")
    parser.add_argument("main_data_csv", help="Path to main data CSV")
    parser.add_argument("graph_file", help="Path to graph gpickle file")
    parser.add_argument("--spatial-weight", type=float, default=0.1, help="Weight for spatial smoothing")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--output", default="mle_l2_logit_results.csv", help="Output CSV file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    rng = np.random.default_rng(args.seed)
    
    # Prepare data
    df, G = prepare_data_step(args.main_data_csv, args.graph_file)
    df.sort_index(inplace=True)
    
    graph_nodes = {f"1500000US{node}" for node in G.nodes()}
    df_nodes = set(df.index)
    node_list = sorted(list(graph_nodes.intersection(df_nodes)))
    df = df.loc[node_list]
    
    adj_matrix = nx.to_scipy_sparse_array(
        G, nodelist=[geoid.replace("1500000US", "") for geoid in node_list], format="csr"
    )
    
    V, D, s = preprocess_data_mle(df)
    
    print(f"\nData shape: {len(df)} precincts, {len(DEMOGRAPHICS)} demographics, {len(VOTE_TYPES)} vote types")
    
    # Optimize
    p_final, logits_final, history = optimize_l2_logit(
        D, V, adj_matrix,
        spatial_weight=args.spatial_weight,
        max_iterations=args.iterations,
        learning_rate=args.learning_rate,
        rng=rng
    )
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    result_df = df.copy()
    
    for demo_idx, demo in enumerate(DEMOGRAPHICS):
        for vote_idx, vote_type in enumerate(VOTE_TYPES):
            result_df[f'{vote_type}_{demo}_prob'] = p_final[:, demo_idx, vote_idx]
    
    result_df.to_csv(args.output, index=True)
    print(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()
