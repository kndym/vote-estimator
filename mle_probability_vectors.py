"""
Maximum Likelihood Estimation for probability vectors using gradient descent.

This script optimizes probability vectors p[precinct, demo, vote_type] using gradient
descent on a combined likelihood function that includes:
1. Dirichlet-Multinomial likelihood: log(DirMult(U | V))
2. Dirichlet priors: sum_over_demos(sum_over_precincts(log(Dir(p_vector_demo | N_demo))))

Where:
- U[precinct, vote_type] = sum_over_demos(p[precinct, demo, vote_type] * D[precinct, demo])
- N_demo[precinct, vote_type] = sum_over_neighbors(p[neighbor, demo, vote_type] * D[neighbor, demo])
- V: scaled vote sums
- D: demo sums (original + 1)
- p: probability vectors (sum to 1 over vote_types)

Input:
    - Prepared data feather file (from prepare_data.py)
    - Graph gpickle file (for neighborhood definition)
Output:
    - CSV file with probability estimates
"""
import pandas as pd
import numpy as np
import argparse
import pickle
import networkx as nx
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.special import gammaln, digamma
import os
import sys

# =============================================================================
# CONSTANTS
# =============================================================================

EPSILON = 1e-12
DEMOGRAPHICS = ['Wht', 'His', 'Blk', 'Asn', 'Oth']
VOTE_TYPES = ['D', 'R', 'O', 'N']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_affgeoid(geoid):
    """Format GEOID to AFFGEOID format."""
    return f"1500000US{geoid}"

def preprocess_data(df):
    """
    Preprocess data: scale vote sums and add 1 to demo sums.
    
    - V (vote sums): Scale by (1 + m/s) per precinct, where m=5 (num demos), s=cvap_total
      For zero votes, set equal to m.
    - D (demo sums): Add 1 to each demographic count per precinct
    
    Args:
        df: DataFrame with AFFGEOID index
        
    Returns:
        V: Array of shape (num_precincts, num_vote_types) with scaled vote sums
        D: Array of shape (num_precincts, num_demos) with demo sums (original + 1)
        s: Array of shape (num_precincts,) with total populations
    """
    print("Preprocessing data: scaling vote sums and adding 1 to demo sums...")
    
    m = len(DEMOGRAPHICS)  # num_demo_types = 5
    
    # Extract vote counts (as floats)
    vote_cols = [f'votes_{vtype}' for vtype in VOTE_TYPES]
    V = np.array([df[col].values.astype(float) for col in vote_cols]).T  # (num_precincts, num_vote_types)
    
    # Extract demographic counts (as floats)
    demo_cols = [f'cvap_{demo}' for demo in DEMOGRAPHICS]
    D = np.array([df[col].values.astype(float) for col in demo_cols]).T  # (num_precincts, num_demos)
    
    # Get total population s
    if 'cvap_total' in df.columns:
        s = df['cvap_total'].values.astype(float)
    else:
        s = D.sum(axis=1)
    
    # Scale V by (1 + m/s) per precinct
    scaling_factors = 1.0 + m / (s + EPSILON)  # (num_precincts,)
    V_scaled = V * scaling_factors[:, np.newaxis]  # (num_precincts, num_vote_types)
    
    # For zero votes, set equal to m
    zero_mask = V == 0
    V_scaled[zero_mask] = float(m)
    
    # Add 1 to each demo sum
    D_scaled = D + 1.0
    
    print(f"  Vote sums scaled. Mean scaling factor: {scaling_factors.mean():.4f}")
    print(f"  Demo sums: added 1 to each count")
    
    return V_scaled, D_scaled, s

def initialize_probabilities(num_precincts, num_demos, num_vote_types, rng):
    """
    Initialize probability vectors using flat Dirichlet distribution.
    
    Samples from Dirichlet(1, 1, ..., 1) for each (precinct, demo) pair,
    which gives a uniform distribution over the probability simplex.
    Then converts to softmax parameterization.
    
    Args:
        num_precincts: Number of precincts
        num_demos: Number of demographics
        num_vote_types: Number of vote types
        rng: Random number generator
        
    Returns:
        theta: Unconstrained parameters (num_precincts, num_demos, num_vote_types)
        p: Probability vectors (num_precincts, num_demos, num_vote_types), each row sums to 1
    """
    # Sample from flat Dirichlet using gamma method (more efficient)
    # Flat Dirichlet means all parameters = 1 (uniform prior)
    # Sample gamma(1, 1) for each component, then normalize
    total_samples = num_precincts * num_demos
    gammas = rng.gamma(1.0, 1.0, size=(total_samples, num_vote_types))
    # Normalize each row to get Dirichlet samples
    p_flat = gammas / (gammas.sum(axis=1, keepdims=True) + EPSILON)
    # Reshape to (num_precincts, num_demos, num_vote_types)
    p = p_flat.reshape(num_precincts, num_demos, num_vote_types)
    
    # Convert to softmax parameterization: p = softmax(theta)
    # For softmax: if p = softmax(theta), then theta = log(p) + constant (constant cancels in softmax)
    # We use: theta = log(p + epsilon) and subtract the max for numerical stability
    p_safe = np.maximum(p, EPSILON)
    log_p = np.log(p_safe)
    # Subtract max for numerical stability (softmax is translation invariant)
    theta = log_p - log_p.max(axis=2, keepdims=True)
    
    return theta, p

def softmax(x, axis=-1):
    """
    Compute softmax function.
    
    Args:
        x: Input array
        axis: Axis along which to compute softmax (default: last axis)
        
    Returns:
        Softmax probabilities (same shape as x)
    """
    # Subtract max for numerical stability
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + EPSILON)

def compute_U(p, D):
    """
    Compute U[precinct, vote_type] = sum_over_demos(p[precinct, demo, vote_type] * D[precinct, demo])
    
    Args:
        p: Probability vectors (num_precincts, num_demos, num_vote_types)
        D: Demo sums (num_precincts, num_demos)
        
    Returns:
        U: Predicted vote sums (num_precincts, num_vote_types)
    """
    # p: (num_precincts, num_demos, num_vote_types)
    # D: (num_precincts, num_demos)
    # We want to compute: sum_over_demos(p[i, j, k] * D[i, j]) for each (i, k)
    
    # Broadcast D: (num_precincts, num_demos, 1)
    D_expanded = D[:, :, np.newaxis]  # (num_precincts, num_demos, 1)
    
    # Element-wise multiply: (num_precincts, num_demos, num_vote_types)
    p_times_D = p * D_expanded
    
    # Sum over demos (axis=1): (num_precincts, num_vote_types)
    U = p_times_D.sum(axis=1)
    
    return U

def compute_N_demo(p, D, adj_matrix):
    """
    Compute N_demo[precinct, vote_type] = sum_over_neighbors(p[neighbor, demo, vote_type] * D[neighbor, demo])
    
    Args:
        p: Probability vectors (num_precincts, num_demos, num_vote_types)
        D: Demo sums (num_precincts, num_demos)
        adj_matrix: Sparse adjacency matrix (num_precincts, num_precincts)
        
    Returns:
        N_demo: Neighborhood aggregated values (num_precincts, num_demos, num_vote_types)
    """
    num_precincts, num_demos, num_vote_types = p.shape
    N_demo = np.zeros((num_precincts, num_demos, num_vote_types), dtype=float)
    
    # For each demographic
    for demo_idx in range(num_demos):
        # p[:, demo_idx, :]: (num_precincts, num_vote_types)
        # D[:, demo_idx]: (num_precincts,)
        # We want: adj_matrix @ (p[:, demo_idx, :] * D[:, demo_idx:demo_idx+1])
        
        p_demo = p[:, demo_idx, :]  # (num_precincts, num_vote_types)
        D_demo = D[:, demo_idx]  # (num_precincts,)
        
        # Multiply: (num_precincts, num_vote_types)
        p_times_D_demo = p_demo * D_demo[:, np.newaxis]
        
        # Aggregate over neighbors using sparse matrix multiplication
        # adj_matrix @ p_times_D_demo: (num_precincts, num_vote_types)
        N_demo[:, demo_idx, :] = adj_matrix @ p_times_D_demo
    
    return N_demo

# =============================================================================
# LIKELIHOOD COMPUTATION
# =============================================================================

def compute_dirichlet_multinomial_loglik(U, V):
    """
    Compute log-likelihood of Dirichlet-Multinomial: log(DirMult(U | V))
    
    The Dirichlet-Multinomial distribution:
    P(X | α, N) = Γ(α₀) / Γ(α₀ + N) * ∏ₖ Γ(αₖ + Xₖ) / Γ(αₖ)
    
    where α₀ = Σₖ αₖ, X is the count vector, N is the total.
    
    Here, V is the concentration parameter (alpha) and U is the observed counts.
    
    Args:
        U: Predicted vote sums (num_precincts, num_vote_types)
        V: Scaled vote sums (concentration parameters) (num_precincts, num_vote_types)
        
    Returns:
        loglik: Scalar log-likelihood
    """
    # Ensure V is positive (concentration parameters must be > 0)
    V = np.maximum(V, EPSILON)
    
    # Compute totals
    V_sum = V.sum(axis=1)  # (num_precincts,)
    U_sum = U.sum(axis=1)  # (num_precincts,)
    
    # Mask for valid entries (non-zero totals)
    valid_mask = (V_sum > 0) & (U_sum > 0)
    
    if not np.any(valid_mask):
        return 0.0
    
    # Term 1: log_gamma(sum(V)) - log_gamma(sum(V) + sum(U))
    term1 = gammaln(V_sum) - gammaln(V_sum + U_sum)
    
    # Term 2: sum(log_gamma(V + U)) - sum(log_gamma(V))
    term2 = gammaln(V + U).sum(axis=1) - gammaln(V).sum(axis=1)
    
    # Combine terms
    loglik_per_precinct = term1 + term2
    
    # Sum only valid entries
    loglik = loglik_per_precinct[valid_mask].sum()
    
    return loglik

def compute_dirichlet_loglik(p_vector_demo, N_demo):
    """
    Compute log-likelihood of Dirichlet: log(Dir(p_vector_demo | N_demo))
    
    The Dirichlet distribution:
    log(P(p | α)) = log(Γ(α₀)) - Σₖ log(Γ(αₖ)) + Σₖ (αₖ - 1) * log(pₖ)
    
    where α₀ = Σₖ αₖ, α = N_demo (concentration parameters), p = p_vector_demo
    
    Args:
        p_vector_demo: Probability vectors for one demographic (num_precincts, num_vote_types)
        N_demo: Concentration parameters (num_precincts, num_vote_types)
        
    Returns:
        loglik: Scalar log-likelihood
    """
    # Ensure N_demo is positive
    N_demo = np.maximum(N_demo, EPSILON)
    
    # Ensure p is positive and sums to 1 (should already be true, but enforce)
    p_vector_demo = np.maximum(p_vector_demo, EPSILON)
    p_vector_demo = p_vector_demo / (p_vector_demo.sum(axis=1, keepdims=True) + EPSILON)
    
    # Compute totals
    N_demo_sum = N_demo.sum(axis=1)  # (num_precincts,)
    
    # Mask for valid entries
    valid_mask = N_demo_sum > 0
    
    if not np.any(valid_mask):
        return 0.0
    
    # Term 1: log_gamma(sum(N_demo))
    term1 = gammaln(N_demo_sum)
    
    # Term 2: - sum(log_gamma(N_demo))
    term2 = -gammaln(N_demo).sum(axis=1)
    
    # Term 3: sum((N_demo - 1) * log(p_vector_demo))
    term3 = ((N_demo - 1.0) * np.log(p_vector_demo + EPSILON)).sum(axis=1)
    
    # Combine terms
    loglik_per_precinct = term1 + term2 + term3
    
    # Sum only valid entries
    loglik = loglik_per_precinct[valid_mask].sum()
    
    return loglik

def compute_total_likelihood(p, D, V, adj_matrix):
    """
    Compute total log-likelihood: log(DirMult(U | V)) + sum_over_demos(log(Dir(p_vector_demo | N_demo)))
    
    Args:
        p: Probability vectors (num_precincts, num_demos, num_vote_types)
        D: Demo sums (num_precincts, num_demos)
        V: Scaled vote sums (num_precincts, num_vote_types)
        adj_matrix: Sparse adjacency matrix (num_precincts, num_precincts)
        
    Returns:
        total_loglik: Scalar total log-likelihood
    """
    # Compute U
    U = compute_U(p, D)
    
    # Compute DirMult log-likelihood
    dir_mult_loglik = compute_dirichlet_multinomial_loglik(U, V)
    
    # Compute N_demo
    N_demo = compute_N_demo(p, D, adj_matrix)
    
    # Compute Dirichlet log-likelihood for each demographic
    dir_loglik = 0.0
    num_demos = p.shape[1]
    for demo_idx in range(num_demos):
        p_vector_demo = p[:, demo_idx, :]  # (num_precincts, num_vote_types)
        N_demo_demo = N_demo[:, demo_idx, :]  # (num_precincts, num_vote_types)
        dir_loglik += compute_dirichlet_loglik(p_vector_demo, N_demo_demo)
    
    total_loglik = dir_mult_loglik + dir_loglik
    
    return total_loglik

# =============================================================================
# GRADIENT COMPUTATION
# =============================================================================

def compute_gradients(p, theta, D, V, adj_matrix):
    """
    Compute gradients of log-likelihood w.r.t. theta parameters.
    
    Uses chain rule: dL/dtheta = dL/dp * dp/dtheta
    where dp/dtheta comes from softmax Jacobian.
    
    Args:
        p: Probability vectors (num_precincts, num_demos, num_vote_types)
        theta: Unconstrained parameters (num_precincts, num_demos, num_vote_types)
        D: Demo sums (num_precincts, num_demos)
        V: Scaled vote sums (num_precincts, num_vote_types)
        adj_matrix: Sparse adjacency matrix (num_precincts, num_precincts)
        
    Returns:
        gradients: Gradients w.r.t. theta (num_precincts, num_demos, num_vote_types)
    """
    num_precincts, num_demos, num_vote_types = p.shape
    
    # Initialize gradient arrays
    grad_p = np.zeros_like(p)
    
    # Compute U
    U = compute_U(p, D)
    
    # ===== Gradients from DirMult term: d(log(DirMult(U|V)))/dp =====
    V_safe = np.maximum(V, EPSILON)
    V_sum = V_safe.sum(axis=1, keepdims=True)  # (num_precincts, 1)
    U_sum = U.sum(axis=1, keepdims=True)  # (num_precincts, 1)
    
    # Gradient of DirMult loglik w.r.t. U
    # d(log(DirMult))/dU = digamma(V + U) - digamma(V_sum + U_sum)
    grad_U_dir_mult = digamma(V_safe + U) - digamma(V_sum + U_sum)  # (num_precincts, num_vote_types)
    
    # Propagate to p: dU/dp = D (since U[i, vote_type] = sum_demo(p[i, demo, vote_type] * D[i, demo]))
    # So dU[i, vote_type]/dp[i, demo, vote_type] = D[i, demo]
    for demo_idx in range(num_demos):
        for vote_idx in range(num_vote_types):
            grad_p[:, demo_idx, vote_idx] += grad_U_dir_mult[:, vote_idx] * D[:, demo_idx]
    
    # ===== Gradients from Dirichlet term: d(log(Dir(p_vector_demo | N_demo)))/dp =====
    N_demo = compute_N_demo(p, D, adj_matrix)
    
    for demo_idx in range(num_demos):
        p_vector_demo = p[:, demo_idx, :]  # (num_precincts, num_vote_types)
        N_demo_demo = N_demo[:, demo_idx, :]  # (num_precincts, num_vote_types)
        N_demo_demo_safe = np.maximum(N_demo_demo, EPSILON)
        N_demo_sum = N_demo_demo_safe.sum(axis=1, keepdims=True)  # (num_precincts, 1)
        
        # Direct gradient w.r.t. p_vector_demo (treating N_demo as constant)
        # d(log(Dir))/dp = (N_demo - 1) / p
        grad_p_dir_direct = (N_demo_demo_safe - 1.0) / (p_vector_demo + EPSILON)  # (num_precincts, num_vote_types)
        grad_p[:, demo_idx, :] += grad_p_dir_direct
        
        # Gradient w.r.t. N_demo (concentration parameters)
        # d(log(Dir))/dN_demo = digamma(N_demo_sum) - digamma(N_demo) + log(p_vector_demo)
        grad_N_demo = (
            digamma(N_demo_sum) - digamma(N_demo_demo_safe) + np.log(p_vector_demo + EPSILON)
        )  # (num_precincts, num_vote_types)
        
        # Propagate through N_demo: dN_demo/dp
        # N_demo[i, demo_idx, vote_idx] = sum_j(adj_matrix[i, j] * p[j, demo_idx, vote_idx] * D[j, demo_idx])
        # So dN_demo[i, demo_idx, vote_idx]/dp[k, demo_idx, vote_idx] = adj_matrix[i, k] * D[k, demo_idx]
        # Backpropagation: grad_p[k, demo_idx, vote_idx] += sum_i(grad_N_demo[i, demo_idx, vote_idx] * adj_matrix[i, k] * D[k, demo_idx])
        # = D[k, demo_idx] * (adj_matrix.T @ grad_N_demo[:, demo_idx, vote_idx])[k]
        for vote_idx in range(num_vote_types):
            # Backpropagate through adj_matrix
            # adj_matrix.T gives reverse direction (from target to source)
            grad_N_demo_backprop = adj_matrix.T @ grad_N_demo[:, vote_idx]  # (num_precincts,)
            grad_p[:, demo_idx, vote_idx] += grad_N_demo_backprop * D[:, demo_idx]
    
    # ===== Convert grad_p to grad_theta using softmax Jacobian =====
    # For softmax: p_i = exp(theta_i) / sum(exp(theta_j))
    # The Jacobian is: dp_i/dtheta_j = p_i * (delta_ij - p_j)
    # So grad_theta_i = sum_j(grad_p_j * dp_j/dtheta_i) = sum_j(grad_p_j * p_j * (delta_ji - p_i))
    #                  = grad_p_i * p_i - p_i * sum_j(grad_p_j * p_j)
    
    # Compute sum_j(grad_p_j * p_j) for each (precinct, demo)
    grad_p_times_p = (grad_p * p).sum(axis=2, keepdims=True)  # (num_precincts, num_demos, 1)
    
    # Compute grad_theta
    grad_theta = grad_p * p - p * grad_p_times_p  # (num_precincts, num_demos, num_vote_types)
    
    return grad_theta

# =============================================================================
# GRADIENT DESCENT OPTIMIZATION
# =============================================================================

def adam_update(gradients, m, v, iteration, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Adam optimizer update step.
    
    Args:
        gradients: Array of gradients
        m: First moment estimate
        v: Second moment estimate
        iteration: Current iteration number (1-indexed)
        learning_rate: Learning rate
        beta1, beta2: Adam hyperparameters
        epsilon: Small constant for numerical stability
        
    Returns:
        update: Parameter update
        m_new: Updated first moment
        v_new: Updated second moment
    """
    # Update biased first moment estimate
    m_new = beta1 * m + (1 - beta1) * gradients
    
    # Update biased second moment estimate
    v_new = beta2 * v + (1 - beta2) * (gradients ** 2)
    
    # Compute bias-corrected estimates
    m_hat = m_new / (1 - beta1 ** iteration)
    v_hat = v_new / (1 - beta2 ** iteration)
    
    # Compute update
    update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    return update, m_new, v_new

def run_optimization(p, theta, D, V, adj_matrix, max_iterations, learning_rate, rng, grad_clip_norm=5.0):
    """
    Run gradient descent optimization using Adam.
    
    Args:
        p: Initial probability vectors (num_precincts, num_demos, num_vote_types)
        theta: Initial unconstrained parameters (num_precincts, num_demos, num_vote_types)
        D: Demo sums (num_precincts, num_demos)
        V: Scaled vote sums (num_precincts, num_vote_types)
        adj_matrix: Sparse adjacency matrix (num_precincts, num_precincts)
        max_iterations: Maximum number of iterations
        learning_rate: Learning rate for Adam optimizer
        rng: Random number generator
        grad_clip_norm: Gradient norm clip threshold
        
    Returns:
        p: Final probability vectors
        history: List of log-likelihood values
    """
    print(f"Starting optimization: {max_iterations} iterations...")
    
    # Initialize Adam state
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    
    history = []
    
    for iteration in tqdm(range(1, max_iterations + 1), desc="Optimization"):
        # Compute likelihood
        loglik = compute_total_likelihood(p, D, V, adj_matrix)
        history.append({"iteration": iteration, "loglik": loglik})
        
        # Compute gradients
        grad_theta = compute_gradients(p, theta, D, V, adj_matrix)
        
        # Gradient clipping
        grad_norm = np.linalg.norm(grad_theta)
        if grad_norm > grad_clip_norm:
            grad_theta = grad_theta * (grad_clip_norm / (grad_norm + EPSILON))
        
        # Update using Adam
        update, m, v = adam_update(grad_theta, m, v, iteration, learning_rate)
        theta = theta + update
        
        # Convert back to probabilities
        p = softmax(theta)
        
        # Print progress occasionally
        if iteration % 10 == 0 or iteration == 1:
            print(f"  Iteration {iteration}: log-likelihood = {loglik:.2f}, grad_norm = {grad_norm:.4f}")
    
    return p, history

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MLE estimation of probability vectors using gradient descent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First, generate the prepared data and graph files (if not already created):
  python prepare_data.py <input_data> <output_prepared_data.feather>
  python graph.py <shapefile> <output_graph.gpickle>
  
  # Then run this script:
  python mle_probability_vectors.py prepared_data.feather blockgroups_graph.gpickle --output results.csv
        """
    )
    parser.add_argument(
        "prepared_data", 
        help="Path to prepared data feather file (from prepare_data.py)."
    )
    parser.add_argument(
        "graph_file", 
        help="Path to graph gpickle file (from graph.py)."
    )
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="Number of gradient descent iterations (default: 100)."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3,
        help="Learning rate for Adam optimizer (default: 1e-3)."
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=5.0,
        help="Gradient norm clip threshold for stability (default: 5.0)."
    )
    parser.add_argument(
        "--output", default="mle_probability_vectors.csv",
        help="Output CSV file path (default: mle_probability_vectors.csv)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)."
    )
    args = parser.parse_args()
    
    # Print usage instructions
    print("=" * 70)
    print("MLE Probability Vectors Estimation")
    print("=" * 70)
    print("\nThis script requires:")
    print("  1. Prepared data file (feather format) from prepare_data.py")
    print("  2. Graph file (gpickle format) from graph.py")
    print("\nTo generate these files, run:")
    print("  python prepare_data.py <input_data> <output_prepared_data.feather>")
    print("  python graph.py <shapefile> <output_graph.gpickle>")
    print("\nThen run this script with:")
    print(f"  python mle_probability_vectors.py {args.prepared_data} {args.graph_file}")
    print("=" * 70)
    print()
    
    # Set random seed
    rng = np.random.default_rng(args.seed)
    
    # Load data
    print(f"Loading prepared data from {args.prepared_data}...")
    try:
        df = pd.read_feather(args.prepared_data).set_index("AFFGEOID")
    except Exception as e:
        print(f"Error loading data file: {e}. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading graph from {args.graph_file}...")
    try:
        with open(args.graph_file, "rb") as f:
            G = pickle.load(f)
    except Exception as e:
        print(f"Error loading graph file: {e}. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    # Align data and graph nodes
    df.sort_index(inplace=True)
    graph_nodes = {f"1500000US{node}" for node in G.nodes()}
    df_nodes = set(df.index)
    node_list = sorted(list(graph_nodes.intersection(df_nodes)))
    
    if len(node_list) != len(df):
        print(
            f"Warning: Mismatch between graph nodes and data nodes. Using {len(node_list)} common nodes."
        )
        df = df.loc[node_list]
    
    print(f"Creating sparse adjacency matrix for {len(node_list)} nodes...")
    adj_matrix = nx.to_scipy_sparse_array(
        G, nodelist=[geoid.replace("1500000US", "") for geoid in node_list], format="csr"
    )
    
    # Preprocess data
    V, D, s = preprocess_data(df)
    num_precincts = len(df)
    num_demos = len(DEMOGRAPHICS)
    num_vote_types = len(VOTE_TYPES)
    
    print(f"Data shape: {num_precincts} precincts, {num_demos} demographics, {num_vote_types} vote types")
    
    # Initialize probabilities
    print("Initializing probability vectors...")
    theta, p = initialize_probabilities(num_precincts, num_demos, num_vote_types, rng)
    
    # Run optimization
    p_final, history = run_optimization(
        p, theta, D, V, adj_matrix,
        max_iterations=args.iterations,
        learning_rate=args.learning_rate,
        rng=rng,
        grad_clip_norm=args.grad_clip_norm
    )
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    result_df = df.copy()
    
    # Add probability columns
    for demo_idx, demo in enumerate(DEMOGRAPHICS):
        for vote_idx, vote_type in enumerate(VOTE_TYPES):
            col_name = f"{vote_type}_{demo}_prob"
            result_df[col_name] = p_final[:, demo_idx, vote_idx]
    
    result_df.reset_index().to_csv(args.output, index=False)
    
    print("\n--- Success! ---")
    print(f"Results saved to '{args.output}'")
    print(f"Output format: CSV with probability estimates for each demographic and vote type")

if __name__ == "__main__":
    main()
