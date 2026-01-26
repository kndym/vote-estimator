"""
Main script that combines data preparation and MLE probability vector estimation.

This script performs:
1. Data preparation (from prepare_data.py logic)
2. MLE optimization (from mle_probability_vectors.py logic)

Input:
    - Main data CSV file with vote and demographic info
    - Graph gpickle file (from graph.py)
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
# HELPER FUNCTIONS (from prepare_data.py)
# =============================================================================

def format_affgeoid(geoid):
    """Format GEOID to AFFGEOID format."""
    return f"1500000US{geoid}"

def prepare_data_step(main_data_csv, graph_file):
    """
    Prepare data for MLE estimation (from prepare_data.py logic).
    
    Args:
        main_data_csv: Path to the main data CSV with vote and demographic info
        graph_file: Path to the blockgroups_graph.gpickle file
        
    Returns:
        df: Prepared DataFrame with AFFGEOID as index
    """
    print("\n" + "=" * 70)
    print("STEP 1: Data Preparation")
    print("=" * 70)
    
    print(f"Loading main data from {main_data_csv}...")
    df = pd.read_csv(main_data_csv, dtype={'AFFGEOID': str})
    
    # Filter to all of New York state (GEOID starts with 36)
    initial_count = len(df)
    df['geoid_part'] = df['AFFGEOID'].str.replace('1500000US', '')
    ny_mask = df['geoid_part'].str.startswith('36')
    df = df[ny_mask]
    df = df.drop(columns=['geoid_part'])
    print(f"Filtered to New York state: {len(df)} rows (from {initial_count}).")
    
    df.set_index('AFFGEOID', inplace=True)

    print(f"Loading graph from {graph_file}...")
    with open(graph_file, 'rb') as f:
        G = pickle.load(f)

    # Build adjacency dictionary (needed for alignment, but not used in MLE)
    print("Building adjacency dictionary from graph...")
    adjacency_dict = {}
    for u_geoid, v_geoid in G.edges():
        u_affgeoid = format_affgeoid(u_geoid)
        v_affgeoid = format_affgeoid(v_geoid)
        adjacency_dict.setdefault(u_affgeoid, []).append(v_affgeoid)
        adjacency_dict.setdefault(v_affgeoid, []).append(u_affgeoid)
    df['neighbors'] = df.index.to_series().map(adjacency_dict).fillna('').apply(list)
    print(f"Adjacency dictionary built for {len(adjacency_dict)} block groups.")

    # Calculate vote shares and initialize probabilities
    print("Calculating real vote shares...")
    
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
    
    df['real_D_share'] = df['votes_D'] / df['cvap_total']
    df['real_R_share'] = df['votes_R'] / df['cvap_total']
    df['real_N_share'] = df['votes_N'] / df['cvap_total']
    df['real_O_share'] = 1 - (df['real_D_share'] + df['real_R_share'] + df['real_N_share'])

    share_cols = ['real_D_share', 'real_R_share', 'real_N_share', 'real_O_share']
    df[share_cols] = df[share_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    print("Data preparation complete.")
    return df, G

# =============================================================================
# MLE FUNCTIONS (from mle_probability_vectors.py)
# =============================================================================

def preprocess_data_mle(df):
    """
    Preprocess data for MLE: scale for consistent population, then set minimums.
    
    Args:
        df: DataFrame with AFFGEOID index
        
    Returns:
        V: Array of shape (num_precincts, num_vote_types) with processed vote sums
        D: Array of shape (num_precincts, num_demos) with processed demo sums
        s: Array of shape (num_precincts,) with total populations
    """
    print("\nPreprocessing data for MLE: scaling for consistent population, then setting minimums...")
    
    # Extract vote counts (as floats)
    vote_cols = [f'votes_{vtype}' for vtype in VOTE_TYPES]
    V = np.array([df[col].values.astype(float) for col in vote_cols]).T  # (num_precincts, num_vote_types)
    
    # Extract demographic counts (as floats)
    demo_cols = [f'cvap_{demo}' for demo in DEMOGRAPHICS]
    D = np.array([df[col].values.astype(float) for col in demo_cols]).T  # (num_precincts, num_demos)
    
    # Compute totals BEFORE setting minimums
    vote_totals = V.sum(axis=1)  # (num_precincts,)
    demo_totals = D.sum(axis=1)  # (num_precincts,)
    
    # Scale vote totals to match demo totals for consistent population
    # Scale each precinct's vote totals to match its demo total
    scaling_factors = demo_totals / (vote_totals + EPSILON)  # (num_precincts,)
    V_scaled = V * scaling_factors[:, np.newaxis]  # (num_precincts, num_vote_types)
    
    # NOW set minimums after scaling (preserves proportions better)
    # Set minimum vote totals to 1 (but only for vote types that had non-zero votes)
    # Use small epsilon for zeros to avoid division issues
    V_scaled = np.maximum(V_scaled, EPSILON)
    
    # Set minimum demo totals to 1
    D = np.maximum(D, 1.0)
    
    # Re-normalize vote totals after setting minimums (to maintain consistency)
    vote_totals_scaled = V_scaled.sum(axis=1)
    renorm_factors = demo_totals / (vote_totals_scaled + EPSILON)
    V_scaled = V_scaled * renorm_factors[:, np.newaxis]
    
    # Get total population (use demo totals as the consistent population)
    s = demo_totals
    
    print(f"  Vote totals: scaled to match demo totals, minimums set after scaling")
    print(f"  Demo totals: minimum set to 1")
    print(f"  Mean scaling factor: {scaling_factors.mean():.4f}")
    
    return V_scaled, D, s

def initialize_probabilities(num_precincts, num_demos, num_vote_types, rng, V=None):
    """
    Initialize probability vectors using aggregate vote proportions from V.
    
    If V is provided, initializes all demographics with aggregate vote proportions
    (with small random perturbation). Otherwise uses flat Dirichlet.
    
    Args:
        num_precincts: Number of precincts
        num_demos: Number of demographics
        num_vote_types: Number of vote types
        rng: Random number generator
        V: Optional array of shape (num_precincts, num_vote_types) with vote totals
    """
    if V is not None:
        # Initialize from aggregate vote proportions
        V_total = V.sum(axis=0)  # Sum across precincts: (num_vote_types,)
        aggregate_props = V_total / (V_total.sum() + EPSILON)  # (num_vote_types,)
        
        # Add small random perturbation to break symmetry
        # Use Dirichlet with concentration parameters proportional to aggregate props
        # This ensures we start close to aggregate but with some variation
        concentration = aggregate_props * 100.0 + 1.0  # Scale up for tighter distribution
        
        # Sample from Dirichlet for each (precinct, demo) pair
        total_samples = num_precincts * num_demos
        p_list = []
        for _ in range(total_samples):
            sample = rng.dirichlet(concentration)
            p_list.append(sample)
        p_flat = np.array(p_list)  # (total_samples, num_vote_types)
        p = p_flat.reshape(num_precincts, num_demos, num_vote_types)
    else:
        # Fallback to flat Dirichlet
        total_samples = num_precincts * num_demos
        gammas = rng.gamma(1.0, 1.0, size=(total_samples, num_vote_types))
        p_flat = gammas / (gammas.sum(axis=1, keepdims=True) + EPSILON)
        p = p_flat.reshape(num_precincts, num_demos, num_vote_types)
    
    # Convert to softmax parameterization: p = softmax(theta)
    p_safe = np.maximum(p, EPSILON)
    log_p = np.log(p_safe)
    theta = log_p - log_p.max(axis=2, keepdims=True)
    
    return theta, p

def softmax(x, axis=-1):
    """Compute softmax function."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + EPSILON)

def compute_U(p, D):
    """Compute U[precinct, vote_type] = sum_over_demos(p[precinct, demo, vote_type] * D[precinct, demo])."""
    # Vectorized: use einsum for efficiency
    # U[i, k] = sum_j(p[i, j, k] * D[i, j])
    U = np.einsum('ijk,ij->ik', p, D)
    return U

def compute_N_demo(p, D, adj_matrix):
    """Compute N_demo[precinct, vote_type] = sum_over_neighbors(p[neighbor, demo, vote_type] * D[neighbor, demo])."""
    # Vectorized: p * D[:, :, np.newaxis] gives (num_precincts, num_demos, num_vote_types)
    # For each demo, we need adj_matrix @ (p[:, demo, :] * D[:, demo])
    # Optimized: use einsum or manual broadcasting
    num_precincts, num_demos, num_vote_types = p.shape
    N_demo = np.zeros((num_precincts, num_demos, num_vote_types), dtype=float)
    
    # Vectorized computation: broadcast D and multiply
    p_times_D = p * D[:, :, np.newaxis]  # (num_precincts, num_demos, num_vote_types)
    
    # For each demo, apply adjacency matrix
    for demo_idx in range(num_demos):
        N_demo[:, demo_idx, :] = adj_matrix @ p_times_D[:, demo_idx, :]
    
    return N_demo

def compute_dirichlet_multinomial_loglik(U, V):
    """Compute log-likelihood of Dirichlet-Multinomial: log(DirMult(U | V))."""
    V_safe = np.maximum(V, EPSILON)
    V_sum = V_safe.sum(axis=1)  # (num_precincts,)
    U_sum = U.sum(axis=1)  # (num_precincts,)
    valid_mask = (V_sum > 0) & (U_sum > 0)
    
    if not np.any(valid_mask):
        return 0.0
    
    term1 = gammaln(V_sum) - gammaln(V_sum + U_sum)
    term2 = gammaln(V_safe + U).sum(axis=1) - gammaln(V_safe).sum(axis=1)
    loglik_per_precinct = term1 + term2
    loglik = loglik_per_precinct[valid_mask].sum()
    return loglik

def compute_dirichlet_loglik(p_vector_demo, N_demo):
    """Compute log-likelihood of Dirichlet: log(Dir(p_vector_demo | N_demo))."""
    N_demo = np.maximum(N_demo, EPSILON)
    p_vector_demo = np.maximum(p_vector_demo, EPSILON)
    p_vector_demo = p_vector_demo / (p_vector_demo.sum(axis=1, keepdims=True) + EPSILON)
    N_demo_sum = N_demo.sum(axis=1)
    valid_mask = N_demo_sum > 0
    
    if not np.any(valid_mask):
        return 0.0
    
    term1 = gammaln(N_demo_sum)
    term2 = -gammaln(N_demo).sum(axis=1)
    term3 = ((N_demo - 1.0) * np.log(p_vector_demo + EPSILON)).sum(axis=1)
    loglik_per_precinct = term1 + term2 + term3
    loglik = loglik_per_precinct[valid_mask].sum()
    return loglik

def compute_total_likelihood(p, D, V, adj_matrix, dir_mult_weight=1.0, spatial_weight=1.0):
    """
    Compute total log-likelihood: dir_mult_weight * log(DirMult(U | V)) + spatial_weight * sum_over_demos(log(Dir(p_vector_demo | N_demo))).
    
    Args:
        p: Probability vectors
        D: Demo sums
        V: Scaled vote sums
        adj_matrix: Adjacency matrix
        dir_mult_weight: Weight for DirMult term (default: 1.0)
        spatial_weight: Weight for spatial smoothing term (default: 1.0)
    
    Returns:
        total_loglik: Weighted total log-likelihood
        (dir_mult_loglik, dir_loglik): Individual term values for diagnostics
    """
    U = compute_U(p, D)
    dir_mult_loglik = compute_dirichlet_multinomial_loglik(U, V)
    
    # Compute N_demo only once
    N_demo = compute_N_demo(p, D, adj_matrix)
    
    # Vectorized computation for Dirichlet loglik
    N_demo_safe = np.maximum(N_demo, EPSILON)
    p_safe = np.maximum(p, EPSILON)
    p_safe = p_safe / (p_safe.sum(axis=2, keepdims=True) + EPSILON)
    
    N_demo_sum = N_demo_safe.sum(axis=2)  # (num_precincts, num_demos)
    valid_mask = N_demo_sum > 0
    
    term1 = gammaln(N_demo_sum)  # (num_precincts, num_demos)
    term2 = -gammaln(N_demo_safe).sum(axis=2)  # (num_precincts, num_demos)
    term3 = ((N_demo_safe - 1.0) * np.log(p_safe + EPSILON)).sum(axis=2)  # (num_precincts, num_demos)
    
    dir_loglik_per_demo = (term1 + term2 + term3) * valid_mask  # Set invalid to 0
    dir_loglik = dir_loglik_per_demo.sum()
    
    # Normalize spatial term by data size to make scaling more stable
    num_precincts, num_demos = p.shape[0], p.shape[1]
    dir_loglik_normalized = dir_loglik / (num_precincts * num_demos)
    
    total_loglik = dir_mult_weight * dir_mult_loglik + spatial_weight * dir_loglik_normalized
    
    return total_loglik, (dir_mult_loglik, dir_loglik)

def compute_gradients(p, theta, D, V, adj_matrix, dir_mult_weight=1.0, spatial_weight=1.0):
    """
    Compute gradients of log-likelihood w.r.t. theta parameters.
    
    Args:
        p: Probability vectors
        theta: Unconstrained parameters
        D: Demo sums
        V: Scaled vote sums
        adj_matrix: Adjacency matrix
        dir_mult_weight: Weight for DirMult term (default: 1.0)
        spatial_weight: Weight for spatial smoothing term (default: 1.0)
    """
    num_precincts, num_demos, num_vote_types = p.shape
    grad_p = np.zeros_like(p)
    
    U = compute_U(p, D)
    
    # Gradients from DirMult term (vectorized)
    V_safe = np.maximum(V, EPSILON)
    V_sum = V_safe.sum(axis=1, keepdims=True)
    U_sum = U.sum(axis=1, keepdims=True)
    grad_U_dir_mult = digamma(V_safe + U) - digamma(V_sum + U_sum)  # (num_precincts, num_vote_types)
    
    # Vectorized: broadcast grad_U_dir_mult and D, apply weight
    grad_p += dir_mult_weight * (grad_U_dir_mult[:, np.newaxis, :] * D[:, :, np.newaxis])
    
    # Gradients from Dirichlet term
    N_demo = compute_N_demo(p, D, adj_matrix)
    N_demo_safe = np.maximum(N_demo, EPSILON)
    N_demo_sum = N_demo_safe.sum(axis=2, keepdims=True)  # (num_precincts, num_demos, 1)
    
    # Direct gradient w.r.t. p (vectorized)
    grad_p_dir_direct = (N_demo_safe - 1.0) / (p + EPSILON)
    
    # Gradient w.r.t. N_demo (vectorized)
    grad_N_demo = (
        digamma(N_demo_sum) - digamma(N_demo_safe) + np.log(p + EPSILON)
    )  # (num_precincts, num_demos, num_vote_types)
    
    # Backpropagate through N_demo (vectorized for each demo)
    for demo_idx in range(num_demos):
        # Vectorized backpropagation: adj_matrix.T @ grad_N_demo for each vote type
        grad_N_demo_backprop = adj_matrix.T @ grad_N_demo[:, demo_idx, :]  # (num_precincts, num_vote_types)
        grad_p_dir_direct[:, demo_idx, :] += grad_N_demo_backprop * D[:, demo_idx, np.newaxis]
    
    # Normalize spatial gradients by data size
    grad_p += spatial_weight * grad_p_dir_direct / (num_precincts * num_demos)
    
    # Convert grad_p to grad_theta using softmax Jacobian (vectorized)
    grad_p_times_p = (grad_p * p).sum(axis=2, keepdims=True)
    grad_theta = grad_p * p - p * grad_p_times_p
    
    return grad_theta

def adam_update(gradients, m, v, iteration, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """Adam optimizer update step."""
    m_new = beta1 * m + (1 - beta1) * gradients
    v_new = beta2 * v + (1 - beta2) * (gradients ** 2)
    m_hat = m_new / (1 - beta1 ** iteration)
    v_hat = v_new / (1 - beta2 ** iteration)
    update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return update, m_new, v_new

def compute_neighbor_similarity(p, D, adj_matrix):
    """
    Compute average neighbor similarity by demographic.
    
    Similarity is measured as 1 - average_absolute_difference in probability space.
    Returns a dict with similarity values for each demographic.
    """
    num_precincts, num_demos, num_vote_types = p.shape
    similarities = {}
    
    for demo_idx, demo in enumerate(DEMOGRAPHICS):
        p_demo = p[:, demo_idx, :]  # (num_precincts, num_vote_types)
        D_demo = D[:, demo_idx]  # (num_precincts,)
        
        diffs = []
        for vote_idx in range(num_vote_types):
            # Compute neighbor average probabilities
            p_vote = p_demo[:, vote_idx]  # (num_precincts,)
            pred_voters = D_demo * p_vote
            
            neighbor_voters_sum = adj_matrix @ pred_voters
            neighbor_pop_sum = adj_matrix @ D_demo
            neighbor_avg_prob = neighbor_voters_sum / (neighbor_pop_sum + EPSILON)
            
            # Absolute difference
            diff = np.abs(p_vote - neighbor_avg_prob)
            diffs.append(diff)
        
        # Weighted average difference across all vote types
        if diffs:
            all_diffs = np.concatenate(diffs)
            weights = np.tile(D_demo, num_vote_types)
            valid_mask = ~np.isnan(all_diffs) & (weights > 0)
            
            if np.any(valid_mask):
                avg_diff = np.average(all_diffs[valid_mask], weights=weights[valid_mask])
                # Similarity: 1 - average_difference (higher is more similar)
                similarities[demo] = 1.0 - avg_diff
            else:
                similarities[demo] = 0.0
        else:
            similarities[demo] = 0.0
    
    return similarities

def compute_avg_probability_vector(p, D):
    """
    Compute average probability vector across all precincts, weighted by demographic population.
    
    Returns a dict with average probabilities for each demographic and vote type combination.
    """
    num_precincts, num_demos, num_vote_types = p.shape
    avg_probs = {}
    
    for demo_idx, demo in enumerate(DEMOGRAPHICS):
        p_demo = p[:, demo_idx, :]  # (num_precincts, num_vote_types)
        D_demo = D[:, demo_idx]  # (num_precincts,)
        total_pop = D_demo.sum()
        
        if total_pop > 0:
            # Weighted average across precincts
            for vote_idx, vote_type in enumerate(VOTE_TYPES):
                p_vote = p_demo[:, vote_idx]  # (num_precincts,)
                avg_prob = np.average(p_vote, weights=D_demo)
                avg_probs[f"{vote_type}_{demo}"] = avg_prob
        else:
            # If no population, use unweighted mean
            for vote_idx, vote_type in enumerate(VOTE_TYPES):
                p_vote = p_demo[:, vote_idx]
                avg_probs[f"{vote_type}_{demo}"] = np.mean(p_vote)
    
    return avg_probs

def run_optimization(p, theta, D, V, adj_matrix, max_iterations, learning_rate, rng, grad_clip_norm=5.0, dir_mult_weight=1.0, spatial_weight=1.0):
    """
    Run gradient descent optimization using Adam.
    
    Args:
        p: Initial probability vectors
        theta: Initial unconstrained parameters
        D: Demo sums
        V: Scaled vote sums
        adj_matrix: Adjacency matrix
        max_iterations: Maximum iterations
        learning_rate: Learning rate
        rng: Random number generator
        grad_clip_norm: Gradient clipping norm
        dir_mult_weight: Weight for DirMult term (default: 1.0)
    """
    print(f"\nStarting optimization: {max_iterations} iterations...")
    print(f"DirMult term weight: {dir_mult_weight}, Spatial term weight: {spatial_weight}")
    
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    history = []
    
    for iteration in tqdm(range(1, max_iterations + 1), desc="Optimization"):
        loglik, (dir_mult_loglik, dir_loglik) = compute_total_likelihood(p, D, V, adj_matrix, dir_mult_weight, spatial_weight)
        history_entry = {"iteration": iteration, "loglik": loglik, "dir_mult_loglik": dir_mult_loglik, "dir_loglik": dir_loglik}
        
        similarities = None
        avg_probs = None
        # Compute neighbor similarity and average probability vector every 10 iterations
        if iteration % 10 == 0:
            similarities = compute_neighbor_similarity(p, D, adj_matrix)
            for demo, sim in similarities.items():
                history_entry[f"neighbor_sim_{demo}"] = sim
            
            avg_probs = compute_avg_probability_vector(p, D)
            for key, prob in avg_probs.items():
                history_entry[f"avg_prob_{key}"] = prob
        
        history.append(history_entry)
        
        grad_theta = compute_gradients(p, theta, D, V, adj_matrix, dir_mult_weight, spatial_weight)
        
        grad_norm = np.linalg.norm(grad_theta)
        if grad_norm > grad_clip_norm:
            grad_theta = grad_theta * (grad_clip_norm / (grad_norm + EPSILON))
        
        update, m, v = adam_update(grad_theta, m, v, iteration, learning_rate)
        theta = theta + update
        p = softmax(theta)
        
        if iteration % 10 == 0 or iteration == 1:
            # Print term magnitudes and ratio for diagnostics
            term_ratio = dir_loglik / (dir_mult_loglik + EPSILON) if dir_mult_loglik != 0 else float('inf')
            print(f"  Iteration {iteration}: total_loglik = {loglik:.2f}, grad_norm = {grad_norm:.4f}")
            print(f"    DirMult: {dir_mult_loglik:.2f}, Dirichlet: {dir_loglik:.2f}, Ratio: {term_ratio:.2f}")
            
            # Compute U and compare to V for diagnostics
            U = compute_U(p, D)
            U_total = U.sum(axis=0)  # Sum across precincts
            V_total = V.sum(axis=0)  # Sum across precincts
            U_props = U_total / (U_total.sum() + EPSILON)
            V_props = V_total / (V_total.sum() + EPSILON)
            print(f"    U proportions: D={U_props[0]:.4f}, R={U_props[1]:.4f}, O={U_props[2]:.4f}, N={U_props[3]:.4f}")
            print(f"    V proportions: D={V_props[0]:.4f}, R={V_props[1]:.4f}, O={V_props[2]:.4f}, N={V_props[3]:.4f}")
            
            if similarities is not None:
                sim_str = ", ".join([f"{demo}={sim:.4f}" for demo, sim in similarities.items()])
                print(f"    Neighbor similarity: {sim_str}")
            if avg_probs is not None:
                # Print average probabilities for first demographic as example
                demo = DEMOGRAPHICS[0]
                prob_str = ", ".join([f"{vtype}={avg_probs[f'{vtype}_{demo}']:.4f}" for vtype in VOTE_TYPES])
                print(f"    Avg prob ({demo}): {prob_str}")
    
    return p, history

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Combined data preparation and MLE probability vector estimation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First, generate the graph file (if not already created):
  python graph.py <shapefile> <output_graph.gpickle>
  
  # Then run this script:
  python main_mle.py <main_data_csv> <graph_file> --output results.csv
        """
    )
    parser.add_argument(
        "main_data_csv",
        help="Path to the main data CSV with vote and demographic info."
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
    parser.add_argument(
        "--dir-mult-weight", type=float, default=1.0,
        help="Weight for DirMult term in likelihood (default: 1.0). Increase to give more weight to data fitting vs spatial smoothing."
    )
    args = parser.parse_args()
    
    # Print banner
    print("=" * 70)
    print("MLE Probability Vectors Estimation")
    print("Combined Data Preparation and Optimization")
    print("=" * 70)
    print("\nThis script performs:")
    print("  1. Data preparation (from prepare_data.py)")
    print("  2. MLE optimization (from mle_probability_vectors.py)")
    print("\nPrerequisites:")
    print("  - Graph file (gpickle format) from graph.py")
    print("  - Main data CSV file with vote and demographic info")
    print("\nTo generate the graph file, run:")
    print("  python graph.py <shapefile> <output_graph.gpickle>")
    print("=" * 70)
    
    # Set random seed
    rng = np.random.default_rng(args.seed)
    
    # Step 1: Data preparation
    df, G = prepare_data_step(args.main_data_csv, args.graph_file)
    
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
    
    print(f"\nCreating sparse adjacency matrix for {len(node_list)} nodes...")
    adj_matrix = nx.to_scipy_sparse_array(
        G, nodelist=[geoid.replace("1500000US", "") for geoid in node_list], format="csr"
    )
    
    # Step 2: MLE preprocessing and optimization
    print("\n" + "=" * 70)
    print("STEP 2: MLE Optimization")
    print("=" * 70)
    
    V, D, s = preprocess_data_mle(df)
    num_precincts = len(df)
    num_demos = len(DEMOGRAPHICS)
    num_vote_types = len(VOTE_TYPES)
    
    print(f"Data shape: {num_precincts} precincts, {num_demos} demographics, {num_vote_types} vote types")
    
    print("\nInitializing probability vectors with flat Dirichlet distribution...")
    theta, p = initialize_probabilities(num_precincts, num_demos, num_vote_types, rng, V=None)
    
    p_final, history = run_optimization(
        p, theta, D, V, adj_matrix,
        max_iterations=args.iterations,
        learning_rate=args.learning_rate,
        rng=rng,
        grad_clip_norm=args.grad_clip_norm,
        dir_mult_weight=args.dir_mult_weight
    )
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    result_df = df.copy()
    
    for demo_idx, demo in enumerate(DEMOGRAPHICS):
        for vote_idx, vote_type in enumerate(VOTE_TYPES):
            col_name = f"{vote_type}_{demo}_prob"
            result_df[col_name] = p_final[:, demo_idx, vote_idx]
    
    result_df.reset_index().to_csv(args.output, index=False)
    
    print("\n" + "=" * 70)
    print("--- Success! ---")
    print("=" * 70)
    print(f"Results saved to '{args.output}'")
    print(f"Output format: CSV with probability estimates for each demographic and vote type")

if __name__ == "__main__":
    main()
