"""
Maximum Likelihood Estimation for Dirichlet parameters using gradient descent.

Clean implementation: For each block group and demographic, we estimate Dirichlet
parameters α that define probability distributions over vote types (D, R, O, N).

Algorithm:
1. Initialize α = (2, 2, 2, 2) for each (block group, demographic)
2. For each iteration:
   a. Sample probability vectors from Dirichlet(α) for all (bg, demo)
   b. Fill vote count matrices using these probabilities (respecting row/column constraints)
   c. Aggregate to neighborhoods
   d. Compute Dirichlet-multinomial log-likelihood on neighborhood counts
   e. Compute gradients and update α using Adam optimizer

Input:
    - Prepared data feather file (from prepare_data.py)
    - Graph gpickle file (for neighborhood definition)
Output:
    - CSV with Dirichlet parameters: AFFGEOID, Wht_alpha_D, Wht_alpha_R, etc. (wide format)
"""
import pandas as pd
import numpy as np
import argparse
import pickle
import networkx as nx
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.special import gammaln, digamma

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
    Preprocess data: round to integers and ensure vote totals match CVAP totals.
    
    Args:
        df: DataFrame with AFFGEOID index
        
    Returns:
        df: Preprocessed DataFrame
    """
    print("Preprocessing data: rounding to integers and validating sums...")
    
    # Round CVAP columns to non-negative integers
    for demo in DEMOGRAPHICS:
        col = f'cvap_{demo}'
        if col in df.columns:
            df[col] = df[col].round().astype(int).clip(lower=0)
    
    # Calculate CVAP total
    cvap_cols = [f'cvap_{demo}' for demo in DEMOGRAPHICS]
    df['cvap_total'] = df[cvap_cols].sum(axis=1)
    
    # Calculate votes from shares/probabilities if not already present
    vote_cols = [f'votes_{vtype}' for vtype in VOTE_TYPES]
    if not all(col in df.columns for col in vote_cols):
        print("  Computing vote totals from shares/probabilities...")
        if 'real_D_share' in df.columns:
            df['votes_D'] = (df['real_D_share'] * df['cvap_total']).round().astype(int).clip(lower=0)
            df['votes_R'] = (df['real_R_share'] * df['cvap_total']).round().astype(int).clip(lower=0)
            df['votes_O'] = (df['real_O_share'] * df['cvap_total']).round().astype(int).clip(lower=0)
            df['votes_N'] = (df['real_N_share'] * df['cvap_total']).round().astype(int).clip(lower=0)
        elif 'real_D_prob' in df.columns:
            df['votes_D'] = (df['real_D_prob'] * df['cvap_total']).round().astype(int).clip(lower=0)
            df['votes_R'] = (df['real_R_prob'] * df['cvap_total']).round().astype(int).clip(lower=0)
            df['votes_O'] = (df['real_O_prob'] * df['cvap_total']).round().astype(int).clip(lower=0)
            df['votes_N'] = (df['real_N_prob'] * df['cvap_total']).round().astype(int).clip(lower=0)
        else:
            raise ValueError("Could not find vote share or probability columns to compute votes")
    
    # Round vote columns to non-negative integers
    for vtype in VOTE_TYPES:
        col = f'votes_{vtype}'
        if col in df.columns:
            df[col] = df[col].round().astype(int).clip(lower=0)
    
    df['votes_total'] = df[vote_cols].sum(axis=1)
    
    # Ensure vote totals match CVAP totals per block group
    print("  Ensuring vote totals match CVAP totals per precinct...")
    mismatches = (df['votes_total'] != df['cvap_total']).sum()
    if mismatches > 0:
        print(f"  Warning: {mismatches} precincts still have mismatched totals. Adjusting...")
        # Adjust votes_N to make totals match
        df['votes_N'] = df['cvap_total'] - (df['votes_D'] + df['votes_R'] + df['votes_O'])
        df['votes_N'] = df['votes_N'].clip(lower=0)
        df['votes_total'] = df[vote_cols].sum(axis=1)
    
    print(f"  CVAP total: {df['cvap_total'].sum():.0f}")
    print(f"  Votes total: {df['votes_total'].sum():.0f}")
    print(f"  Mismatched precincts: {(df['votes_total'] != df['cvap_total']).sum()}")
    
    return df

def build_neighborhoods(df, graph):
    """
    Build neighborhood dictionary from graph.
    
    Args:
        df: DataFrame with AFFGEOID index
        graph: NetworkX graph
        
    Returns:
        neighborhoods: Dict mapping AFFGEOID to list of neighbor AFFGEOIDs (including self)
    """
    print("Building neighborhoods from graph...")
    
    neighborhoods = {}
    graph_nodes = {f"1500000US{node}" for node in graph.nodes()}
    df_nodes = set(df.index)
    common_nodes = graph_nodes.intersection(df_nodes)
    
    for node in common_nodes:
        geoid = node.replace("1500000US", "")
        neighbors = [format_affgeoid(n) for n in graph.neighbors(geoid)]
        # Include self in neighborhood
        neighborhoods[node] = [node] + [n for n in neighbors if n in df_nodes]
    
    # For nodes not in graph, just include self
    for node in df_nodes:
        if node not in neighborhoods:
            neighborhoods[node] = [node]
    
    avg_neighbors = np.mean([len(n) for n in neighborhoods.values()])
    print(f"  Built neighborhoods for {len(neighborhoods)} precincts")
    print(f"  Average neighborhood size: {avg_neighbors:.2f}")
    
    return neighborhoods

def build_sparse_neighbor_matrix(neighbor_indices, num_blockgroups):
    """
    Build sparse matrix for neighborhood aggregation.
    
    Args:
        neighbor_indices: List of lists, neighbor_indices[i] contains indices of neighbors for block group i
        num_blockgroups: Number of block groups
        
    Returns:
        neighbor_matrix: Sparse CSR matrix of shape (num_blockgroups, num_blockgroups)
    """
    rows = []
    cols = []
    
    for i, neighbors in enumerate(neighbor_indices):
        valid_neighbors = [n for n in neighbors if n >= 0 and n < num_blockgroups]
        for j in valid_neighbors:
            rows.append(i)
            cols.append(j)
    
    neighbor_matrix = csr_matrix((np.ones(len(rows), dtype=float), (rows, cols)), 
                                 shape=(num_blockgroups, num_blockgroups))
    
    return neighbor_matrix

# =============================================================================
# VOTE COUNT MATRIX GENERATION
# =============================================================================

def ipf_adjust(vote_matrix, row_sums, col_sums, max_iter=20):
    """
    Iterative Proportional Fitting to adjust matrix to satisfy row/column constraints exactly.
    
    Args:
        vote_matrix: Array of shape (4, 5) with vote counts
        row_sums: Array of shape (4,) with target row sums (vote type totals)
        col_sums: Array of shape (5,) with target column sums (demo totals)
        max_iter: Maximum number of IPF iterations
        
    Returns:
        vote_matrix: Adjusted matrix satisfying constraints exactly
    """
    vote_matrix = vote_matrix.astype(float)
    
    # IPF iterations
    for _ in range(max_iter):
        # Adjust rows to match row sums
        row_totals = vote_matrix.sum(axis=1)
        mask = row_totals > EPSILON
        if mask.any():
            vote_matrix[mask, :] = vote_matrix[mask, :] * (row_sums[mask, np.newaxis] / (row_totals[mask, np.newaxis] + EPSILON))
        
        # Adjust columns to match column sums
        col_totals = vote_matrix.sum(axis=0)
        mask = col_totals > EPSILON
        if mask.any():
            vote_matrix[:, mask] = vote_matrix[:, mask] * (col_sums[mask] / (col_totals[mask] + EPSILON))
        
        # Check convergence
        row_error = np.abs(vote_matrix.sum(axis=1) - row_sums).max()
        col_error = np.abs(vote_matrix.sum(axis=0) - col_sums).max()
        if row_error < 0.1 and col_error < 0.1:
            break
    
    # Round to integers and ensure non-negative
    vote_matrix = np.round(vote_matrix).astype(int)
    vote_matrix = np.maximum(vote_matrix, 0)
    
    # Final adjustment to satisfy constraints exactly
    row_diff = row_sums - vote_matrix.sum(axis=1)
    col_diff = col_sums - vote_matrix.sum(axis=0)
    
    # Distribute row differences proportionally
    for i in range(len(row_sums)):
        if row_diff[i] != 0 and vote_matrix[i, :].sum() > 0:
            weights = vote_matrix[i, :].astype(float)
            weights = weights / (weights.sum() + EPSILON)
            vote_matrix[i, :] += np.round(weights * row_diff[i]).astype(int)
    
    # Distribute column differences proportionally
    col_totals = vote_matrix.sum(axis=0)
    col_diff = col_sums - col_totals
    for j in range(len(col_sums)):
        if col_diff[j] != 0 and vote_matrix[:, j].sum() > 0:
            weights = vote_matrix[:, j].astype(float)
            weights = weights / (weights.sum() + EPSILON)
            vote_matrix[:, j] += np.round(weights * col_diff[j]).astype(int)
    
    return np.maximum(vote_matrix, 0)

def fill_vote_matrix_from_probs(prob_vectors, vote_totals, demo_totals, rng):
    """
    Fill vote count matrix using probability vectors, respecting row/column constraints.
    
    Args:
        prob_vectors: Array of shape (5, 4) with probability vectors for each demographic
        vote_totals: Array of shape (4,) with vote type totals (row sums)
        demo_totals: Array of shape (5,) with demographic totals (column sums)
        rng: Random number generator
        
    Returns:
        vote_matrix: Array of shape (4, 5) with vote counts
    """
    # Create joint probability matrix weighted by demo totals
    # prob_vectors is (5 demos, 4 vote types)
    # We want (4 vote types, 5 demos)
    prob_matrix = prob_vectors.T  # (4, 5)
    
    # Weight by demographic totals
    cell_probs = prob_matrix * demo_totals[np.newaxis, :]  # (4, 5)
    
    # Normalize to create proper probability distribution
    total_prob = cell_probs.sum()
    if total_prob > EPSILON:
        cell_probs = cell_probs / total_prob
    else:
        # Uniform fallback
        cell_probs = np.ones((4, 5)) / 20.0
    
    # Sample all votes at once using multinomial
    total_votes = vote_totals.sum()
    sampled_flat = rng.multinomial(total_votes, cell_probs.flatten())
    sampled_matrix = sampled_flat.reshape(4, 5)
    
    # Use IPF to ensure exact constraint satisfaction
    vote_matrix = ipf_adjust(sampled_matrix, vote_totals, demo_totals, max_iter=20)
    
    return vote_matrix

def generate_vote_matrices(alpha, vote_totals, demo_totals, rng):
    """
    Generate vote count matrices for all block groups.
    
    For each block group:
    1. Sample probability vectors from Dirichlet(α) for each demographic
    2. Use these probabilities to fill the vote count matrix (respecting constraints)
    
    Args:
        alpha: Array of shape (num_blockgroups, num_demos, 4) with Dirichlet parameters
        vote_totals: Array of shape (num_blockgroups, 4) with vote type totals
        demo_totals: Array of shape (num_blockgroups, 5) with demographic totals
        rng: Random number generator
        
    Returns:
        vote_matrices: Array of shape (num_blockgroups, 4, 5) with vote counts
    """
    num_blockgroups = alpha.shape[0]
    num_demos = len(DEMOGRAPHICS)
    num_votes = len(VOTE_TYPES)
    
    vote_matrices = np.zeros((num_blockgroups, num_votes, num_demos), dtype=int)
    
    # Sample probability vectors from Dirichlet(α) for all (bg, demo) pairs
    # alpha shape: (num_blockgroups, num_demos, 4)
    # Sample gamma variates: gamma(alpha, 1) for each component
    gammas = rng.gamma(alpha, 1.0)  # Same shape as alpha
    # Normalize to get Dirichlet samples
    prob_vectors_all = gammas / np.maximum(gammas.sum(axis=2, keepdims=True), EPSILON)
    # Shape: (num_blockgroups, num_demos, 4)
    
    # Fill vote matrices for each block group
    for bg_idx in range(num_blockgroups):
        # Get probability vectors for this block group: (num_demos, 4)
        prob_vectors_bg = prob_vectors_all[bg_idx]  # (5, 4)
        
        # Get constraints for this block group
        votes_bg = vote_totals[bg_idx]  # (4,)
        demos_bg = demo_totals[bg_idx]  # (5,)
        
        # Fill matrix using probabilities
        vote_matrix = fill_vote_matrix_from_probs(
            prob_vectors_bg, votes_bg, demos_bg, rng
        )
        
        vote_matrices[bg_idx] = vote_matrix
    
    return vote_matrices

# =============================================================================
# NEIGHBORHOOD AGGREGATION
# =============================================================================

def aggregate_to_neighborhoods(vote_matrices, neighbor_matrix, cvap_array):
    """
    Aggregate vote counts to neighborhoods for each (block group, demo).
    
    Args:
        vote_matrices: Array of shape (num_blockgroups, 4, 5) with vote counts
        neighbor_matrix: Sparse CSR matrix for neighborhood aggregation
        cvap_array: Array of shape (num_blockgroups, 5) with demographic populations
        
    Returns:
        neighborhood_counts: Array of shape (num_blockgroups, num_demos, 4) with aggregated vote counts
        neighborhood_pops: Array of shape (num_blockgroups, num_demos) with aggregated demo populations
    """
    num_blockgroups = vote_matrices.shape[0]
    num_demos = len(DEMOGRAPHICS)
    
    neighborhood_counts = np.zeros((num_blockgroups, num_demos, 4), dtype=int)
    neighborhood_pops = np.zeros((num_blockgroups, num_demos), dtype=int)
    
    for demo_idx in range(num_demos):
        # Extract vote counts for this demographic: (num_blockgroups, 4)
        vote_counts_demo = vote_matrices[:, :, demo_idx].astype(float)  # (num_blockgroups, 4)
        
        # Aggregate neighbors using sparse matrix multiplication
        neighborhood_counts[:, demo_idx, :] = (neighbor_matrix @ vote_counts_demo).astype(int)
        
        # Aggregate demo populations
        cvap_demo = cvap_array[:, demo_idx]  # (num_blockgroups,)
        neighborhood_pops[:, demo_idx] = (neighbor_matrix @ cvap_demo).astype(int)
    
    return neighborhood_counts, neighborhood_pops

# =============================================================================
# LIKELIHOOD COMPUTATION
# =============================================================================

def compute_total_likelihood(neighborhood_counts, neighborhood_pops, alpha):
    """
    Compute total log-likelihood across all (block group, demo) neighborhoods.
    Vectorized for performance.
    
    The likelihood is Dirichlet-multinomial:
    P(X | α, N) = Γ(α₀) / Γ(α₀ + N) * ∏ₖ Γ(αₖ + Xₖ) / Γ(αₖ)
    
    where α₀ = Σₖ αₖ and X is the count vector, N is the total.
    
    Args:
        neighborhood_counts: Array of shape (num_blockgroups, num_demos, 4) with aggregated counts
        neighborhood_pops: Array of shape (num_blockgroups, num_demos) with aggregated populations
        alpha: Array of shape (num_blockgroups, num_demos, 4) with Dirichlet parameters
        
    Returns:
        total_loglik: Scalar total log-likelihood
    """
    # Vectorized computation
    alpha_sums = alpha.sum(axis=2)  # (num_blockgroups, num_demos)
    totals = neighborhood_pops  # (num_blockgroups, num_demos)
    
    # Mask for valid entries (non-zero population)
    valid_mask = totals > 0
    
    # Compute log-likelihood components vectorized
    # term1: log_gamma(sum(alpha)) - log_gamma(sum(alpha) + total)
    term1 = gammaln(alpha_sums) - gammaln(alpha_sums + totals)
    
    # term2: sum(log_gamma(alpha + counts)) - sum(log_gamma(alpha))
    term2 = gammaln(alpha + neighborhood_counts).sum(axis=2) - gammaln(alpha).sum(axis=2)
    
    # Combine terms
    loglik_per_cell = term1 + term2
    
    # Sum only valid entries
    total_loglik = loglik_per_cell[valid_mask].sum()
    
    return total_loglik

# =============================================================================
# GRADIENT COMPUTATION
# =============================================================================

def compute_gradients(neighborhood_counts, neighborhood_pops, alpha):
    """
    Compute gradients of log-likelihood w.r.t. alpha parameters.
    Vectorized for performance.
    
    Gradient formula for Dirichlet-multinomial:
    dL/dαₖ = ψ(α₀) - ψ(α₀ + N) + ψ(αₖ + Xₖ) - ψ(αₖ)
    
    where ψ is the digamma function, α₀ = Σⱼ αⱼ, N is total, Xₖ is count for k.
    
    Args:
        neighborhood_counts: Array of shape (num_blockgroups, num_demos, 4) with aggregated counts
        neighborhood_pops: Array of shape (num_blockgroups, num_demos) with aggregated populations
        alpha: Array of shape (num_blockgroups, num_demos, 4) with Dirichlet parameters
        
    Returns:
        gradients: Array of shape (num_blockgroups, num_demos, 4) with gradients
    """
    # Vectorized computation
    alpha_sums = alpha.sum(axis=2, keepdims=True)  # (num_blockgroups, num_demos, 1)
    totals = neighborhood_pops[:, :, np.newaxis]  # (num_blockgroups, num_demos, 1)
    
    # Gradient formula for Dirichlet-multinomial (vectorized):
    digamma_alpha_sum = digamma(alpha_sums)  # (num_blockgroups, num_demos, 1)
    digamma_alpha_sum_plus_total = digamma(alpha_sums + totals)  # (num_blockgroups, num_demos, 1)
    
    gradients = (
        digamma_alpha_sum
        - digamma_alpha_sum_plus_total
        + digamma(alpha + neighborhood_counts)  # (num_blockgroups, num_demos, 4)
        - digamma(alpha)  # (num_blockgroups, num_demos, 4)
    )
    
    # Set gradients to zero where neighborhood population is zero
    zero_pop_mask = (neighborhood_pops == 0)  # (num_blockgroups, num_demos)
    gradients[zero_pop_mask, :] = 0.0
    
    return gradients

# =============================================================================
# METHOD OF MOMENTS INITIALIZATION
# =============================================================================

def method_of_moments_initialization(
    vote_totals, demo_totals, neighbor_matrix, rng, num_warmup_iterations=10, alpha_sum_target=8.0
):
    """
    Initialize alpha parameters using Method of Moments.
    
    Runs a few iterations of sampling and MoM estimation to get good starting values.
    
    Args:
        vote_totals: Array of shape (num_blockgroups, 4) with vote type totals
        demo_totals: Array of shape (num_blockgroups, 5) with demographic totals
        neighbor_matrix: Sparse CSR matrix for neighborhood aggregation
        rng: Random number generator
        num_warmup_iterations: Number of warmup iterations for MoM
        alpha_sum_target: Target total concentration for alpha
        
    Returns:
        alpha: Array of shape (num_blockgroups, num_demos, 4) with initialized Dirichlet parameters
    """
    num_blockgroups = vote_totals.shape[0]
    num_demos = len(DEMOGRAPHICS)
    
    print(f"Method of Moments initialization ({num_warmup_iterations} warmup iterations)...")
    
    # Start with flat prior
    alpha = np.ones((num_blockgroups, num_demos, 4)) * (alpha_sum_target / 4.0)
    
    # Track mean and variance using Welford's algorithm
    mean_probs = np.zeros((num_blockgroups, num_demos, 4))
    M2 = np.zeros((num_blockgroups, num_demos, 4))
    count = 0
    
    for iteration in range(num_warmup_iterations):
        # Sample probability vectors from current alpha
        gammas = rng.gamma(alpha, 1.0)
        prob_vectors = gammas / np.maximum(gammas.sum(axis=2, keepdims=True), EPSILON)
        
        # Generate vote matrices
        vote_matrices = generate_vote_matrices(alpha, vote_totals, demo_totals, rng)
        
        # Aggregate to neighborhoods
        neighborhood_counts, neighborhood_pops = aggregate_to_neighborhoods(
            vote_matrices, neighbor_matrix, demo_totals
        )
        
        # Update Welford statistics
        count += 1
        delta = prob_vectors - mean_probs
        mean_probs += delta / count
        delta2 = prob_vectors - mean_probs
        M2 += delta * delta2
        
        # Estimate alpha using method of moments
        if count >= 2:
            variance = M2 / (count - 1)
            mu = mean_probs
            
            # Method of moments: S_k = μ_k(1-μ_k)/σ_k² - 1
            numerator = mu * (1 - mu)
            s_squared = np.maximum(variance, EPSILON)
            S_k = numerator / s_squared - 1
            S_k = np.clip(S_k, 0, 1000)  # Prevent explosion
            
            # Average S_k across vote types
            S_total = S_k.mean(axis=2)  # (num_blockgroups, num_demos)
            
            # Calculate alpha: α_k = S_total * μ_k
            alpha_new = S_total[:, :, np.newaxis] * mu
            
            # Scale to target concentration
            alpha_sum_current = alpha_new.sum(axis=2)
            scale_factor = alpha_sum_target / np.maximum(alpha_sum_current, EPSILON)
            alpha = alpha_new * scale_factor[:, :, np.newaxis]
            
            # Ensure positivity
            alpha = np.maximum(alpha, EPSILON)
    
    print(f"  Initialized {num_blockgroups * num_demos} alpha vectors")
    print(f"  Mean alpha sum: {alpha.sum(axis=2).mean():.2f}")
    
    return alpha

# =============================================================================
# HIERARCHICAL SMOOTHING
# =============================================================================

def apply_hierarchical_smoothing(alpha, neighbor_matrix, smoothing_strength=0.1):
    """
    Apply hierarchical smoothing to alpha parameters using neighbor information.
    
    For each (block group, demo), blend its alpha with the mean of its neighbors:
    α_smoothed = (1 - λ) * α_local + λ * α_neighbor_mean
    
    Args:
        alpha: Array of shape (num_blockgroups, num_demos, 4) with Dirichlet parameters
        neighbor_matrix: Sparse CSR matrix for neighborhood aggregation
        smoothing_strength: Strength of smoothing (0 = no smoothing, 1 = full neighbor mean)
        
    Returns:
        alpha_smoothed: Smoothed alpha parameters
    """
    num_blockgroups = alpha.shape[0]
    num_demos = alpha.shape[1]
    
    # Compute neighbor means for each (bg, demo)
    # For each demo, aggregate alpha values across neighborhoods
    alpha_smoothed = alpha.copy()
    
    for demo_idx in range(num_demos):
        # Extract alpha for this demographic: (num_blockgroups, 4)
        alpha_demo = alpha[:, demo_idx, :].astype(float)
        
        # Compute neighbor means using sparse matrix
        # neighbor_matrix @ alpha_demo gives sum, need to divide by neighbor counts
        neighbor_sums = neighbor_matrix @ alpha_demo  # (num_blockgroups, 4)
        neighbor_counts = neighbor_matrix @ np.ones((num_blockgroups, 1))  # (num_blockgroups, 1)
        neighbor_means = neighbor_sums / np.maximum(neighbor_counts, 1.0)  # (num_blockgroups, 4)
        
        # Blend local and neighbor means
        alpha_smoothed[:, demo_idx, :] = (
            (1 - smoothing_strength) * alpha_demo + smoothing_strength * neighbor_means
        )
    
    return alpha_smoothed

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

def run_mle_optimization(
    df,
    neighborhoods,
    max_iterations,
    learning_rate,
    rng,
    lambda_reg=1e-4,
    samples_per_iter=3,
    grad_clip_norm=5.0,
    init_method="random",
    use_hierarchical_smoothing=False,
    smoothing_strength=0.1,
    mom_warmup_iterations=10,
):
    """
    Run MLE optimization using gradient descent.
    
    Algorithm:
    1. Initialize α = (2, 2, 2, 2) for each (block group, demographic)
    2. For each iteration:
       a. Sample probability vectors from Dirichlet(α) for all (bg, demo)
       b. Fill vote count matrices using these probabilities (respecting constraints)
       c. Aggregate to neighborhoods
       d. Compute Dirichlet-multinomial log-likelihood on neighborhood counts
       e. Compute gradients and update α using Adam optimizer
    
    Args:
        df: Prepared DataFrame
        neighborhoods: Dict mapping AFFGEOID to list of neighbor AFFGEOIDs
        max_iterations: Maximum number of iterations
        learning_rate: Learning rate for Adam optimizer
        rng: Random number generator
        lambda_reg: L2 regularization strength on log_alpha
        
    Returns:
        alpha: Array of shape (num_blockgroups, num_demos, 4) with final Dirichlet parameters
    """
    num_blockgroups = len(df)
    num_demos = len(DEMOGRAPHICS)
    
    print(f"Starting MLE optimization: {max_iterations} iterations (samples/iter={samples_per_iter})...")
    
    # Pre-compute arrays
    vote_totals = np.array([df[f'votes_{vtype}'].values for vtype in VOTE_TYPES], dtype=int).T  # (num_blockgroups, 4)
    demo_totals = np.array([df[f'cvap_{demo}'].values for demo in DEMOGRAPHICS], dtype=int).T  # (num_blockgroups, 5)
    
    # Build neighbor indices and sparse matrix
    geoid_to_idx = {geoid: idx for idx, geoid in enumerate(df.index)}
    neighbor_indices = []
    for geoid in df.index:
        neighbors = neighborhoods.get(geoid, [geoid])
        neighbor_indices.append([geoid_to_idx.get(n, -1) for n in neighbors if n in geoid_to_idx])
    
    neighbor_matrix = build_sparse_neighbor_matrix(neighbor_indices, num_blockgroups)
    print(f"  Sparse matrix shape: {neighbor_matrix.shape}, nnz: {neighbor_matrix.nnz}")
    
    # Initialize alpha parameters based on chosen method
    alpha_sum_init = 8.0
    
    if init_method == "mom":
        # Method of Moments initialization
        alpha = method_of_moments_initialization(
            vote_totals, demo_totals, neighbor_matrix, rng,
            num_warmup_iterations=mom_warmup_iterations,
            alpha_sum_target=alpha_sum_init
        )
    elif init_method == "random":
        # Random initialization: proportions from flat Dirichlet prior
        gammas_init = rng.gamma(1.0, 1.0, size=(num_blockgroups, num_demos, 4))
        probs_init = gammas_init / np.maximum(gammas_init.sum(axis=2, keepdims=True), EPSILON)
        alpha = alpha_sum_init * probs_init
    else:
        # Uniform initialization
        alpha = np.ones((num_blockgroups, num_demos, 4)) * (alpha_sum_init / 4.0)
    
    # Apply hierarchical smoothing if requested
    if use_hierarchical_smoothing:
        print(f"  Applying hierarchical smoothing (strength={smoothing_strength})...")
        alpha = apply_hierarchical_smoothing(alpha, neighbor_matrix, smoothing_strength)
    
    # Optimize in log-space to ensure positivity
    log_alpha = np.log(alpha)
    
    # Initialize Adam optimizer state
    m = np.zeros_like(log_alpha)
    v = np.zeros_like(log_alpha)
    
    # Track likelihood for convergence
    prev_likelihood = -np.inf
    no_improvement_count = 0
    
    # Main optimization loop
    for iteration in tqdm(range(1, max_iterations + 1), desc="MLE optimization"):
        # Monte Carlo averaging over multiple samples to reduce gradient noise
        total_likelihood = 0.0
        total_gradients = np.zeros_like(alpha)

        for _ in range(samples_per_iter):
            # 1. Sample probability vectors from Dirichlet(α) and generate vote matrices
            vote_matrices = generate_vote_matrices(alpha, vote_totals, demo_totals, rng)

            # 2. Aggregate to neighborhoods
            cvap_array = demo_totals
            neighborhood_counts, neighborhood_pops = aggregate_to_neighborhoods(
                vote_matrices, neighbor_matrix, cvap_array
            )

            # 3. Compute likelihood
            total_likelihood += compute_total_likelihood(neighborhood_counts, neighborhood_pops, alpha)

            # 4. Compute gradients
            total_gradients += compute_gradients(neighborhood_counts, neighborhood_pops, alpha)

        # Average likelihood and gradients across samples
        likelihood = total_likelihood / samples_per_iter
        gradients = total_gradients / samples_per_iter

        # 5. Convert gradients from d/dalpha to d/dlog_alpha using chain rule
        gradients_log_space = gradients * alpha

        # 6. Add L2 regularization on log_alpha
        reg_gradients = gradients_log_space + 2 * lambda_reg * log_alpha

        # 7. Gradient clipping to improve stability
        grad_norm = np.linalg.norm(reg_gradients)
        if grad_norm > grad_clip_norm:
            reg_gradients = reg_gradients * (grad_clip_norm / (grad_norm + EPSILON))

        # 8. Update log_alpha using Adam
        update, m, v = adam_update(reg_gradients, m, v, iteration, learning_rate)
        log_alpha += update

        # 9. Update alpha and clip to ensure positivity
        alpha = np.exp(log_alpha)
        alpha = np.maximum(alpha, EPSILON)
        log_alpha = np.log(alpha)  # Keep log_alpha in sync
        
        # 10. Apply hierarchical smoothing periodically if enabled
        if use_hierarchical_smoothing and iteration % 10 == 0:
            alpha = apply_hierarchical_smoothing(alpha, neighbor_matrix, smoothing_strength * 0.1)
            log_alpha = np.log(alpha)  # Keep log_alpha in sync

        # Check convergence
        likelihood_change = likelihood - prev_likelihood
        if likelihood_change < 1e-6:
            no_improvement_count += 1
            if no_improvement_count >= 10:
                print(f"\nConverged after {iteration} iterations (no improvement for 10 iterations)")
                break
        else:
            no_improvement_count = 0
        
        prev_likelihood = likelihood
        
        # Print progress every 10 iterations
        if iteration % 10 == 0:
            print(f"  Iteration {iteration}: log-likelihood = {likelihood:.2f}")
    
    return alpha

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MLE estimation of Dirichlet parameters using gradient descent."
    )
    parser.add_argument(
        "--prepared-data", required=True,
        help="Path to prepared data feather file (from prepare_data.py)."
    )
    parser.add_argument(
        "--graph-file", required=True,
        help="Path to graph gpickle file."
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
        "--lambda-reg", type=float, default=1e-4,
        help="L2 regularization strength on log_alpha (default: 1e-4)."
    )
    parser.add_argument(
        "--samples-per-iter", type=int, default=3,
        help="Number of Monte Carlo samples per iteration to reduce gradient noise (default: 3)."
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=5.0,
        help="Gradient norm clip threshold for stability (default: 5.0)."
    )
    parser.add_argument(
        "--init-method", type=str, default="random", choices=["random", "mom", "uniform"],
        help="Initialization method: 'random' (flat Dirichlet prior), 'mom' (Method of Moments), 'uniform' (default: random)."
    )
    parser.add_argument(
        "--mom-warmup-iterations", type=int, default=10,
        help="Number of warmup iterations for Method of Moments initialization (default: 10)."
    )
    parser.add_argument(
        "--use-hierarchical-smoothing", action="store_true",
        help="Apply hierarchical smoothing using neighbor information."
    )
    parser.add_argument(
        "--smoothing-strength", type=float, default=0.1,
        help="Strength of hierarchical smoothing (0-1, default: 0.1)."
    )
    parser.add_argument(
        "--output", default="mle_dirichlet_parameters.csv",
        help="Output CSV path (default: mle_dirichlet_parameters.csv)."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--test-mode", action="store_true",
        help="Test mode: filter to Queens county block groups only."
    )
    args = parser.parse_args()
    
    # Set random seed
    rng = np.random.default_rng(args.seed)
    
    # Load data
    print(f"Loading prepared data from {args.prepared_data}...")
    df = pd.read_feather(args.prepared_data).set_index("AFFGEOID")
    
    # Test mode: filter to Queens county only
    if args.test_mode:
        initial_count = len(df)
        geoid_series = df.index.str.replace('1500000US', '')
        queens_mask = geoid_series.str.startswith('36081')
        df = df[queens_mask]
        print(f"Test mode: Filtered to Queens county: {len(df)} block groups (from {initial_count})")
    
    print(f"Loading graph from {args.graph_file}...")
    with open(args.graph_file, "rb") as f:
        graph = pickle.load(f)
    
    # Match nodes between graph and data
    df.sort_index(inplace=True)
    graph_nodes = {f"1500000US{node}" for node in graph.nodes()}
    df_nodes = set(df.index)
    node_list = sorted(list(graph_nodes.intersection(df_nodes)))
    
    if len(node_list) != len(df):
        print(f"Warning: Mismatch between graph nodes and data nodes. Using {len(node_list)} common nodes.")
        df = df.loc[node_list]
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Build neighborhoods
    neighborhoods = build_neighborhoods(df, graph)
    
    # Ensure neighborhoods are ordered by df.index
    ordered_neighborhoods = {geoid: neighborhoods.get(geoid, [geoid]) for geoid in df.index}
    
    # Run MLE optimization
    alpha = run_mle_optimization(
        df,
        ordered_neighborhoods,
        args.iterations,
        args.learning_rate,
        rng,
        lambda_reg=args.lambda_reg,
        samples_per_iter=args.samples_per_iter,
        grad_clip_norm=args.grad_clip_norm,
        init_method=args.init_method,
        use_hierarchical_smoothing=args.use_hierarchical_smoothing,
        smoothing_strength=args.smoothing_strength,
        mom_warmup_iterations=args.mom_warmup_iterations,
    )
    
    # Format output (wide format: one row per AFFGEOID)
    print("Formatting output...")
    output_data = []
    for i, geoid in enumerate(df.index):
        row = {'AFFGEOID': geoid}
        for j, demo in enumerate(DEMOGRAPHICS):
            row[f'{demo}_alpha_D'] = alpha[i, j, 0]
            row[f'{demo}_alpha_R'] = alpha[i, j, 1]
            row[f'{demo}_alpha_O'] = alpha[i, j, 2]
            row[f'{demo}_alpha_N'] = alpha[i, j, 3]
        output_data.append(row)
    
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(args.output, index=False)
    
    print(f"\nOutput saved to {args.output}")
    print(f"Final DataFrame shape: {output_df.shape}")
    print("\n--- Success! ---")

if __name__ == "__main__":
    main()
