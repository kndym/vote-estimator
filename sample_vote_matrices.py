"""
Gibbs sampling script for generating vote count matrices and probability vectors.

Uses Gibbs sampling to generate vote count matrices for each precinct, then
uses Welford's algorithm to estimate Dirichlet parameters via method of moments.

Input:
    - Prepared data feather file (from prepare_data.py)
    - Graph gpickle file (for neighborhood definition)
Output:
    - CSV with Dirichlet parameters: AFFGEOID, demo, alpha_D, alpha_R, alpha_O, alpha_N
"""
import pandas as pd
import numpy as np
import argparse
import pickle
import networkx as nx
from tqdm import tqdm
from scipy.sparse import csr_matrix

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
    """Format GEOID to AFFGEOID standard."""
    return f"1500000US{geoid}"

def preprocess_data(df):
    """
    Round all demo totals and vote totals to non-negative integers.
    Ensure sum of vote totals equals sum of demo totals.
    
    Args:
        df: DataFrame with cvap_* columns and vote share/probability columns
        
    Returns:
        df: Modified DataFrame with rounded values and votes_* columns
    """
    print("Preprocessing data: rounding to integers and validating sums...")
    
    # Round CVAP columns to integers
    cvap_cols = [f'cvap_{demo}' for demo in DEMOGRAPHICS]
    for col in cvap_cols:
        if col in df.columns:
            df[col] = np.round(df[col]).astype(int).clip(lower=0)
    
    # Ensure cvap_total exists
    if 'cvap_total' not in df.columns:
        df['cvap_total'] = df[cvap_cols].sum(axis=1)
    
    # Calculate votes from shares/probabilities if not already present
    vote_cols = [f'votes_{vtype}' for vtype in VOTE_TYPES]
    if not all(col in df.columns for col in vote_cols):
        print("  Computing vote totals from shares/probabilities...")
        # Try to get votes from share columns
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
    
    # Round vote columns to integers (in case they already existed)
    for col in vote_cols:
        if col in df.columns:
            df[col] = np.round(df[col]).astype(int).clip(lower=0)
    
    # Calculate totals
    df['cvap_total'] = df[cvap_cols].sum(axis=1)
    df['votes_total'] = df[vote_cols].sum(axis=1)
    
    # Ensure vote totals equal CVAP totals for each precinct
    print("  Ensuring vote totals match CVAP totals per precinct...")
    for idx in df.index:
        cvap_total = df.loc[idx, 'cvap_total']
        votes_total = df.loc[idx, 'votes_total']
        
        if votes_total != cvap_total:
            diff = cvap_total - votes_total
            # Adjust votes_N to make totals match
            df.loc[idx, 'votes_N'] = df.loc[idx, 'votes_N'] + diff
            df.loc[idx, 'votes_N'] = max(0, df.loc[idx, 'votes_N'])
    
    # Recalculate votes_total after adjustment
    df['votes_total'] = df[vote_cols].sum(axis=1)
    
    # Verify all precincts have matching totals
    mismatches = (df['votes_total'] != df['cvap_total']).sum()
    if mismatches > 0:
        print(f"  Warning: {mismatches} precincts still have mismatched totals. Adjusting...")
        # Final adjustment: set votes_N to make totals match exactly
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

def initialize_prob_vectors(num_precincts, num_demos, rng, cvap_array=None):
    """
    Initialize probability vectors from flat Dirichlet prior.
    For zero-population demographics, use uniform probabilities.
    
    Args:
        num_precincts: Number of precincts
        num_demos: Number of demographics
        rng: Random number generator
        cvap_array: Optional array of shape (num_precincts, num_demos) to identify zero-population demographics
        
    Returns:
        prob_vectors: Array of shape (num_precincts, num_demos, 4) with probability vectors
    """
    print("Initializing probability vectors from flat Dirichlet prior...")
    
    # Vectorized sampling from flat Dirichlet (all parameters = 1)
    # Sample all at once: (num_precincts * num_demos, 4)
    total_samples = num_precincts * num_demos
    gammas = rng.gamma(1.0, 1.0, size=(total_samples, 4))
    # Normalize each row
    prob_vectors_flat = gammas / gammas.sum(axis=1, keepdims=True)
    # Reshape to (num_precincts, num_demos, 4)
    prob_vectors = prob_vectors_flat.reshape(num_precincts, num_demos, 4)
    
    # Set zero-population demographics to uniform probabilities
    if cvap_array is not None:
        zero_pop_mask = cvap_array == 0  # (num_precincts, num_demos)
        prob_vectors[zero_pop_mask] = 0.25  # Uniform: [0.25, 0.25, 0.25, 0.25]
    
    print(f"  Initialized {total_samples} probability vectors")
    return prob_vectors

def calculate_minimum_counts(total_votes, cvap_data):
    """
    Calculate minimum feasible counts for each cell in the contingency table.
    Uses the formula: min_cell[i,j] = max(0, row_sum[i] + col_sum[j] - total)
    
    Args:
        total_votes: Array of shape (4,) with row sums (vote types)
        cvap_data: Array of shape (5,) with column sums (demographics)
        
    Returns:
        min_matrix: Array of shape (4, 5) with minimum feasible counts
    """
    total = total_votes.sum()
    if total == 0:
        return np.zeros((4, 5), dtype=int)
    
    # Calculate minimums: max(0, row_sum + col_sum - total)
    min_matrix = np.maximum(0, total_votes[:, np.newaxis] + cvap_data[np.newaxis, :] - total)
    return min_matrix.astype(int)

def adjust_for_minimums(total_votes, cvap_data, min_matrix):
    """
    Adjust row and column sums so that minimum counts are zero.
    This makes the problem easier to sample from.
    
    Args:
        total_votes: Array of shape (4,) with row sums
        cvap_data: Array of shape (5,) with column sums
        min_matrix: Array of shape (4, 5) with minimum counts
        
    Returns:
        adjusted_votes: Adjusted row sums
        adjusted_cvap: Adjusted column sums
        allocated_matrix: Matrix with minimum counts already allocated
    """
    # Allocate minimum counts
    allocated_matrix = min_matrix.copy()
    
    # Adjust row and column sums
    adjusted_votes = total_votes - min_matrix.sum(axis=1)
    adjusted_cvap = cvap_data - min_matrix.sum(axis=0)
    
    # Ensure non-negative
    adjusted_votes = np.maximum(adjusted_votes, 0)
    adjusted_cvap = np.maximum(adjusted_cvap, 0)
    
    return adjusted_votes, adjusted_cvap, allocated_matrix

def ipf_adjust(vote_matrix, row_sums, col_sums, max_iter=20):
    """
    Iterative Proportional Fitting to adjust matrix to satisfy row/column constraints.
    Uses IPF to converge to constraints, then rounds and adjusts to satisfy exactly.
    
    Args:
        vote_matrix: Array of shape (4, 5) with vote counts
        row_sums: Array of shape (4,) with target row sums
        col_sums: Array of shape (5,) with target column sums
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
    
    # Adjust to satisfy constraints exactly by distributing rounding errors
    row_totals = vote_matrix.sum(axis=1)
    col_totals = vote_matrix.sum(axis=0)
    row_diff = row_sums - row_totals
    col_diff = col_sums - col_totals
    
    # Distribute row differences proportionally
    for i in range(len(row_sums)):
        if row_diff[i] != 0:
            col_weights = vote_matrix[i, :].astype(float)
            if col_weights.sum() > EPSILON:
                col_weights = col_weights / col_weights.sum()
                adjustments = np.round(col_weights * row_diff[i]).astype(int)
                vote_matrix[i, :] += adjustments
    
    # Distribute column differences proportionally
    col_totals = vote_matrix.sum(axis=0)
    col_diff = col_sums - col_totals
    for j in range(len(col_sums)):
        if col_diff[j] != 0:
            row_weights = vote_matrix[:, j].astype(float)
            if row_weights.sum() > EPSILON:
                row_weights = row_weights / row_weights.sum()
                adjustments = np.round(row_weights * col_diff[j]).astype(int)
                vote_matrix[:, j] += adjustments
    
    # Final clipping to ensure non-negative
    vote_matrix = np.maximum(vote_matrix, 0)
    
    return vote_matrix

def fill_vote_matrix(precinct_idx, prob_vectors, total_votes, cvap_data, rng):
    """
    Fill vote count matrix using direct multinomial sampling + IPF adjustment.
    Much faster than iterative constraint-satisfying sampling.
    
    Args:
        precinct_idx: Index of the precinct
        prob_vectors: Array of shape (num_precincts, num_demos, 4) with probability vectors
        total_votes: Array of shape (4,) with total votes per vote type
        cvap_data: Array of shape (5,) with CVAP per demographic
        rng: Random number generator
        
    Returns:
        vote_matrix: Array of shape (4, 5) with vote counts
    """
    # Calculate and allocate minimum counts
    min_matrix = calculate_minimum_counts(total_votes, cvap_data)
    adjusted_votes, adjusted_cvap, allocated_matrix = adjust_for_minimums(total_votes, cvap_data, min_matrix)
    
    total_remaining = adjusted_votes.sum()
    
    # If no votes to allocate after minimums, return minimum matrix
    if total_remaining == 0:
        return allocated_matrix
    
    # Create joint probability matrix: prob_matrix[vote_type, demo] = prob_vectors[precinct, demo, vote_type]
    # Shape: (4 vote types, 5 demographics)
    prob_matrix = prob_vectors[precinct_idx].T  # (4, 5)
    
    # Weight by adjusted CVAP to create proper distribution
    cell_probs = prob_matrix * adjusted_cvap[np.newaxis, :]
    
    # Normalize to create proper probability distribution
    total_prob = cell_probs.sum()
    if total_prob > EPSILON:
        cell_probs = cell_probs / total_prob
    else:
        # Uniform if all probabilities are zero
        cell_probs = np.ones((4, 5)) / 20.0
    
    # Sample all remaining votes at once using multinomial
    sampled_flat = rng.multinomial(total_remaining, cell_probs.flatten())
    sampled_matrix = sampled_flat.reshape(4, 5)
    
    # Combine with allocated minimums
    vote_matrix = allocated_matrix + sampled_matrix
    
    # Use IPF to adjust to satisfy constraints exactly
    vote_matrix = ipf_adjust(vote_matrix, total_votes, cvap_data, max_iter=10)
    
    return vote_matrix

def build_sparse_neighbor_matrix(neighbor_indices, num_precincts):
    """
    Build sparse matrix for neighborhood aggregation.
    
    Args:
        neighbor_indices: List of lists, neighbor_indices[i] contains indices of neighbors for precinct i
        num_precincts: Number of precincts
        
    Returns:
        neighbor_matrix: Sparse CSR matrix of shape (num_precincts, num_precincts)
                         where neighbor_matrix[i, j] = 1 if j is neighbor of i (including self)
    """
    rows = []
    cols = []
    
    for i, neighbors in enumerate(neighbor_indices):
        valid_neighbors = [n for n in neighbors if n >= 0 and n < num_precincts]
        for j in valid_neighbors:
            rows.append(i)
            cols.append(j)
    
    # Create sparse matrix (1s indicate neighbor relationship)
    neighbor_matrix = csr_matrix((np.ones(len(rows), dtype=float), (rows, cols)), 
                                 shape=(num_precincts, num_precincts))
    
    return neighbor_matrix

def update_prob_vectors(vote_matrices, neighbor_matrix, prob_vectors, cvap_arrays, rng):
    """
    Update probability vectors using sparse matrix operations for neighborhood aggregation.
    Fully vectorized - processes all precincts and demographics at once.
    
    Args:
        vote_matrices: Array of shape (num_precincts, 4, 5) with vote matrices
        neighbor_matrix: Sparse CSR matrix of shape (num_precincts, num_precincts) for neighborhood aggregation
        prob_vectors: Array of shape (num_precincts, num_demos, 4) with current probability vectors
        cvap_arrays: Array of shape (num_precincts, num_demos) with CVAP data
        rng: Random number generator
        
    Returns:
        prob_vectors: Updated probability vectors
    """
    num_precincts = len(vote_matrices)
    num_demos = len(DEMOGRAPHICS)
    
    # Pre-sample all random Dirichlet vectors at once
    total_samples = num_precincts * num_demos
    gammas = rng.gamma(1.0, 1.0, size=(total_samples, 4))
    p_random_all = gammas / gammas.sum(axis=1, keepdims=True)
    p_random_all = p_random_all.reshape(num_precincts, num_demos, 4)
    
    # Vectorized update for all demographics at once
    for demo_idx in range(num_demos):
        # Extract vote counts for this demographic: (num_precincts, 4)
        vote_counts_demo = vote_matrices[:, :, demo_idx].astype(float)  # (num_precincts, 4)
        
        # Aggregate neighbors using sparse matrix multiplication
        # neighbor_matrix @ vote_counts_demo gives sum of neighbor votes for each precinct
        X_all = neighbor_matrix @ vote_counts_demo  # (num_precincts, 4)
        
        # Aggregate neighbor CVAP populations
        cvap_demo = cvap_arrays[:, demo_idx]  # (num_precincts,)
        demo_pop_neighborhood = neighbor_matrix @ cvap_demo  # (num_precincts,)
        
        # Vectorized update for all precincts: (X + p_random) / (demo_pop_neighborhood + 1)
        prob_vectors[:, demo_idx, :] = (X_all + p_random_all[:, demo_idx, :]) / (demo_pop_neighborhood[:, np.newaxis] + 1 + EPSILON)
        
        # Vectorized normalization
        prob_sums = prob_vectors[:, demo_idx, :].sum(axis=1)  # (num_precincts,)
        mask = prob_sums > EPSILON
        prob_vectors[mask, demo_idx, :] = prob_vectors[mask, demo_idx, :] / (prob_sums[mask, np.newaxis] + EPSILON)
        
        # Set zero-population demographics to uniform probabilities
        zero_pop_mask = cvap_arrays[:, demo_idx] == 0
        prob_vectors[zero_pop_mask, demo_idx, :] = 0.25  # Uniform: [0.25, 0.25, 0.25, 0.25]
    
    return prob_vectors

class WelfordState:
    """State for Welford's online algorithm."""
    def __init__(self, num_precincts, num_demos):
        self.count = 0
        self.mean = np.zeros((num_precincts, num_demos, 4))
        self.M2 = np.zeros((num_precincts, num_demos, 4))  # Sum of squared differences
        
    def update(self, prob_vectors):
        """Update Welford state with new probability vectors."""
        self.count += 1
        delta = prob_vectors - self.mean
        self.mean += delta / self.count
        delta2 = prob_vectors - self.mean
        self.M2 += delta * delta2
        
    def get_variance(self):
        """Get variance estimate."""
        if self.count < 2:
            return np.zeros_like(self.M2)
        return self.M2 / (self.count - 1)
    
    def estimate_dirichlet_params(self, cvap_array=None):
        """
        Estimate Dirichlet parameters using method of moments with Welford's algorithm.
        Uses the formula: S_k = μ_k(1-μ_k)/s_k² - 1, then average S_k to get S_total,
        then α_k = S_total * μ_k.
        
        Args:
            cvap_array: Optional array of shape (num_precincts, num_demos) to identify zero-population demographics
        
        Returns:
            alpha: Array of shape (num_precincts, num_demos, 4) with Dirichlet parameters
        """
        variance = self.get_variance()  # (num_precincts, num_demos, 4)
        
        # For each category k, calculate precision S_k = μ_k(1-μ_k)/s_k² - 1
        # Shape: (num_precincts, num_demos, 4)
        mu = self.mean  # (num_precincts, num_demos, 4)
        s_squared = variance  # (num_precincts, num_demos, 4)
        
        # Calculate S_k for each category
        # S_k = μ_k(1-μ_k)/s_k² - 1
        numerator = mu * (1 - mu)  # (num_precincts, num_demos, 4)
        S_k = numerator / (s_squared + EPSILON) - 1  # (num_precincts, num_demos, 4)
        
        # Average S_k across all categories to get S_total for each (precinct, demo)
        # Shape: (num_precincts, num_demos)
        S_total = S_k.mean(axis=2)  # Average across the 4 vote types
        
        # Calculate individual alphas: α_k = S_total * μ_k
        # Broadcast S_total to match mu shape
        alpha = S_total[:, :, np.newaxis] * mu  # (num_precincts, num_demos, 4)
        
        # Handle edge cases where variance is zero or mean is at boundaries
        # If variance is zero or very small, use fallback
        zero_variance_mask = (s_squared < EPSILON).any(axis=2)  # (num_precincts, num_demos)
        boundary_mean_mask = (mu < EPSILON).any(axis=2) | (mu > (1 - EPSILON)).any(axis=2)  # (num_precincts, num_demos)
        invalid_mask = zero_variance_mask | boundary_mean_mask
        
        # For invalid cases, use default concentration
        if invalid_mask.any():
            # Use mean to scale with default concentration parameter
            default_concentration = 4.0  # Flat Dirichlet concentration
            alpha[invalid_mask] = self.mean[invalid_mask] * default_concentration
        
        # Handle negative or zero alphas (can happen if S_k is negative)
        alpha = np.maximum(alpha, EPSILON)
        
        return alpha

def create_precinct_strata(total_votes_array, num_strata=3):
    """
    Create strata based on total vote counts.
    
    Args:
        total_votes_array: Array of shape (num_precincts, 4) with vote totals
        num_strata: Number of strata to create
        
    Returns:
        strata_indices: List of lists, each containing precinct indices for that stratum
        stratum_bounds: List of (min, max) vote totals for each stratum
    """
    total_counts = total_votes_array.sum(axis=1)
    
    # Calculate percentile boundaries
    percentiles = np.linspace(0, 100, num_strata + 1)
    bounds = np.percentile(total_counts, percentiles)
    
    strata_indices = []
    stratum_bounds = []
    
    for i in range(num_strata):
        if i == 0:
            mask = total_counts >= bounds[i]
        else:
            mask = (total_counts >= bounds[i]) & (total_counts < bounds[i+1])
        
        # Sort by total count (descending) within each stratum
        stratum_idx = np.where(mask)[0]
        stratum_totals = total_counts[stratum_idx]
        sorted_order = np.argsort(-stratum_totals)  # Descending order
        strata_indices.append(stratum_idx[sorted_order])
        stratum_bounds.append((bounds[i], bounds[i+1] if i < num_strata - 1 else np.inf))
    
    return strata_indices, stratum_bounds

def run_gibbs_sampling(df, neighborhoods, epochs, burnin, rng):
    """
    Run Gibbs sampling algorithm with heuristics for high-count precincts.
    Optimized with vectorized operations, pre-computed arrays, and stratified processing.
    
    Args:
        df: Prepared DataFrame
        neighborhoods: Dict mapping AFFGEOID to list of neighbor AFFGEOIDs
        epochs: Number of epochs
        burnin: Number of burn-in epochs
        rng: Random number generator
        
    Returns:
        dirichlet_params: Array of shape (num_precincts, num_demos, 4) with Dirichlet parameters
    """
    num_precincts = len(df)
    num_demos = len(DEMOGRAPHICS)
    
    print(f"Starting Gibbs sampling: {epochs} epochs ({burnin} burn-in)...")
    
    # Pre-compute arrays (vectorized, avoiding iterrows)
    total_votes_array = np.array([df[f'votes_{vtype}'].values for vtype in VOTE_TYPES], dtype=int).T  # (num_precincts, 4)
    cvap_array = np.array([df[f'cvap_{demo}'].values for demo in DEMOGRAPHICS], dtype=int).T  # (num_precincts, 5)
    
    # Initialize probability vectors (with zero-population handling)
    prob_vectors = initialize_prob_vectors(num_precincts, num_demos, rng, cvap_array)
    
    # Initialize Welford state
    welford_state = WelfordState(num_precincts, num_demos)
    
    # Build neighbor indices (list of lists of indices)
    geoid_to_idx = {geoid: idx for idx, geoid in enumerate(df.index)}
    neighbor_indices = []
    for geoid in df.index:
        neighbors = neighborhoods.get(geoid, [geoid])
        neighbor_indices.append([geoid_to_idx.get(n, -1) for n in neighbors if n in geoid_to_idx])
    
    # Build sparse neighborhood matrix for vectorized operations
    print("Building sparse neighborhood matrix...")
    neighbor_matrix = build_sparse_neighbor_matrix(neighbor_indices, num_precincts)
    print(f"  Sparse matrix shape: {neighbor_matrix.shape}, nnz: {neighbor_matrix.nnz}")
    
    # Create strata based on total vote counts
    print("Creating precinct strata by total vote counts...")
    strata_indices, stratum_bounds = create_precinct_strata(total_votes_array, num_strata=3)
    print(f"  Stratum 1 (high): {len(strata_indices[0])} precincts, totals >= {stratum_bounds[0][0]:.0f}")
    print(f"  Stratum 2 (medium): {len(strata_indices[1])} precincts, {stratum_bounds[1][0]:.0f} <= totals < {stratum_bounds[1][1]:.0f}")
    print(f"  Stratum 3 (low): {len(strata_indices[2])} precincts, totals < {stratum_bounds[2][0]:.0f}")
    
    # Create processing order: high-count first, then medium, then low
    processing_order = []
    for stratum in strata_indices:
        processing_order.extend(stratum.tolist())
    
    # Main loop
    for epoch in tqdm(range(epochs), desc="Gibbs sampling"):
        # Step 1: Fill all matrices (process high-count precincts first)
        vote_matrices = np.zeros((num_precincts, 4, 5), dtype=int)
        for i in processing_order:
            vote_matrix = fill_vote_matrix(i, prob_vectors, total_votes_array[i], cvap_array[i], rng)
            vote_matrices[i] = vote_matrix
        
        # Step 2: Update probability vectors using sparse matrix operations
        prob_vectors = update_prob_vectors(vote_matrices, neighbor_matrix, prob_vectors, 
                                          cvap_array, rng)
        
        # Step 3: Update Welford state (only after burn-in)
        if epoch >= burnin:
            welford_state.update(prob_vectors)
    
    # Estimate Dirichlet parameters (with zero-population handling)
    print("Estimating Dirichlet parameters from Welford state...")
    dirichlet_params = welford_state.estimate_dirichlet_params(cvap_array)
    
    return dirichlet_params

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gibbs sampling for vote count matrices and Dirichlet parameter estimation."
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
        "--epochs", type=int, default=1000,
        help="Number of epochs (default: 1000)."
    )
    parser.add_argument(
        "--burnin", type=int, default=100,
        help="Number of burn-in epochs (default: 100)."
    )
    parser.add_argument(
        "--output", default="dirichlet_parameters.csv",
        help="Output CSV path (default: dirichlet_parameters.csv)."
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
        # Extract GEOID part (after "1500000US")
        geoid_series = df.index.str.replace('1500000US', '')
        # Filter to Queens (starts with 36081)
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
    
    # Run Gibbs sampling
    dirichlet_params = run_gibbs_sampling(df, ordered_neighborhoods, args.epochs, args.burnin, rng)
    
    # Format output (wide format: one row per AFFGEOID)
    print("Formatting output...")
    output_data = []
    for i, geoid in enumerate(df.index):
        row = {'AFFGEOID': geoid}
        for j, demo in enumerate(DEMOGRAPHICS):
            row[f'{demo}_alpha_D'] = dirichlet_params[i, j, 0]
            row[f'{demo}_alpha_R'] = dirichlet_params[i, j, 1]
            row[f'{demo}_alpha_O'] = dirichlet_params[i, j, 2]
            row[f'{demo}_alpha_N'] = dirichlet_params[i, j, 3]
        output_data.append(row)
    
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(args.output, index=False)
    
    print(f"\nOutput saved to {args.output}")
    print(f"Final DataFrame shape: {output_df.shape}")
    print("\n--- Success! ---")

if __name__ == "__main__":
    main()

