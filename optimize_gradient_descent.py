"""
Gradient descent optimization for ecological inference in probability space.

Works directly with vote count matrices (4 vote types × 5 demographics) for each block group.
Uses fundamental circuit basis matrices (cycles) to maintain sum constraints and ensure non-negativity.

- Input:
    - Prepared data file (feather) from prepare_data.py
    - Graph file (gpickle) from graph.py
- Output:
    - CSV file with vote count estimates for each demographic and vote type
"""
import pandas as pd
import numpy as np
import argparse
import pickle
import networkx as nx
from scipy.sparse import csr_matrix
from tqdm import tqdm
import sys

# =============================================================================
# CONSTANTS AND HELPER FUNCTIONS
# =============================================================================

EPSILON = 1e-12

def format_affgeoid(geoid):
    """Format GEOID to AFFGEOID standard."""
    return f"1500000US{geoid}"

def compute_probabilities_from_votes(votes_matrix, cvap_array):
    """
    Compute probability matrix from vote count matrix.
    
    Args:
        votes_matrix: (4, 5) array of vote counts [D, R, O, N] × [Wht, His, Blk, Asn, Oth]
        cvap_array: (5,) array of CVAP populations for each demographic
        
    Returns:
        prob_matrix: (4, 5) array of probabilities
    """
    prob_matrix = np.zeros_like(votes_matrix)
    for j in range(5):
        if cvap_array[j] > EPSILON:
            prob_matrix[:, j] = votes_matrix[:, j] / cvap_array[j]
        else:
            prob_matrix[:, j] = 0.0
    return prob_matrix

# =============================================================================
# BASIS MATRIX PROJECTION
# =============================================================================

def create_basis_matrices(n=4, m=5):
    """
    Create all fundamental circuit basis matrices for n×m contingency table.
    
    For a table with n rows (vote types) and m columns (demographics), we have 
    k = (n-1)(m-1) basis matrices. Each basis matrix E(i,j) for 1 ≤ i < n and 
    1 ≤ j < m maintains row/column sum constraints.
    
    Each E(i,j) has:
    - +1 at (i, j)
    - -1 at (i, m-1) [last column]
    - -1 at (n-1, j) [last row]
    - +1 at (n-1, m-1) [bottom-right corner]
    
    Args:
        n: Number of vote types (D, R, O, N) = 4, where N = non-voter (last row)
        m: Number of demographics (Wht, His, Blk, Asn, Oth) = 5
        
    Returns:
        basis_matrices: List of k basis matrices, each of shape (n, m)
    """
    basis_matrices = []
    
    # Vote type indices: 0=D, 1=R, 2=O, 3=N (N = non-voter is last row, n-1 = 3)
    # Demographic indices: 0=Wht, 1=His, 2=Blk, 3=Asn, 4=Oth (m-1 = 4, so j = 0, 1, 2, 3)
    
    for i in range(n - 1):  # i = 0, 1, 2 (D, R, O)
        for j in range(m - 1):  # j = 0, 1, 2, 3 (Wht, His, Blk, Asn)
            # Create basis matrix E(i,j)
            E = np.zeros((n, m))
            
            # +1 at (i, j)
            E[i, j] = +1
            
            # -1 at (i, m-1) = (i, Oth)
            E[i, m-1] = -1
            
            # -1 at (n-1, j) = (N, j) where N is non-voter (last row)
            E[n-1, j] = -1
            
            # +1 at (n-1, m-1) = (N, Oth)
            E[n-1, m-1] = +1
            
            basis_matrices.append(E)
    
    return basis_matrices

def project_gradient_matrix_onto_basis(gradient_matrix, basis_matrices):
    """
    Project a gradient matrix onto the basis space spanned by fundamental circuits.
    
    Args:
        gradient_matrix: (n, m) array of gradients for all vote types × demographics
        basis_matrices: List of k basis matrices, each (n, m)
        
    Returns:
        projected_gradient: (n, m) array projected onto basis space
    """
    # Flatten gradient and basis matrices
    g_flat = gradient_matrix.flatten()  # Shape: (n*m,)
    
    # Create basis matrix as columns (each basis matrix flattened as a column)
    B = np.array([E.flatten() for E in basis_matrices]).T  # Shape: (n*m, k)
    
    # Project: g_proj = B @ (B.T @ g)
    # First compute coefficients: c = B.T @ g
    c = B.T @ g_flat  # Shape: (k,)
    
    # Then reconstruct: g_proj = B @ c
    g_proj_flat = B @ c  # Shape: (n*m,)
    
    # Reshape back to (n, m)
    return g_proj_flat.reshape(gradient_matrix.shape)


def calculate_adaptive_bounds(votes_matrix, basis_matrix, total_votes, cvap_array):
    """
    Calculate adaptive bounds for cycle coefficient to maintain constraints.
    
    Args:
        votes_matrix: Current (4, 5) vote count matrix
        basis_matrix: (4, 5) fundamental circuit basis matrix
        total_votes: (4,) array of total votes for each vote type (row sums)
        cvap_array: (5,) array of CVAP populations (column sums)
        
    Returns:
        min_coeff, max_coeff: Bounds for cycle coefficient
    """
    # Upper bounds: votes[i,j] <= min(total_votes[i], cvap[j])
    upper_bounds = np.minimum(
        total_votes[:, np.newaxis],
        cvap_array[np.newaxis, :]
    )
    
    # For each cell, calculate bounds from:
    # 1. Non-negativity: votes[i,j] + coeff * basis[i,j] >= 0
    # 2. Upper bound: votes[i,j] + coeff * basis[i,j] <= upper_bounds[i,j]
    
    # Initialize bounds
    min_coeff = -np.inf
    max_coeff = np.inf
    
    # Iterate over all cells
    for i in range(4):
        for j in range(5):
            basis_val = basis_matrix[i, j]
            votes_val = votes_matrix[i, j]
            upper_bound = upper_bounds[i, j]
            
            if abs(basis_val) < EPSILON:
                # Zero basis value, no constraint from this cell
                continue
            
            # Non-negativity constraint: votes_val + coeff * basis_val >= 0
            if basis_val > 0:
                # For positive basis: coeff >= -votes_val / basis_val
                min_coeff = max(min_coeff, -votes_val / basis_val)
                # Upper bound: votes_val + coeff * basis_val <= upper_bound
                # => coeff <= (upper_bound - votes_val) / basis_val
                max_coeff = min(max_coeff, (upper_bound - votes_val) / basis_val)
            else:  # basis_val < 0
                # For negative basis: coeff <= -votes_val / basis_val (which is positive)
                max_coeff = min(max_coeff, -votes_val / basis_val)
                # Upper bound: votes_val + coeff * basis_val <= upper_bound
                # => coeff >= (upper_bound - votes_val) / basis_val (which is negative)
                min_coeff = max(min_coeff, (upper_bound - votes_val) / basis_val)
    
    # Ensure reasonable bounds (fallback if calculation failed)
    if not np.isfinite(min_coeff):
        min_coeff = -1.0
    if not np.isfinite(max_coeff):
        max_coeff = 1.0
    
    # Clip to reasonable range
    min_coeff = max(min_coeff, -100.0)
    max_coeff = min(max_coeff, 100.0)
    
    # Ensure min < max
    if min_coeff >= max_coeff:
        min_coeff = -0.1
        max_coeff = 0.1
    
    return min_coeff, max_coeff

# =============================================================================
# COST FUNCTION
# =============================================================================

def compute_cost(
    vote_matrices_array,
    edge_list,
    cvap_arrays,
    spatial_weight
):
    """
    Compute spatial cost: minimize probability vector differences between neighbors.
    
    Args:
        vote_matrices_array: (num_nodes, 4, 5) array of vote count matrices
        edge_list: Pre-computed edge list as (num_edges, 2) array
        cvap_arrays: Dict of CVAP population arrays for each demographic
        spatial_weight: Weight for spatial smoothness term (not used, kept for compatibility)
        
    Returns:
        total_cost, cost_breakdown dict
    """
    demographics = ['Wht', 'His', 'Blk', 'Asn', 'Oth']
    num_nodes = vote_matrices_array.shape[0]
    
    spatial_cost = 0.0
    
    # For each demographic, compute probability vectors and compare neighbors
    for demo_idx, demo in enumerate(demographics):
        cvap_demo = cvap_arrays[demo]  # (num_nodes,) array
        
        # Vectorized probability computation: (num_nodes, 4) array
        # votes[:, :, demo_idx] extracts all vote counts for this demographic: (num_nodes, 4)
        cvap_demo_safe = np.maximum(cvap_demo, EPSILON)
        prob_vectors = vote_matrices_array[:, :, demo_idx] / cvap_demo_safe[:, np.newaxis]
        
        # Vectorized edge processing
        i_indices = edge_list[:, 0]
        j_indices = edge_list[:, 1]
        
        # Weight by geometric mean of CVAP populations
        weights = np.sqrt(cvap_demo[i_indices] * cvap_demo[j_indices])
        
        # Probability vector differences: (num_edges, 4)
        prob_diffs = prob_vectors[i_indices, :] - prob_vectors[j_indices, :]
        
        # Sum of squared differences for each edge: (num_edges,)
        prob_diff_sq_sum = np.sum(prob_diffs ** 2, axis=1)
        
        # Weighted sum
        spatial_cost += np.sum(weights * prob_diff_sq_sum)
    
    cost_breakdown = {
        'spatial': spatial_cost,
        'total': spatial_cost
    }
    
    return spatial_cost, cost_breakdown

# =============================================================================
# GRADIENT COMPUTATION
# =============================================================================

def compute_gradients(
    vote_matrices_array,
    edge_list,
    cvap_arrays,
    spatial_weight
):
    """
    Compute gradients of cost function w.r.t. vote counts.
    
    Args:
        vote_matrices_array: (num_nodes, 4, 5) array of vote count matrices
        edge_list: Pre-computed edge list as (num_edges, 2) array
        cvap_arrays: Dict of CVAP population arrays for each demographic
        spatial_weight: Weight for spatial smoothness term
        
    Returns:
        gradient_matrices_array: (num_nodes, 4, 5) array of gradient matrices
    """
    demographics = ['Wht', 'His', 'Blk', 'Asn', 'Oth']
    num_nodes = vote_matrices_array.shape[0]
    
    # Initialize gradient matrices
    gradient_matrices_array = np.zeros_like(vote_matrices_array)
    
    # Compute gradients for spatial cost
    for demo_idx, demo in enumerate(demographics):
        cvap_demo = cvap_arrays[demo]  # (num_nodes,) array
        
        # Vectorized probability computation: (num_nodes, 4) array
        cvap_demo_safe = np.maximum(cvap_demo, EPSILON)
        prob_vectors = vote_matrices_array[:, :, demo_idx] / cvap_demo_safe[:, np.newaxis]
        
        # Vectorized edge processing
        i_indices = edge_list[:, 0]
        j_indices = edge_list[:, 1]
        
        # Weight by geometric mean of CVAP populations
        weights = np.sqrt(cvap_demo[i_indices] * cvap_demo[j_indices])
        
        # Probability vector differences: (num_edges, 4)
        prob_diffs = prob_vectors[i_indices, :] - prob_vectors[j_indices, :]
        
        # Gradients for node i: (num_edges, 4)
        cvap_i_safe = np.maximum(cvap_demo[i_indices], EPSILON)
        grad_i_all = 2 * weights[:, np.newaxis] * prob_diffs / cvap_i_safe[:, np.newaxis]
        
        # Gradients for node j: (num_edges, 4)
        cvap_j_safe = np.maximum(cvap_demo[j_indices], EPSILON)
        grad_j_all = -2 * weights[:, np.newaxis] * prob_diffs / cvap_j_safe[:, np.newaxis]
        
        # Accumulate gradients using advanced indexing
        # For each edge, add to gradient matrices
        np.add.at(gradient_matrices_array, (i_indices, slice(None), demo_idx), grad_i_all)
        np.add.at(gradient_matrices_array, (j_indices, slice(None), demo_idx), grad_j_all)
    
    return gradient_matrices_array

# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_vote_matrices(df, total_votes_dict, cvap_arrays):
    """
    Initialize vote count matrices proportionally.
    
    Args:
        df: DataFrame with block group data
        total_votes_dict: Dict with keys 'D', 'R', 'O', 'N', each (num_nodes,) array
        cvap_arrays: Dict of CVAP population arrays for each demographic
        
    Returns:
        vote_matrices_array: (num_nodes, 4, 5) array of vote count matrices
    """
    demographics = ['Wht', 'His', 'Blk', 'Asn', 'Oth']
    vote_types = ['D', 'R', 'O', 'N']
    num_nodes = len(df)
    
    # Create 3D array: (num_nodes, 4, 5)
    vote_matrices_array = np.zeros((num_nodes, 4, 5))
    
    # Get cvap_total as array
    cvap_total = df['cvap_total'].values  # (num_nodes,)
    
    # Vectorized initialization
    cvap_total_safe = np.maximum(cvap_total, EPSILON)
    
    for vtype_idx, vtype in enumerate(vote_types):
        total_votes_vtype = total_votes_dict[vtype]  # (num_nodes,)
        for demo_idx, demo in enumerate(demographics):
            cvap_demo = cvap_arrays[demo]  # (num_nodes,)
            # Proportional initialization: votes[i,j] = (total_votes[i] × cvap[j]) / cvap_total
            vote_matrices_array[:, vtype_idx, demo_idx] = (total_votes_vtype * cvap_demo) / cvap_total_safe
    
    return vote_matrices_array

def perturb_vote_matrices(vote_matrices_array, basis_matrices, total_votes_dict, cvap_arrays, perturbation_scale=0.1):
    """
    Apply random perturbation to vote matrices using cycles with adaptive bounds.
    
    Args:
        vote_matrices_array: (num_nodes, 4, 5) array of vote count matrices
        basis_matrices: List of fundamental circuit basis matrices
        total_votes_dict: Dict with total votes for each vote type
        cvap_arrays: Dict of CVAP population arrays
        perturbation_scale: Scale factor for perturbation (multiplies the adaptive bounds)
        
    Returns:
        perturbed_vote_matrices_array: (num_nodes, 4, 5) array of perturbed vote count matrices
    """
    num_nodes = vote_matrices_array.shape[0]
    
    # Convert total_votes_dict to arrays
    vote_types = ['D', 'R', 'O', 'N']
    total_votes_array = np.array([total_votes_dict[vtype] for vtype in vote_types])  # (4, num_nodes)
    
    demographics = ['Wht', 'His', 'Blk', 'Asn', 'Oth']
    cvap_array = np.array([cvap_arrays[demo] for demo in demographics])  # (5, num_nodes)
    
    # Copy to avoid modifying original
    perturbed_vote_matrices_array = vote_matrices_array.copy()
    
    for node_idx in range(num_nodes):
        votes = perturbed_vote_matrices_array[node_idx, :, :]  # (4, 5)
        total_votes = total_votes_array[:, node_idx]  # (4,)
        cvap = cvap_array[:, node_idx]  # (5,)
        
        # Apply perturbation using each basis matrix
        for basis_matrix in basis_matrices:
            # Calculate adaptive bounds
            min_coeff, max_coeff = calculate_adaptive_bounds(
                votes, basis_matrix, total_votes, cvap
            )
            
            # Scale bounds by perturbation_scale
            scaled_min = min_coeff * perturbation_scale
            scaled_max = max_coeff * perturbation_scale
            
            # Sample uniform random coefficient
            coeff = np.random.uniform(scaled_min, scaled_max)
            
            # Apply perturbation
            votes += coeff * basis_matrix
        
        # Clip to ensure bounds
        upper_bounds = np.minimum(
            total_votes[:, np.newaxis],
            cvap[np.newaxis, :]
        )
        votes = np.clip(votes, 0.0, upper_bounds)
        
        perturbed_vote_matrices_array[node_idx, :, :] = votes
    
    return perturbed_vote_matrices_array

# =============================================================================
# OPTIMIZATION LOOP
# =============================================================================

def run_optimization(
    df,
    adj_matrix,
    learning_rate,
    spatial_weight,
    iterations,
    convergence_threshold,
    output_dir,
    diagnostic_mode=False,
    diagnostic_interval=10,
    perturbation_scale=0.1
):
    """
    Run gradient descent optimization with vote count matrices.
    
    Args:
        df: Prepared DataFrame
        adj_matrix: Sparse adjacency matrix
        learning_rate: Step size for gradient descent
        spatial_weight: Weight for spatial smoothness (not used, kept for compatibility)
        iterations: Maximum number of iterations
        convergence_threshold: Stop if cost change < threshold
        output_dir: Directory for output files
        diagnostic_mode: If True, collect detailed diagnostics
        diagnostic_interval: Print diagnostics every N iterations
        perturbation_scale: Scale for random perturbation
        
    Returns:
        final_vote_matrices: List of final vote count matrices
        history: List of cost values per iteration
        diagnostic_history: List of diagnostic entries (if diagnostic_mode=True)
    """
    demographics = ['Wht', 'His', 'Blk', 'Asn', 'Oth']
    vote_types = ['D', 'R', 'O', 'N']
    num_nodes = len(df)
    
    print("Initializing optimization...")
    
    # Pre-compute arrays
    cvap_arrays = {demo: df[f"cvap_{demo}"].values for demo in demographics}
    
    # Get total votes for each vote type
    total_votes_dict = {}
    for vtype in vote_types:
        if f"votes_{vtype}" in df.columns:
            total_votes_dict[vtype] = df[f"votes_{vtype}"].values
        elif f"real_{vtype}_share" in df.columns:
            # Compute from share
            total_votes_dict[vtype] = df[f"real_{vtype}_share"].values * df["cvap_total"].values
            # Ensure non-negative (especially important for 'O' which is computed as residual)
            total_votes_dict[vtype] = np.maximum(total_votes_dict[vtype], 0.0)
        else:
            raise ValueError(f"Could not find votes_{vtype} or real_{vtype}_share in DataFrame")
    
    # Initialize vote count matrices proportionally
    print("Initializing vote count matrices proportionally...")
    vote_matrices = initialize_vote_matrices(df, total_votes_dict, cvap_arrays)
    
    # Apply random perturbation
    print(f"Applying random perturbation (scale={perturbation_scale})...")
    basis_matrices = create_basis_matrices(n=4, m=5)
    vote_matrices = perturb_vote_matrices(
        vote_matrices, basis_matrices, total_votes_dict, cvap_arrays, perturbation_scale
    )
    
    print(f"Created {len(basis_matrices)} fundamental circuit basis matrices (k = (4-1)(5-1) = 12)")
    
    # Pre-compute edge list once (only upper triangle to avoid double counting)
    print("Pre-computing edge list...")
    adj_matrix_coo = adj_matrix.tocoo()
    edge_mask = adj_matrix_coo.row < adj_matrix_coo.col
    edge_list = np.column_stack([adj_matrix_coo.row[edge_mask], adj_matrix_coo.col[edge_mask]])
    print(f"Found {len(edge_list)} unique edges")
    
    # Pre-compute basis matrix for faster projection
    print("Pre-computing basis matrix...")
    B = np.array([E.flatten() for E in basis_matrices]).T  # Shape: (n*m, k)
    
    # Pre-compute arrays for bounds checking
    vote_types = ['D', 'R', 'O', 'N']
    total_votes_array = np.array([total_votes_dict[vtype] for vtype in vote_types])  # (4, num_nodes)
    demographics = ['Wht', 'His', 'Blk', 'Asn', 'Oth']
    cvap_array = np.array([cvap_arrays[demo] for demo in demographics])  # (5, num_nodes)
    
    # Optimization loop
    history = []
    diagnostic_history = [] if diagnostic_mode else None
    prev_cost = None
    prev_cost_change = None
    
    print(f"Starting optimization for {iterations} iterations...")
    print(f"Parameters: learning_rate={learning_rate}, spatial_weight={spatial_weight}")
    if diagnostic_mode:
        print(f"Diagnostic mode enabled (printing every {diagnostic_interval} iterations)")
    
    for i in tqdm(range(iterations), desc="Optimizing"):
        # Compute cost
        cost, cost_breakdown = compute_cost(
            vote_matrices, edge_list, cvap_arrays, spatial_weight
        )
        
        # Calculate cost change
        cost_change = abs(prev_cost - cost) if prev_cost is not None else None
        cost_change_rate = None
        if prev_cost is not None and prev_cost_change is not None:
            cost_change_rate = cost_change / (prev_cost_change + EPSILON) if prev_cost_change > EPSILON else None
        
        history.append({
            'iteration': i + 1,
            **cost_breakdown
        })
        
        # Check convergence
        if cost_change is not None and cost_change < convergence_threshold:
            print(f"\nConverged at iteration {i+1}: cost change = {cost_change:.2e}")
            break
        
        prev_cost = cost
        prev_cost_change = cost_change if cost_change is not None else None
        
        # Compute gradients
        gradient_matrices = compute_gradients(
            vote_matrices, edge_list, cvap_arrays, spatial_weight
        )
        
        # Collect diagnostic information if enabled
        if diagnostic_mode:
            # Compute gradient norms (vectorized)
            all_gradients = gradient_matrices.flatten()
            gradient_norm_max = np.max(np.abs(all_gradients))
            gradient_norm_mean = np.mean(np.abs(all_gradients))
            gradient_norm_std = np.std(np.abs(all_gradients))
            
            # Compute step sizes (vectorized)
            step_sizes = np.abs(learning_rate * all_gradients)
            step_size_mean = np.mean(step_sizes)
            
            # Constraint violations (vectorized)
            # Check row sums: (num_nodes, 4)
            row_sums = vote_matrices.sum(axis=2)  # (num_nodes, 4)
            row_violations = np.abs(row_sums - total_votes_array.T)  # (num_nodes, 4)
            
            # Check column sums: (num_nodes, 5)
            col_sums = vote_matrices.sum(axis=1)  # (num_nodes, 5)
            col_violations = np.abs(col_sums - cvap_array.T)  # (num_nodes, 5)
            
            constraint_violations_max = max(np.max(row_violations), np.max(col_violations))
            constraint_violations_mean = (np.mean(row_violations) + np.mean(col_violations)) / 2.0
            
            diagnostic_entry = {
                'iteration': i + 1,
                'cost_total': cost,
                'cost_spatial': cost_breakdown['spatial'],
                'cost_change': cost_change if cost_change is not None else 0.0,
                'cost_change_rate': cost_change_rate if cost_change_rate is not None else 0.0,
                'gradient_norm_max': gradient_norm_max,
                'gradient_norm_mean': gradient_norm_mean,
                'gradient_norm_std': gradient_norm_std,
                'step_size_mean': step_size_mean,
                'constraint_violations_max': constraint_violations_max,
                'constraint_violations_mean': constraint_violations_mean,
            }
            diagnostic_history.append(diagnostic_entry)
            
            # Print diagnostics at specified interval
            if (i + 1) % diagnostic_interval == 0 or i == 0:
                print(f"\n--- Iteration {i+1} Diagnostics ---")
                print(f"Cost: total={cost:.6f}, spatial={cost_breakdown['spatial']:.6f}")
                print(f"Cost change: {cost_change:.6e}" if cost_change is not None else "Cost change: N/A")
                print(f"Gradient norms: max={gradient_norm_max:.6f}, mean={gradient_norm_mean:.6f}, std={gradient_norm_std:.6f}")
                print(f"Step size mean: {step_size_mean:.6e}")
                print(f"Constraint violations: max={constraint_violations_max:.6e}, mean={constraint_violations_mean:.6e}")
        
        # Project gradients onto basis and update using fundamental circuits (vectorized)
        # Flatten all gradient matrices: (num_nodes, 20)
        g_flat_all = gradient_matrices.reshape(num_nodes, -1)  # (num_nodes, 20)
        
        # Project all gradients: g_proj = B @ (B.T @ g)
        # First compute coefficients: c = B.T @ g for all nodes: (k, num_nodes)
        c_all = B.T @ g_flat_all.T  # (k, num_nodes)
        # Then reconstruct: g_proj = B @ c: (n*m, num_nodes)
        g_proj_flat_all = (B @ c_all).T  # (num_nodes, n*m)
        
        # Reshape back to (num_nodes, 4, 5) and update
        g_proj_all = g_proj_flat_all.reshape(num_nodes, 4, 5)
        vote_matrices -= learning_rate * g_proj_all
        
        # Vectorized clipping to maintain bounds
        upper_bounds = np.minimum(
            total_votes_array[:, np.newaxis, :],  # (4, 1, num_nodes)
            cvap_array[np.newaxis, :, :]  # (1, 5, num_nodes)
        )  # (4, 5, num_nodes)
        upper_bounds = np.transpose(upper_bounds, (2, 0, 1))  # (num_nodes, 4, 5)
        # Ensure upper bounds are non-negative (safety measure)
        upper_bounds = np.maximum(upper_bounds, 0.0)
        vote_matrices = np.clip(vote_matrices, 0.0, upper_bounds)
        
        # Print progress every 100 iterations (if not in diagnostic mode or not at diagnostic interval)
        if not diagnostic_mode and (i + 1) % 100 == 0:
            print(f"\nIteration {i+1}: cost = {cost:.6f} (spatial={cost_breakdown['spatial']:.6f})")
    
    print("\nOptimization complete.")
    return vote_matrices, history, diagnostic_history

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gradient descent optimization for ecological inference in probability space."
    )
    parser.add_argument(
        "prepared_data_path", help="Path to the prepared input .feather file."
    )
    parser.add_argument(
        "graph_file", help="Path to the blockgroups_graph.gpickle file."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.01, help="Learning rate for gradient descent."
    )
    parser.add_argument(
        "--spatial-weight", type=float, default=1.0, help="Weight for spatial smoothness term (not used, kept for compatibility)."
    )
    parser.add_argument(
        "--iterations", type=int, default=1000, help="Maximum number of iterations."
    )
    parser.add_argument(
        "--convergence-threshold", type=float, default=1e-6, help="Convergence threshold for cost change."
    )
    parser.add_argument(
        "--output", default="estimates.csv", help="Path for output CSV file."
    )
    parser.add_argument(
        "--diagnostic-mode", action="store_true", help="Enable comprehensive diagnostic tracking."
    )
    parser.add_argument(
        "--diagnostic-output", default="diagnostics.csv", help="Path for diagnostic output CSV file."
    )
    parser.add_argument(
        "--diagnostic-interval", type=int, default=10, help="Print diagnostics every N iterations."
    )
    parser.add_argument(
        "--perturbation-scale", type=float, default=1, help="Scale factor for random perturbation (0.0 to 1.0)."
    )
    args = parser.parse_args()
    
    # Load data
    print(f"Loading prepared data from {args.prepared_data_path}...")
    try:
        df = pd.read_feather(args.prepared_data_path).set_index("AFFGEOID")
    except Exception as e:
        print(f"Error loading data file: {e}. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    # Load graph and build adjacency matrix
    print(f"Loading graph from {args.graph_file}...")
    try:
        with open(args.graph_file, "rb") as f:
            G = pickle.load(f)
    except Exception as e:
        print(f"Error loading graph file: {e}. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    # Match nodes between graph and data
    df.sort_index(inplace=True)
    graph_nodes = {f"1500000US{node}" for node in G.nodes()}
    df_nodes = set(df.index)
    node_list = sorted(list(graph_nodes.intersection(df_nodes)))
    
    if len(node_list) != len(df):
        print(f"Warning: Mismatch between graph nodes and data nodes. Using {len(node_list)} common nodes.")
        df = df.loc[node_list]
    
    print(f"Creating sparse adjacency matrix for {len(node_list)} nodes...")
    adj_matrix = nx.to_scipy_sparse_array(
        G, nodelist=[geoid.replace("1500000US", "") for geoid in node_list], format="csr"
    )
    
    # Run optimization
    vote_matrices, history, diagnostic_history = run_optimization(
        df=df,
        adj_matrix=adj_matrix,
        learning_rate=args.learning_rate,
        spatial_weight=args.spatial_weight,
        iterations=args.iterations,
        convergence_threshold=args.convergence_threshold,
        output_dir=".",
        diagnostic_mode=args.diagnostic_mode,
        diagnostic_interval=args.diagnostic_interval,
        perturbation_scale=args.perturbation_scale
    )
    
    # Save diagnostics if enabled
    if args.diagnostic_mode and diagnostic_history:
        diagnostic_df = pd.DataFrame(diagnostic_history)
        diagnostic_df.to_csv(args.diagnostic_output, index=False)
        print(f"\nDiagnostics saved to {args.diagnostic_output}")
    
    # Convert vote matrices to output format
    print("Converting vote matrices to output format...")
    demographics = ['Wht', 'His', 'Blk', 'Asn', 'Oth']
    vote_types = ['D', 'R', 'O', 'N']
    
    output_data = {'AFFGEOID': df.index.values}
    
    # Convert vote_matrices back to list format for output (if needed)
    # Actually, we can work directly with the array
    # Add vote counts for each vote type and demographic (vectorized)
    for vtype_idx, vtype in enumerate(vote_types):
        for demo_idx, demo in enumerate(demographics):
            output_data[f'votes_{vtype}_{demo}'] = vote_matrices[:, vtype_idx, demo_idx]
    
    # Also add probabilities for convenience (vectorized)
    cvap_arrays = {demo: df[f"cvap_{demo}"].values for demo in demographics}
    for vtype_idx, vtype in enumerate(vote_types):
        for demo_idx, demo in enumerate(demographics):
            votes_array = vote_matrices[:, vtype_idx, demo_idx]
            cvap_demo = cvap_arrays[demo]
            prob_array = np.where(cvap_demo > EPSILON, votes_array / cvap_demo, 0.0)
            output_data[f'prob_{vtype}_{demo}'] = prob_array
    
    # Save to CSV
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(args.output, index=False)
    print(f"\nOutput saved to {args.output}")
    print(f"Final DataFrame shape: {output_df.shape}")
    
    # Print final cost
    if history:
        final_cost = history[-1]['total']
        print(f"Final cost: {final_cost:.6f}")
        print(f"  - Spatial: {history[-1]['spatial']:.6f}")

if __name__ == "__main__":
    main()
