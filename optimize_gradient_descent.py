"""
Gradient descent optimization for ecological inference with constraint maintenance.

Uses basis matrix projection to maintain probability sum constraints while
optimizing logit values to match aggregate vote shares and spatial smoothness.

- Input:
    - Prepared data file (feather) from prepare_data.py
    - Graph file (gpickle) from graph.py
- Output:
    - CSV file with probability estimates for each demographic and vote type
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

def logit(p):
    """Convert probability to logit."""
    p = np.clip(p, EPSILON, 1 - EPSILON)
    return np.log(p / (1 - p))

def un_logit(x):
    """Convert logit to probability."""
    return 1 / (1 + np.exp(-x))

def format_affgeoid(geoid):
    """Format GEOID to AFFGEOID standard."""
    return f"1500000US{geoid}"

def geometric_mean_logits_to_probs(logits_d, logits_r, logits_o):
    """
    Convert logits for D, R, O to probabilities that sum to 1 using geometric mean normalization.
    
    Uses geometric mean normalization: prob = exp(logit) / (geometric_mean × normalization_factor)
    where geometric_mean = (exp_D × exp_R × exp_O × exp_N)^(1/4) and normalization ensures sum = 1.
    This ensures D + R + O + N = 1 where N is treated as having logit = 0 (exp_N = 1).
    
    Args:
        logits_d: Array of logits for D vote type
        logits_r: Array of logits for R vote type  
        logits_o: Array of logits for O (other votes) vote type
        
    Returns:
        probs_d, probs_r, probs_o, probs_n: Probability arrays
    """
    # Compute exp of logits
    exp_d = np.exp(logits_d)
    exp_r = np.exp(logits_r)
    exp_o = np.exp(logits_o)
    exp_n = np.ones_like(exp_d)  # N is remainder, treat as exp(0) = 1
    
    # Geometric mean of all four probabilities
    geo_mean = (exp_d * exp_r * exp_o * exp_n) ** (1/4)
    
    # Normalize by geometric mean
    raw_probs_d = exp_d / geo_mean
    raw_probs_r = exp_r / geo_mean
    raw_probs_o = exp_o / geo_mean
    raw_probs_n = exp_n / geo_mean
    
    # Normalize so sum = 1
    total = raw_probs_d + raw_probs_r + raw_probs_o + raw_probs_n
    probs_d = raw_probs_d / total
    probs_r = raw_probs_r / total
    probs_o = raw_probs_o / total
    probs_n = raw_probs_n / total
    
    return probs_d, probs_r, probs_o, probs_n

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

# =============================================================================
# COST FUNCTION
# =============================================================================

def compute_cost(
    logit_arrays,
    df,
    adj_matrix,
    demographic_shares,
    cvap_arrays,
    real_prob_arrays,
    spatial_weight
):
    """
    Compute total cost: aggregate error + spatial smoothness.
    Constraint violations are tracked for diagnostics only (not penalized).
    
    Args:
        logit_arrays: Dict with keys like 'D_Wht', 'R_Wht', etc., each (N,) array
        df: DataFrame with block group data
        adj_matrix: Sparse adjacency matrix
        demographic_shares: Dict of demographic share arrays
        cvap_arrays: Dict of CVAP population arrays
        real_prob_arrays: Dict of observed probability arrays
        spatial_weight: Weight for spatial smoothness term
        
    Returns:
        total_cost, cost_breakdown dict (includes constraint violations for diagnostics)
    """
    demographics = ['Wht', 'His', 'Blk', 'Asn', 'Oth']
    vote_types = ['D', 'R', 'O']  # O = other votes, N = non-voter (remainder)
    num_nodes = len(df)
    
    # Convert logits to probabilities ensuring sum constraint
    probs = {}
    for demo in demographics:
        logits_d = logit_arrays[f"D_{demo}"]
        logits_r = logit_arrays[f"R_{demo}"]
        logits_o = logit_arrays[f"O_{demo}"]
        probs_d, probs_r, probs_o, probs_n = geometric_mean_logits_to_probs(logits_d, logits_r, logits_o)
        probs[f"D_{demo}"] = probs_d
        probs[f"R_{demo}"] = probs_r
        probs[f"O_{demo}"] = probs_o
        probs[f"N_{demo}"] = probs_n
    
    # --- 1. Aggregate Cost: predicted vs observed vote shares ---
    predicted_shares = {}
    for vtype in vote_types + ['N']:
        predicted_shares[vtype] = np.zeros(num_nodes)
        for demo in demographics:
            predicted_shares[vtype] += demographic_shares[demo] * probs[f"{vtype}_{demo}"]
    
    aggregate_cost = 0.0
    for vtype in vote_types + ['N']:
        diff = predicted_shares[vtype] - real_prob_arrays[vtype]
        aggregate_cost += np.sum(diff ** 2)
    
    # --- 2. Spatial Cost: smoothness between neighbors ---
    spatial_cost = 0.0
    for vtype in vote_types:
        for demo in demographics:
            key = f"{vtype}_{demo}"
            logit_vals = logit_arrays[key]
            
            # Compute neighbor averages using adjacency matrix
            neighbor_logits_sum = adj_matrix @ logit_vals
            neighbor_counts = adj_matrix @ np.ones(num_nodes)
            neighbor_avg_logits = neighbor_logits_sum / (neighbor_counts + EPSILON)
            
            # Spatial difference
            spatial_diff = logit_vals - neighbor_avg_logits
            spatial_cost += np.sum(spatial_diff ** 2)
    
    # --- 3. Constraint Violations (tracked for diagnostics only, not penalized) ---
    # Basis matrices maintain row/column sums, logit transform maintains 0-1 bounds
    constraint_violations = 0.0
    
    # Sum constraint: probabilities should sum to 1 for each demographic
    for demo in demographics:
        for i in range(num_nodes):
            prob_sum = (probs[f"D_{demo}"][i] + probs[f"R_{demo}"][i] + 
                       probs[f"N_{demo}"][i] + probs[f"O_{demo}"][i])
            constraint_violations += (prob_sum - 1.0) ** 2
    
    # Non-negativity constraint: probabilities >= 0
    for vtype in vote_types + ['N']:
        for demo in demographics:
            key = f"{vtype}_{demo}"
            negative_probs = np.maximum(0, -probs[key])
            constraint_violations += np.sum(negative_probs ** 2)
    
    # Total cost: only aggregate and spatial (constraints handled by basis matrices and logit transform)
    total_cost = aggregate_cost + spatial_weight * spatial_cost
    
    cost_breakdown = {
        'aggregate': aggregate_cost,
        'spatial': spatial_cost,
        'constraint': constraint_violations,  # Tracked for diagnostics only
        'total': total_cost
    }
    
    return total_cost, cost_breakdown

# =============================================================================
# GRADIENT COMPUTATION
# =============================================================================

def compute_gradients(
    logit_arrays,
    df,
    adj_matrix,
    demographic_shares,
    cvap_arrays,
    real_prob_arrays,
    spatial_weight
):
    """
    Compute gradients of cost function w.r.t. logit arrays.
    
    Returns:
        gradients: Dict with same keys as logit_arrays, each (N,) array
    """
    demographics = ['Wht', 'His', 'Blk', 'Asn', 'Oth']
    vote_types = ['D', 'R', 'O']  # O = other votes, N = non-voter (remainder)
    num_nodes = len(df)
    
    # Convert logits to probabilities ensuring sum constraint
    # Use geometric mean normalization
    probs = {}
    for demo in demographics:
        logits_d = logit_arrays[f"D_{demo}"]
        logits_r = logit_arrays[f"R_{demo}"]
        logits_o = logit_arrays[f"O_{demo}"]
        probs_d, probs_r, probs_o, probs_n = geometric_mean_logits_to_probs(logits_d, logits_r, logits_o)
        probs[f"D_{demo}"] = probs_d
        probs[f"R_{demo}"] = probs_r
        probs[f"O_{demo}"] = probs_o
        probs[f"N_{demo}"] = probs_n
    
    # Initialize gradients
    gradients = {}
    for vtype in vote_types:
        for demo in demographics:
            gradients[f"{vtype}_{demo}"] = np.zeros(num_nodes)
    
    # Aggregate term gradients
    predicted_shares = {}
    for vtype in vote_types + ['N']:
        predicted_shares[vtype] = np.zeros(num_nodes)
        for demo in demographics:
            predicted_shares[vtype] += demographic_shares[demo] * probs[f"{vtype}_{demo}"]
    
    for vtype in vote_types:
        for demo in demographics:
            key = f"{vtype}_{demo}"
            # Gradient from aggregate error
            # Need derivatives for geometric mean normalization
            # For geometric mean: geo_mean = (exp_d * exp_r * exp_o)^(1/4)
            # raw_prob_i = exp_i / geo_mean, prob_i = raw_prob_i / sum(raw_probs)
            # This is complex, so we'll use numerical differentiation or simplified approximation
            # For now, use a simplified approach: treat as if probabilities are directly from exp(logit)
            # with normalization, which gives similar derivatives to softmax but scaled
            for target_vtype in vote_types + ['N']:
                diff = predicted_shares[target_vtype] - real_prob_arrays[target_vtype]
                prob_i = probs[key]
                if target_vtype == vtype:
                    # d(prob_i)/d(logit_i) ≈ prob_i * (1 - prob_i) for geometric mean (similar to softmax)
                    prob_derivative = prob_i * (1 - prob_i)
                    gradients[key] += 2 * diff * demographic_shares[demo] * prob_derivative
                elif target_vtype == 'N':
                    # d(prob_N)/d(logit_i) ≈ -prob_N * prob_i
                    prob_n = probs[f"N_{demo}"]
                    prob_derivative = -prob_n * prob_i
                    gradients[key] += 2 * diff * demographic_shares[demo] * prob_derivative
                else:
                    # Cross-term: d(prob_j)/d(logit_i) ≈ -prob_j * prob_i
                    prob_j = probs[f"{target_vtype}_{demo}"]
                    prob_derivative = -prob_j * prob_i
                    gradients[key] += 2 * diff * demographic_shares[demo] * prob_derivative
    
    # Spatial term gradients
    for vtype in vote_types:
        for demo in demographics:
            key = f"{vtype}_{demo}"
            logit_vals = logit_arrays[key]
            
            # Neighbor averages
            neighbor_logits_sum = adj_matrix @ logit_vals
            neighbor_counts = adj_matrix @ np.ones(num_nodes)
            neighbor_avg_logits = neighbor_logits_sum / (neighbor_counts + EPSILON)
            
            # Gradient: 2 * (logit - neighbor_avg)
            # Also account for this node being a neighbor of others
            spatial_grad = 2 * (logit_vals - neighbor_avg_logits)
            # Contribution from neighbors that reference this node
            neighbor_grad = adj_matrix.T @ (2 * (logit_vals - neighbor_avg_logits) / (neighbor_counts + EPSILON))
            gradients[key] += spatial_weight * (spatial_grad + neighbor_grad)
    
    # Constraint penalty gradients removed - constraints maintained by basis matrices and logit transform
    
    return gradients

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
    diagnostic_interval=10
):
    """
    Run gradient descent optimization with basis matrix projection.
    
    Args:
        df: Prepared DataFrame
        adj_matrix: Sparse adjacency matrix
        learning_rate: Step size for gradient descent
        spatial_weight: Weight for spatial smoothness
        iterations: Maximum number of iterations
        convergence_threshold: Stop if cost change < threshold
        output_dir: Directory for output files
        diagnostic_mode: If True, collect detailed diagnostics
        diagnostic_interval: Print diagnostics every N iterations
        
    Returns:
        final_logit_arrays: Dict of final logit values
        history: List of cost values per iteration
        diagnostic_history: List of diagnostic entries (if diagnostic_mode=True)
    """
    demographics = ['Wht', 'His', 'Blk', 'Asn', 'Oth']
    vote_types = ['D', 'R', 'O']  # O = other votes, N = non-voter (remainder)
    num_nodes = len(df)
    
    print("Initializing optimization...")
    
    # Pre-compute arrays
    demographic_shares = {
        demo: (df[f"cvap_{demo}"] / (df["cvap_total"] + EPSILON)).values
        for demo in demographics
    }
    cvap_arrays = {demo: df[f"cvap_{demo}"].values for demo in demographics}
    real_prob_arrays = {
        vtype: df[f"real_{vtype}_prob"].values if f"real_{vtype}_prob" in df.columns
        else un_logit(df[f"real_{vtype}_logit"].values)
        for vtype in vote_types + ['N']
    }
    
    # Initialize logits using (row sum * column sum) / total sum method
    # If pre-initialized values exist, use them; otherwise compute from aggregate data
    logit_arrays = {}
    for vtype in vote_types:
        for demo in demographics:
            key = f"{vtype}_{demo}"
            if f"{key}_logit" in df.columns:
                logit_arrays[key] = df[f"{key}_logit"].values.copy()
            elif f"{key}_prob" in df.columns:
                logit_arrays[key] = logit(df[f"{key}_prob"].values)
            else:
                # Initialize using (votes_vote_type * cvap_total) / (cvap_total^2)
                # Formula: prob = (votes_vtype * cvap_total) / (cvap_total^2) = votes_vtype / cvap_total
                # This initializes all demographics with the same probabilities (block group's overall vote shares)
                real_share_key = f"real_{vtype}_share"
                if real_share_key in df.columns:
                    # prob = votes_vtype / cvap_total (which is real_vtype_share)
                    initial_probs = df[real_share_key].values
                elif f"real_{vtype}_prob" in df.columns:
                    initial_probs = df[f"real_{vtype}_prob"].values
                elif f"real_{vtype}_logit" in df.columns:
                    initial_probs = un_logit(df[f"real_{vtype}_logit"].values)
                else:
                    # Fallback: compute from votes and cvap_total
                    # prob = (votes_vtype * cvap_total) / (cvap_total^2) = votes_vtype / cvap_total
                    votes_key = f"votes_{vtype}" if f"votes_{vtype}" in df.columns else None
                    if votes_key and "cvap_total" in df.columns:
                        initial_probs = df[votes_key].values / (df["cvap_total"].values + EPSILON)
                    else:
                        # Final fallback: use uniform distribution
                        initial_probs = np.ones(num_nodes) / len(vote_types + ['N'])
                
                logit_arrays[key] = logit(initial_probs)
    
    # Create basis matrices (fundamental circuits)
    basis_matrices = create_basis_matrices(n=4, m=5)  # 4 vote types, 5 demographics
    print(f"Created {len(basis_matrices)} fundamental circuit basis matrices (k = (4-1)(5-1) = 12)")
    
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
            logit_arrays, df, adj_matrix, demographic_shares,
            cvap_arrays, real_prob_arrays, spatial_weight
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
        gradients = compute_gradients(
            logit_arrays, df, adj_matrix, demographic_shares,
            cvap_arrays, real_prob_arrays, spatial_weight
        )
        
        # Collect diagnostic information if enabled
        if diagnostic_mode:
            # Compute gradient norms
            all_gradients = np.concatenate([g.flatten() for g in gradients.values()])
            gradient_norm_max = np.max(np.abs(all_gradients))
            gradient_norm_mean = np.mean(np.abs(all_gradients))
            gradient_norm_std = np.std(np.abs(all_gradients))
            
            # Compute step sizes
            step_sizes = []
            for key in gradients:
                step_sizes.extend(np.abs(learning_rate * gradients[key]))
            step_size_mean = np.mean(step_sizes) if step_sizes else 0.0
            
            # Aggregate error by vote type
            predicted_shares = {}
            for vtype in vote_types + ['N']:
                predicted_shares[vtype] = np.zeros(num_nodes)
                for demo in demographics:
                    logits_d = logit_arrays[f"D_{demo}"]
                    logits_r = logit_arrays[f"R_{demo}"]
                    logits_o = logit_arrays[f"O_{demo}"]
                    probs_d, probs_r, probs_o, probs_n = geometric_mean_logits_to_probs(logits_d, logits_r, logits_o)
                    probs = {f"D_{demo}": probs_d, f"R_{demo}": probs_r, f"O_{demo}": probs_o, f"N_{demo}": probs_n}
                    predicted_shares[vtype] += demographic_shares[demo] * probs[f"{vtype}_{demo}"]
            
            aggregate_error_by_vtype = {}
            for vtype in vote_types + ['N']:
                diff = predicted_shares[vtype] - real_prob_arrays[vtype]
                aggregate_error_by_vtype[vtype] = np.mean(np.abs(diff))
            
            # Spatial smoothness by demographic
            spatial_smoothness_by_demo = {}
            for demo in demographics:
                demo_spatial_diffs = []
                for vtype in vote_types:
                    key = f"{vtype}_{demo}"
                    logit_vals = logit_arrays[key]
                    neighbor_logits_sum = adj_matrix @ logit_vals
                    neighbor_counts = adj_matrix @ np.ones(num_nodes)
                    neighbor_avg_logits = neighbor_logits_sum / (neighbor_counts + EPSILON)
                    spatial_diff = np.abs(logit_vals - neighbor_avg_logits)
                    demo_spatial_diffs.extend(spatial_diff)
                spatial_smoothness_by_demo[demo] = np.mean(demo_spatial_diffs) if demo_spatial_diffs else 0.0
            
            # Largest gradients (top 10)
            gradient_items = [(key, idx, val) for key, arr in gradients.items() for idx, val in enumerate(arr)]
            gradient_items.sort(key=lambda x: abs(x[2]), reverse=True)
            largest_gradients = [
                {'key': key, 'index': idx, 'value': float(val)}
                for key, idx, val in gradient_items[:10]
            ]
            
            # Constraint violations (for diagnostics)
            constraint_violations_list = []
            for demo in demographics:
                logits_d = logit_arrays[f"D_{demo}"]
                logits_r = logit_arrays[f"R_{demo}"]
                logits_o = logit_arrays[f"O_{demo}"]
                probs_d, probs_r, probs_o, probs_n = geometric_mean_logits_to_probs(logits_d, logits_r, logits_o)
                prob_sum = probs_d + probs_r + probs_o + probs_n
                constraint_violations_list.extend(np.abs(prob_sum - 1.0))
            
            constraint_violations_max = np.max(constraint_violations_list) if constraint_violations_list else 0.0
            constraint_violations_mean = np.mean(constraint_violations_list) if constraint_violations_list else 0.0
            
            diagnostic_entry = {
                'iteration': i + 1,
                'cost_total': cost,
                'cost_aggregate': cost_breakdown['aggregate'],
                'cost_spatial': cost_breakdown['spatial'],
                'cost_change': cost_change if cost_change is not None else 0.0,
                'cost_change_rate': cost_change_rate if cost_change_rate is not None else 0.0,
                'gradient_norm_max': gradient_norm_max,
                'gradient_norm_mean': gradient_norm_mean,
                'gradient_norm_std': gradient_norm_std,
                'step_size_mean': step_size_mean,
                'constraint_violations_max': constraint_violations_max,
                'constraint_violations_mean': constraint_violations_mean,
                'aggregate_error_D': aggregate_error_by_vtype.get('D', 0.0),
                'aggregate_error_R': aggregate_error_by_vtype.get('R', 0.0),
                'aggregate_error_O': aggregate_error_by_vtype.get('O', 0.0),
                'aggregate_error_N': aggregate_error_by_vtype.get('N', 0.0),
                'spatial_smoothness_Wht': spatial_smoothness_by_demo.get('Wht', 0.0),
                'spatial_smoothness_His': spatial_smoothness_by_demo.get('His', 0.0),
                'spatial_smoothness_Blk': spatial_smoothness_by_demo.get('Blk', 0.0),
                'spatial_smoothness_Asn': spatial_smoothness_by_demo.get('Asn', 0.0),
                'spatial_smoothness_Oth': spatial_smoothness_by_demo.get('Oth', 0.0),
            }
            diagnostic_history.append(diagnostic_entry)
            
            # Print diagnostics at specified interval
            if (i + 1) % diagnostic_interval == 0 or i == 0:
                print(f"\n--- Iteration {i+1} Diagnostics ---")
                print(f"Cost: total={cost:.6f}, agg={cost_breakdown['aggregate']:.6f}, spatial={cost_breakdown['spatial']:.6f}")
                print(f"Cost change: {cost_change:.6e}" if cost_change is not None else "Cost change: N/A")
                print(f"Gradient norms: max={gradient_norm_max:.6f}, mean={gradient_norm_mean:.6f}, std={gradient_norm_std:.6f}")
                print(f"Step size mean: {step_size_mean:.6e}")
                print(f"Constraint violations: max={constraint_violations_max:.6e}, mean={constraint_violations_mean:.6e}")
                print(f"Aggregate errors: D={aggregate_error_by_vtype.get('D', 0):.6f}, R={aggregate_error_by_vtype.get('R', 0):.6f}, "
                      f"O={aggregate_error_by_vtype.get('O', 0):.6f}, N={aggregate_error_by_vtype.get('N', 0):.6f}")
        
        # Project gradients onto basis and update using fundamental circuits
        # Work with full gradient matrix (4 vote types × 5 demographics) for each block group
        vote_types_full = ['D', 'R', 'O', 'N']  # O = other votes, N = non-voter
        
        for node_idx in range(num_nodes):
            # Build full gradient matrix (4 vote types × 5 demographics)
            gradient_matrix = np.zeros((4, 5))
            
            # Fill in gradients for D, R, O (we have these directly)
            for vtype_idx, vtype in enumerate(vote_types):  # D, R, O (indices 0, 1, 2)
                for demo_idx, demo in enumerate(demographics):
                    key = f"{vtype}_{demo}"
                    gradient_matrix[vtype_idx, demo_idx] = gradients[key][node_idx]
            
            # N gradients (row 3) are determined by constraint to maintain column sums
            # Since N = 1 - D - R - O, the gradient for N maintains: dN = -(dD + dR + dO)
            for demo_idx in range(5):
                gradient_matrix[3, demo_idx] = -(gradient_matrix[0, demo_idx] + 
                                                  gradient_matrix[1, demo_idx] + 
                                                  gradient_matrix[2, demo_idx])
            
            # Project onto basis space spanned by fundamental circuits
            projected = project_gradient_matrix_onto_basis(gradient_matrix, basis_matrices)
            
            # Update logits (only D, R, O; N is derived from constraint)
            for vtype_idx, vtype in enumerate(vote_types):
                for demo_idx, demo in enumerate(demographics):
                    key = f"{vtype}_{demo}"
                    logit_arrays[key][node_idx] -= learning_rate * projected[vtype_idx, demo_idx]
        
        # Clip logits to prevent overflow
        for key in logit_arrays:
            logit_arrays[key] = np.clip(logit_arrays[key], -10, 10)
        
        # Print progress every 100 iterations (if not in diagnostic mode or not at diagnostic interval)
        if not diagnostic_mode and (i + 1) % 100 == 0:
            print(f"\nIteration {i+1}: cost = {cost:.6f} (agg={cost_breakdown['aggregate']:.6f}, "
                  f"spatial={cost_breakdown['spatial']:.6f}, constraint={cost_breakdown['constraint']:.6f})")
    
    print("\nOptimization complete.")
    return logit_arrays, history, diagnostic_history

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gradient descent optimization for ecological inference."
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
        "--spatial-weight", type=float, default=0.5, help="Weight for spatial smoothness term."
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
    logit_arrays, history, diagnostic_history = run_optimization(
        df=df,
        adj_matrix=adj_matrix,
        learning_rate=args.learning_rate,
        spatial_weight=args.spatial_weight,
        iterations=args.iterations,
        convergence_threshold=args.convergence_threshold,
        output_dir=".",
        diagnostic_mode=args.diagnostic_mode,
        diagnostic_interval=args.diagnostic_interval
    )
    
    # Save diagnostics if enabled
    if args.diagnostic_mode and diagnostic_history:
        diagnostic_df = pd.DataFrame(diagnostic_history)
        diagnostic_df.to_csv(args.diagnostic_output, index=False)
        print(f"\nDiagnostics saved to {args.diagnostic_output}")
    
    # Convert logits to probabilities
    print("Converting logits to probabilities...")
    demographics = ['Wht', 'His', 'Blk', 'Asn', 'Oth']
    vote_types = ['D', 'R', 'O']  # O = other votes, N = non-voter (remainder)
    
    output_data = {'AFFGEOID': df.index.values}
    
    for demo in demographics:
        logits_d = logit_arrays[f"D_{demo}"]
        logits_r = logit_arrays[f"R_{demo}"]
        logits_o = logit_arrays[f"O_{demo}"]
        probs_d, probs_r, probs_o, probs_n = geometric_mean_logits_to_probs(logits_d, logits_r, logits_o)
        output_data[f"D_{demo}_prob"] = probs_d
        output_data[f"R_{demo}_prob"] = probs_r
        output_data[f"O_{demo}_prob"] = probs_o
        output_data[f"N_{demo}_prob"] = probs_n
    
    # Save to CSV
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(args.output, index=False)
    print(f"\nOutput saved to {args.output}")
    print(f"Final DataFrame shape: {output_df.shape}")
    
    # Print final cost
    if history:
        final_cost = history[-1]['total']
        print(f"Final cost: {final_cost:.6f}")
        print(f"  - Aggregate: {history[-1]['aggregate']:.6f}")
        print(f"  - Spatial: {history[-1]['spatial']:.6f}")
        print(f"  - Constraint: {history[-1]['constraint']:.6f}")

if __name__ == "__main__":
    main()

