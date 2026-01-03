# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import networkx as nx
import argparse
import pickle

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def logit(p):
    """Converts a probability (p) into log-odds."""
    # Epsilon is a small value to prevent log(0) or division by zero errors.
    epsilon = 1e-9
    p = np.clip(p, epsilon, 1 - epsilon)
    return np.log(p / (1 - p))

def un_logit(x):
    """Converts log-odds (x) back into a probability."""
    return 1 / (1 + np.exp(-x))

def format_affgeoid(geoid):
    """Formats a GEOID from the graph file to match the AFFGEOID standard."""
    return f"1500000US{geoid}"

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Prepare data for spatial voting simulation.")
    parser.add_argument("main_data_csv", help="Path to the main data CSV with vote and demographic info.")
    parser.add_argument("graph_file", help="Path to the blockgroups_graph.gpickle file.")
    parser.add_argument("output_path", help="Path to save the prepared output file (e.g., data.feather).")
    parser.add_argument("--output-mode", choices=['logit', 'prob'], default='logit', help="Output mode: 'logit' or 'prob'.")
    args = parser.parse_args()

    print("--- Starting Data Preparation ---")

    # --- 2. Load All Input Data ---
    print(f"Loading main data from {args.main_data_csv}...")
    df = pd.read_csv(args.main_data_csv, dtype={'AFFGEOID': str})
    
    # Filter to NY state only (new_state == 36)
    if 'new_state' in df.columns:
        initial_count = len(df)
        df = df[df['new_state'] == 36]
        print(f"Filtered to NY state: {len(df)} rows (from {initial_count}).")
    else:
        print("Warning: 'new_state' column not found. Proceeding without state filtering.")
    
    df.set_index('AFFGEOID', inplace=True)

    print(f"Loading graph from {args.graph_file}...")
    with open(args.graph_file, 'rb') as f:
        G = pickle.load(f)

    # --- 3. Build Adjacency Dictionary ---
    print("Building adjacency dictionary from graph...")
    adjacency_dict = {}
    for u_geoid, v_geoid in G.edges():
        u_affgeoid = format_affgeoid(u_geoid)
        v_affgeoid = format_affgeoid(v_geoid)
        adjacency_dict.setdefault(u_affgeoid, []).append(v_affgeoid)
        adjacency_dict.setdefault(v_affgeoid, []).append(u_affgeoid)
    df['neighbors'] = df.index.to_series().map(adjacency_dict).fillna('').apply(list)
    print(f"Adjacency dictionary built for {len(adjacency_dict)} block groups.")

    # --- 4. Calculate "Real" Vote Shares and Logits/Probs for Each Block Group ---
    print(f"Calculating real vote shares and initializing for {args.output_mode} mode...")
    
    demographics = ['Wht', 'His', 'Blk', 'Asn', 'Oth']
    vote_types = ['D', 'R', 'O', 'N']

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

    if args.output_mode == 'logit':
        df['real_D_logit'] = logit(df['real_D_share'])
        df['real_R_logit'] = logit(df['real_R_share'])
        df['real_N_logit'] = logit(df['real_N_share'])
        df['real_O_logit'] = logit(df['real_O_share'])
    else: # prob mode
        df['real_D_prob'] = df['real_D_share']
        df['real_R_prob'] = df['real_R_share']
        df['real_N_prob'] = df['real_N_share']
        df['real_O_prob'] = df['real_O_share']

    # --- 5. Initialize Demographic-Specific Values using (row sum * column sum) / total^2 ---
    # For each block group: initialize each demographic's vote probabilities
    # Formula: prob = (votes_vote_type * cvap_total) / (cvap_total^2) = votes_vote_type / cvap_total
    # This initializes all demographics with the same probabilities (block group's overall vote shares)
    print("Initializing demographic-specific values using (votes_vote_type * cvap_total) / (cvap_total^2) method...")
    
    for demo in demographics:
        # For each vote type, initialize probability as: (votes_vtype * cvap_total) / (cvap_total^2)
        # This simplifies to: prob = votes_vtype / cvap_total (same for all demographics initially)
        
        if args.output_mode == 'logit':
            df[f'D_{demo}_logit'] = logit(df['real_D_share'])
            df[f'R_{demo}_logit'] = logit(df['real_R_share'])
            df[f'N_{demo}_logit'] = logit(df['real_N_share'])
        else: # prob mode
            df[f'D_{demo}_prob'] = df['real_D_share']
            df[f'R_{demo}_prob'] = df['real_R_share']
            df[f'N_{demo}_prob'] = df['real_N_share']

    # --- 6. Save the Prepared Data ---
    print(f"Preparation complete. Saving output to {args.output_path}...")
    
    if args.output_mode == 'logit':
        final_cols_mask = ['cvap' in col or 'real_' in col or '_logit' in col for col in df.columns]
    else: # prob mode
        final_cols_mask = ['cvap' in col or 'real_' in col or '_prob' in col for col in df.columns]

    final_columns = df.columns[final_cols_mask].tolist() + ['neighbors']
    df[final_columns].reset_index().to_feather(args.output_path)
    
    print("\n--- Success! ---")
    print(f"DataFrame with shape {df[final_columns].shape} saved.")
    print("The DataFrame is now ready for the simulation script.")


if __name__ == "__main__":
    main()
