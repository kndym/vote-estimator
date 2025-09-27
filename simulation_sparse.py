"""
Run the spatial ecological inference simulation using a high-performance
sparse matrix approach.

This script iteratively adjusts demographic voting estimates for block groups
based on internal consistency (matching real election results) and spatial
cohesion (behaving like neighbors). It is optimized for speed for representing
the neighborhood graph as a SciPy sparse matrix, replacing slow row-by-row
operations with vectorized matrix algebra.

- Input:
    - The prepared data file (e.g., data.feather) from prepare_data.py.
    - The graph file (e.g., blockgroups_graph.gpickle) from graph.py.
- Output:
    - final_estimates.feather: A DataFrame with the final logit estimates.
    - convergence.png: A plot showing the model's error over iterations.
"""
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import pickle
import networkx as nx
from scipy.sparse import csr_matrix

# =============================================================================
# HELPER FUNCTIONS (Identical to previous script)
# =============================================================================

EPSILON = 1e-12

def logit(p):
    p = np.clip(p, EPSILON, 1 - EPSILON)
    return np.log(p / (1 - p))

def un_logit(x):
    return 1 / (1 + np.exp(-x))

def plot_convergence(history, output_path, final=False):
    """Plots the overall internal consistency error (real vs. predicted)."""
    if not history:
        return
    df_history = pd.DataFrame(history)

    if "mae" not in df_history.columns or df_history["mae"].isnull().all():
        print("Warning: MAE (Logit) data is missing or all NaN.", file=sys.stderr)
        return

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except:
        plt.style.use("ggplot")

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot Logit MAE
    color1 = "dodgerblue"
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Mean Absolute Error (Logit Space)", fontsize=12, color=color1)
    ax1.plot(df_history["iteration"], df_history["mae"], label="Overall MAE (Logit)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_yscale("log")
    lines, labels = ax1.get_legend_handles_labels()

    # Plot Probability MAE on a shared X-axis
    if "mae_prob" in df_history.columns and not df_history["mae_prob"].isnull().all():
        ax2 = ax1.twinx()
        color2 = "crimson"
        ax2.set_ylabel("Mean Absolute Error (Probability Space)", fontsize=12, color=color2)
        ax2.plot(df_history["iteration"], df_history["mae_prob"], label="Overall MAE (Prob)", color=color2, linestyle="--")
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_yscale("log")
        more_lines, more_labels = ax2.get_legend_handles_labels()
        lines.extend(more_lines)
        labels.extend(more_labels)

    ax1.set_title("Model Convergence: Internal Consistency Error", fontsize=16, fontweight="bold")
    ax1.grid(True, which="both", ls="-", alpha=0.7)
    ax1.legend(lines, labels, loc="upper right")
    plt.tight_layout()
    
    filename = "convergence_final.png" if final else "convergence.png"
    # Correctly join path and filename
    output_path = os.path.join(os.path.dirname(output_path), filename)


    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_spatial_convergence(history, output_dir, final=False):
    """Plots the spatial cohesion error, broken down by race and overall."""
    if not history:
        return
    df = pd.DataFrame(history).set_index("iteration")
    demographics = ["Wht", "His", "Blk", "Asn", "Oth"]
    
    # Define colors for demographics
    colors = plt.cm.viridis(np.linspace(0, 1, len(demographics)))
    demo_colors = {demo: colors[i] for i, demo in enumerate(demographics)}

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except:
        plt.style.use("ggplot")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 18), sharex=True)
    fig.suptitle("Model Convergence: Spatial Cohesion Error", fontsize=18, fontweight="bold")

    # --- Logit Plot ---
    ax1.set_title("Logit Space", fontsize=14)
    for demo in demographics:
        col_name = f"mae_logit_{demo}"
        if col_name in df.columns:
            ax1.plot(df.index, df[col_name], label=f"{demo}", color=demo_colors[demo], alpha=0.8)
    
    if "mae_logit_overall" in df.columns:
        ax1.plot(df.index, df["mae_logit_overall"], label="Overall", color='black', linewidth=2.5, linestyle=':')
    
    ax1.set_ylabel("Mean Absolute Difference from Neighbors")
    ax1.set_yscale("log")
    ax1.legend(title="Demographic")
    ax1.grid(True, which="both", ls="-", alpha=0.6)

    # --- Probability Plot ---
    ax2.set_title("Probability Space", fontsize=14)
    for demo in demographics:
        col_name = f"mae_prob_{demo}"
        if col_name in df.columns:
            ax2.plot(df.index, df[col_name], label=f"{demo}", color=demo_colors[demo], alpha=0.8)

    if "mae_prob_overall" in df.columns:
        ax2.plot(df.index, df["mae_prob_overall"], label="Overall", color='black', linewidth=2.5, linestyle=':')

    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Mean Absolute Difference from Neighbors")
    ax2.set_yscale("log")
    ax2.legend(title="Demographic")
    ax2.grid(True, which="both", ls="-", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    filename = "spatial_convergence_final.png" if final else "spatial_convergence.png"
    output_path = os.path.join(output_dir, filename)
    
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_neighbor_differences(df, adj_matrix, output_dir):
    """
    Calculates and plots the average absolute difference in logit space
    between each block group and its neighbors, broken down by race.

    Args:
        df (pd.DataFrame): The final DataFrame with logit estimates.
        adj_matrix (scipy.sparse.csr_matrix): The adjacency matrix.
        output_dir (str): Directory to save the plot.
    """
    demographics = ["Wht", "His", "Blk", "Asn", "Oth"]
    vote_types = ["D", "R", "N"]
    EPSILON = 1e-12

    cvap_arrays = {demo: df[f"cvap_{demo}"].values for demo in demographics}
    current_logit_arrays = {
        f"{vtype}_{demo}": df[f"{vtype}_{demo}_logit"].values
        for vtype in vote_types
        for demo in demographics
    }

    avg_diffs = {}

    print("Calculating neighbor differences by race...")
    for demo in tqdm(demographics, desc="Processing demographics"):
        diffs = []
        for vtype in vote_types:
            pred_voters = cvap_arrays[demo] * un_logit(
                current_logit_arrays[f"{vtype}_{demo}"]
            )
            total_pop = cvap_arrays[demo]

            neighbor_voters_sum = adj_matrix @ pred_voters
            neighbor_pop_sum = adj_matrix @ total_pop
            neighbor_avg_share = neighbor_voters_sum / (neighbor_pop_sum + EPSILON)
            neighbor_logit = logit(neighbor_avg_share)

            neighbor_diff = current_logit_arrays[f"{vtype}_{demo}"] - neighbor_logit
            diffs.append(np.abs(neighbor_diff))

        if diffs:
            # Concatenate all differences for the current demographic
            all_diffs = np.concatenate(diffs)

            # Create a corresponding weights array (CVAP population), tiled for each vote type
            weights = np.tile(cvap_arrays[demo], len(vote_types))

            # Filter out NaN values from both diffs and weights to ensure they align
            valid_indices = ~np.isnan(all_diffs)
            valid_diffs = all_diffs[valid_indices]
            valid_weights = weights[valid_indices]

            # Calculate the weighted average if possible, otherwise fall back
            if np.sum(valid_weights) > 0:
                avg_diffs[demo] = np.average(valid_diffs, weights=valid_weights)
            else:
                # Fallback to unweighted mean if total weight is zero
                avg_diffs[demo] = np.nanmean(all_diffs)
        else:
            avg_diffs[demo] = 0

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except:
        plt.style.use("ggplot")

    fig, ax = plt.subplots(figsize=(10, 6))
    races = list(avg_diffs.keys())
    values = list(avg_diffs.values())
    
    if not any(values):
        print("Warning: All neighbor difference values are zero. Skipping plot.", file=sys.stderr)
        plt.close(fig)
        return

    ax.bar(races, values, color="skyblue", edgecolor='black')

    ax.set_title("Average Logit Difference from Neighbors by Race", fontsize=15, fontweight="bold")
    ax.set_xlabel("Race/Ethnicity", fontsize=12)
    ax.set_ylabel("Mean Absolute Difference (Logit Space)", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(values):
        ax.text(i, v + 0.01 * max(values), f"{v:.3f}", ha='center', va='bottom')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "neighbor_differences_by_race.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved neighbor differences plot to {output_path}")


# =============================================================================
# HIGH-PERFORMANCE SIMULATION CORE
# =============================================================================

def run_simulation(
    df,
    adj_matrix,
    alpha,
    beta,
    iterations,
    plot_interval,
    output_dir,
    initial_dist,
    initial_std_dev,
):
    """
    Executes the main iterative learning loop using sparse matrix operations.

    Args:
        df (pd.DataFrame): The prepared, sorted input DataFrame.
        adj_matrix (scipy.sparse.csr_matrix): The adjacency matrix of the graph.
        alpha (float): The main learning rate.
        beta (float): The spatial weight parameter (0 to 1).
        iterations (int): Total number of iterations to run.
        plot_interval (int): How often to save the convergence plot.
        output_dir (str): Directory to save outputs.
        initial_dist (str): Method for initializing logits ('exit_poll' or 'normal').
        initial_std_dev (float): Std dev for normal distribution.
    """

    print(f"Initializing simulation with sparse matrix backend (dist: {initial_dist})...")
    demographics = ["Wht", "His", "Blk", "Asn", "Oth"]
    vote_types = ["D", "R", "N"]

    # Pre-calculate demographic shares and convert key columns to numpy arrays for speed.
    demographic_shares = {
        demo: (df[f"cvap_{demo}"] / (df["cvap_total"] + EPSILON)).values
        for demo in demographics
    }
    cvap_arrays = {demo: df[f"cvap_{demo}"].values for demo in demographics}
    real_logit_arrays = {
        vtype: df[f"real_{vtype}_logit"].values for vtype in vote_types + ["O"]
    }

    # Create numpy arrays for the logit values we will be updating.
    current_logit_arrays = {}
    num_nodes = len(df)
    for vtype in vote_types:
        for demo in demographics:
            key = f"{vtype}_{demo}"
            if initial_dist == "normal":
                mean_logit = 0
                current_logit_arrays[key] = np.random.normal(
                    loc=mean_logit, scale=initial_std_dev, size=num_nodes
                )
            else:  # Default 'exit_poll' behavior
                current_logit_arrays[key] = df[key].values.copy()

    history = []
    spatial_history = []  # New history for spatial metrics
    print(f"Starting simulation for {iterations} iterations...")
    print(f"Parameters: alpha={alpha}, beta={beta}")

    # --- Main Iteration Loop ---
    for i in tqdm(
        range(iterations), desc="Simulating", unit="iteration", dynamic_ncols=True
    ):

        # --- A. Calculate Predicted Shares and Internal Error ---
        predicted_shares = {}
        for vtype in vote_types:
            predicted_shares[vtype] = sum(
                demographic_shares[demo]
                * un_logit(current_logit_arrays[f"{vtype}_{demo}"])
                for demo in demographics
            )
        predicted_shares["O"] = 1 - sum(predicted_shares.values())

        logit_diffs = {
            vtype: real_logit_arrays[vtype] - logit(predicted_shares[vtype])
            for vtype in vote_types + ["O"]
        }

        # --- B. Calculate Spatial Error and Update Step ---
        spatial_diffs_logit = {demo: [] for demo in demographics}
        spatial_diffs_prob = {demo: [] for demo in demographics}

        for vtype in vote_types:
            for demo in demographics:
                # Calculate predicted number of voters for this group
                pred_voters = cvap_arrays[demo] * un_logit(
                    current_logit_arrays[f"{vtype}_{demo}"]
                )
                total_pop = cvap_arrays[demo]

                # Use matrix multiplication to get neighbor averages
                neighbor_voters_sum = adj_matrix @ pred_voters
                neighbor_pop_sum = adj_matrix @ total_pop
                neighbor_avg_share = neighbor_voters_sum / (neighbor_pop_sum + EPSILON)
                neighbor_logit = logit(neighbor_avg_share)

                # --- Calculate Spatial Difference (for metrics) ---
                neighbor_diff = current_logit_arrays[f"{vtype}_{demo}"] - neighbor_logit
                spatial_diffs_logit[demo].append(np.abs(neighbor_diff))
                
                prob_current = un_logit(current_logit_arrays[f"{vtype}_{demo}"])
                prob_neighbor = un_logit(neighbor_logit)
                spatial_diffs_prob[demo].append(np.abs(prob_current - prob_neighbor))

                # --- C. Calculate and Apply Update Step ---
                logit_diff = logit_diffs[vtype]
                o_logit_diff = logit_diffs["O"]
                
                internal_term = beta * logit_diff
                spatial_term = (1 - beta) * neighbor_diff
                o_correction_term = (alpha * beta / 3) * o_logit_diff
                update_step = alpha * (internal_term + spatial_term) - o_correction_term
                current_logit_arrays[f"{vtype}_{demo}"] += update_step

        # --- D. Record Metrics and Periodically Plot ---
        
        # 1. Internal Consistency Metrics (Overall)
        mae_logit = np.mean(
            np.abs(logit_diffs["D"])
            + np.abs(logit_diffs["R"])
            + np.abs(logit_diffs["N"])
            + np.abs(logit_diffs["O"])
        )
        
        prob_diffs = {
            vtype: un_logit(real_logit_arrays[vtype]) - predicted_shares[vtype]
            for vtype in vote_types + ["O"]
        }
        mae_prob = np.mean(
            np.abs(prob_diffs["D"])
            + np.abs(prob_diffs["R"])
            + np.abs(prob_diffs["N"])
            + np.abs(prob_diffs["O"])
        )
        
        history.append({"iteration": i + 1, "mae": mae_logit, "mae_prob": mae_prob})

        # 2. Spatial Cohesion Metrics (By Race and Overall)
        spatial_metrics = {"iteration": i + 1}
        all_races_logit_diffs = []
        all_races_prob_diffs = []

        for demo in demographics:
            demo_all_logit = np.concatenate(spatial_diffs_logit[demo])
            demo_all_prob = np.concatenate(spatial_diffs_prob[demo])
            
            spatial_metrics[f"mae_logit_{demo}"] = np.nanmean(demo_all_logit)
            spatial_metrics[f"mae_prob_{demo}"] = np.nanmean(demo_all_prob)
            
            all_races_logit_diffs.append(demo_all_logit)
            all_races_prob_diffs.append(demo_all_prob)

        spatial_metrics["mae_logit_overall"] = np.nanmean(np.concatenate(all_races_logit_diffs))
        spatial_metrics["mae_prob_overall"] = np.nanmean(np.concatenate(all_races_prob_diffs))
        spatial_history.append(spatial_metrics)

        if i == 0:
            print(f"\n--- DEBUG: First MAE (Internal): Logit={mae_logit:.6f}, Prob={mae_prob:.6f} ---\n")

        if (i + 1) % plot_interval == 0 or (i + 1) == iterations:
            plot_convergence(history, os.path.join(output_dir, "convergence.png"))
            plot_spatial_convergence(spatial_history, output_dir)

    print("\n--- Simulation Complete ---")

    # --- Write final results back to the DataFrame ---
    print("Writing final estimates back to DataFrame...")
    for key, value_array in current_logit_arrays.items():
        df[key] = value_array

    return df, history, spatial_history

def main():
    parser = argparse.ArgumentParser(
        description="Run the high-performance spatial voting simulation."
    )
    parser.add_argument(
        "prepared_data_path", help="Path to the prepared input .feather file."
    )
    parser.add_argument(
        "graph_file", help="Path to the blockgroups_graph.gpickle file."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01, help="Learning rate (alpha)."
    )
    parser.add_argument(
        "--beta", type=float, default=0.5, help="Spatial weight (beta)."
    )
    parser.add_argument(
        "--iterations", type=int, default=200, help="Number of simulation iterations."
    )
    parser.add_argument(
        "--plot_interval",
        type=int,
        default=10,
        help="How often to update the convergence plot.",
    )
    parser.add_argument(
        "--output_dir", default="results_sparse", help="Directory to save output files."
    )
    parser.add_argument(
        "--initial-dist",
        choices=["exit_poll", "normal"],
        default="exit_poll",
        help="Method for initializing vote logits ('exit_poll' or 'normal').",
    )
    parser.add_argument(
        "--initial-std-dev",
        type=float,
        default=1,
        help="Standard deviation for 'normal' initial distribution.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Load Data ---
    print(f"Loading prepared data from {args.prepared_data_path}...")
    try:
        df = pd.read_feather(args.prepared_data_path).set_index("AFFGEOID")
    except Exception as e:
        print(f"Error loading data file: {e}. Exiting.", file=sys.stderr)
        sys.exit(1)

    # --- 2. Load Graph and Build Sparse Adjacency Matrix ---
    print(f"Loading graph from {args.graph_file}...")
    try:
        with open(args.graph_file, "rb") as f:
            G = pickle.load(f)
    except Exception as e:
        print(f"Error loading graph file: {e}. Exiting.", file=sys.stderr)
        sys.exit(1)

    # CRITICAL STEP: Ensure the DataFrame and the graph's node list are in the
    # same order before creating the matrix. Sorting the DataFrame index is the
    # most reliable way to enforce a canonical order.
    df.sort_index(inplace=True)

    # Create a list of nodes that are present in BOTH the graph and the DataFrame
    graph_nodes = {f"1500000US{node}" for node in G.nodes()}
    df_nodes = set(df.index)

    # The final node list for the matrix is the intersection, sorted.
    node_list = sorted(list(graph_nodes.intersection(df_nodes)))

    # If some nodes in the DataFrame are not in the graph (or vice-versa),
    # subset the DataFrame to only include the common nodes.
    if len(node_list) != len(df):
        print(
            f"Warning: Mismatch between graph nodes and data nodes. Using {len(node_list)} common nodes."
        )
        df = df.loc[node_list]

    print(f"Creating sparse adjacency matrix for {len(node_list)} nodes...")
    adj_matrix = nx.to_scipy_sparse_array(
        G, nodelist=[geoid.replace("1500000US", "") for geoid in node_list], format="csr"
    )

    # --- 3. Run Simulation ---
    final_df, history, spatial_history = run_simulation(
        df=df,
        adj_matrix=adj_matrix,
        alpha=args.alpha,
        beta=args.beta,
        iterations=args.iterations,
        plot_interval=args.plot_interval,
        output_dir=args.output_dir,
        initial_dist=args.initial_dist,
        initial_std_dev=args.initial_std_dev,
    )

    # --- 4. Save Final Results ---
    final_output_path = os.path.join(args.output_dir, "final_estimates.feather")
    print(f"Saving final estimates to {final_output_path}...")
    final_df.reset_index().to_feather(final_output_path)

    # --- 5. Generate Final Plots ---
    print("\n--- Generating Final Plots ---")
    plot_convergence(history, os.path.join(args.output_dir, "convergence.png"), final=True)
    plot_spatial_convergence(spatial_history, args.output_dir, final=True)
    plot_neighbor_differences(final_df, adj_matrix, args.output_dir)

    print("\n--- Success! ---")
    print(f"Outputs saved in '{args.output_dir}' directory.")


if __name__ == "__main__":
    main()
