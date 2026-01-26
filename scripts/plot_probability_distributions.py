"""
Plot distribution of probabilities by race (demographic) and vote type.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Constants
DEMOGRAPHICS = ['Wht', 'His', 'Blk', 'Asn', 'Oth']
VOTE_TYPES = ['D', 'R', 'O', 'N']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

def plot_distributions(csv_file, output_dir='output/plots'):
    """Plot probability distributions by demographic and vote type, weighted by demographic population."""
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file, index_col=0)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Create subplots: one row per demographic, one column per vote type
    fig, axes = plt.subplots(len(DEMOGRAPHICS), len(VOTE_TYPES), figsize=(16, 20))
    fig.suptitle('Probability Distributions by Demographic and Vote Type (Weighted by Population)', fontsize=16, y=0.995)
    
    for demo_idx, demo in enumerate(DEMOGRAPHICS):
        # Get population weights for this demographic
        pop_col = f'cvap_{demo}'
        if pop_col not in df.columns:
            print(f"Warning: {pop_col} not found, using uniform weights")
            weights = None
        else:
            weights = df[pop_col].values
        
        for vote_idx, vote_type in enumerate(VOTE_TYPES):
            ax = axes[demo_idx, vote_idx]
            col_name = f'{vote_type}_{demo}_prob'
            
            if col_name in df.columns:
                probs = df[col_name].values
                
                # Filter out NaN values and corresponding weights
                valid_mask = ~np.isnan(probs)
                probs_valid = probs[valid_mask]
                weights_valid = weights[valid_mask] if weights is not None else None
                
                if len(probs_valid) > 0:
                    # Create weighted histogram
                    ax.hist(probs_valid, bins=50, alpha=0.7, color=COLORS[vote_idx], edgecolor='black', weights=weights_valid)
                    ax.set_xlabel('Probability', fontsize=9)
                    ax.set_ylabel('Weighted Frequency', fontsize=9)
                    ax.set_title(f'{vote_type} - {demo}', fontsize=10, fontweight='bold')
                    
                    # Add weighted statistics
                    if weights_valid is not None:
                        mean_prob = np.average(probs_valid, weights=weights_valid)
                        # For weighted median, we need to compute it differently
                        sorted_indices = np.argsort(probs_valid)
                        sorted_probs = probs_valid[sorted_indices]
                        sorted_weights = weights_valid[sorted_indices]
                        cumsum_weights = np.cumsum(sorted_weights)
                        median_idx = np.searchsorted(cumsum_weights, cumsum_weights[-1] / 2)
                        median_prob = sorted_probs[median_idx] if median_idx < len(sorted_probs) else sorted_probs[-1]
                    else:
                        mean_prob = probs_valid.mean()
                        median_prob = np.median(probs_valid)
                    
                    ax.axvline(mean_prob, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_prob:.3f}')
                    ax.axvline(median_prob, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_prob:.3f}')
                    ax.legend(fontsize=7, loc='upper right')
                    ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'{col_name}\nnot found', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'probability_distributions_by_demo_vote.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.close()
    
    # Create summary statistics table (weighted)
    print("\nSummary Statistics (Weighted by Population):")
    print("=" * 80)
    stats_data = []
    for demo in DEMOGRAPHICS:
        pop_col = f'cvap_{demo}'
        weights = df[pop_col].values if pop_col in df.columns else None
        
        for vote_type in VOTE_TYPES:
            col_name = f'{vote_type}_{demo}_prob'
            if col_name in df.columns:
                probs = df[col_name].values
                valid_mask = ~np.isnan(probs)
                probs_valid = probs[valid_mask]
                weights_valid = weights[valid_mask] if weights is not None else None
                
                if len(probs_valid) > 0:
                    if weights_valid is not None:
                        mean_prob = np.average(probs_valid, weights=weights_valid)
                        # Weighted std
                        variance = np.average((probs_valid - mean_prob)**2, weights=weights_valid)
                        std_prob = np.sqrt(variance)
                        # Weighted median
                        sorted_indices = np.argsort(probs_valid)
                        sorted_probs = probs_valid[sorted_indices]
                        sorted_weights = weights_valid[sorted_indices]
                        cumsum_weights = np.cumsum(sorted_weights)
                        median_idx = np.searchsorted(cumsum_weights, cumsum_weights[-1] / 2)
                        median_prob = sorted_probs[median_idx] if median_idx < len(sorted_probs) else sorted_probs[-1]
                    else:
                        mean_prob = probs_valid.mean()
                        median_prob = np.median(probs_valid)
                        std_prob = probs_valid.std()
                    
                    stats_data.append({
                        'Demographic': demo,
                        'Vote Type': vote_type,
                        'Mean': mean_prob,
                        'Median': median_prob,
                        'Std': std_prob,
                        'Min': probs_valid.min(),
                        'Max': probs_valid.max(),
                        'Count': len(probs_valid)
                    })
    
    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False))
    
    # Save statistics to CSV
    stats_file = os.path.join(output_dir, 'probability_statistics.csv')
    stats_df.to_csv(stats_file, index=False)
    print(f"\nSaved statistics to {stats_file}")
    
    # Create additional plot: Box plots by demographic (weighted)
    print("\nCreating weighted box plot...")
    fig, axes = plt.subplots(1, len(DEMOGRAPHICS), figsize=(20, 6))
    fig.suptitle('Probability Distributions by Demographic (Box Plots, Weighted by Population)', fontsize=16)
    
    for demo_idx, demo in enumerate(DEMOGRAPHICS):
        ax = axes[demo_idx]
        demo_data = []
        demo_weights = []
        labels = []
        
        pop_col = f'cvap_{demo}'
        weights = df[pop_col].values if pop_col in df.columns else None
        
        for vote_type in VOTE_TYPES:
            col_name = f'{vote_type}_{demo}_prob'
            if col_name in df.columns:
                probs = df[col_name].values
                valid_mask = ~np.isnan(probs)
                probs_valid = probs[valid_mask]
                weights_valid = weights[valid_mask] if weights is not None else None
                
                if len(probs_valid) > 0:
                    demo_data.append(probs_valid)
                    demo_weights.append(weights_valid)
                    labels.append(vote_type)
        
        if demo_data:
            # Create box plot (matplotlib boxplot doesn't directly support weights, but we can sample)
            # For visualization, we'll use regular boxplot but note it's from weighted data
            bp = ax.boxplot(demo_data, tick_labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], COLORS[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Probability', fontsize=10)
            ax.set_title(f'{demo}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'probability_boxplots_by_demo.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved box plot to {output_file}")
    plt.close()
    
    # Create density plots (weighted)
    print("\nCreating weighted density plots...")
    fig, axes = plt.subplots(len(DEMOGRAPHICS), 1, figsize=(12, 16))
    fig.suptitle('Probability Density by Demographic (All Vote Types, Weighted by Population)', fontsize=16)
    
    for demo_idx, demo in enumerate(DEMOGRAPHICS):
        ax = axes[demo_idx]
        
        pop_col = f'cvap_{demo}'
        weights = df[pop_col].values if pop_col in df.columns else None
        
        for vote_idx, vote_type in enumerate(VOTE_TYPES):
            col_name = f'{vote_type}_{demo}_prob'
            if col_name in df.columns:
                probs = df[col_name].values
                valid_mask = ~np.isnan(probs)
                probs_valid = probs[valid_mask]
                weights_valid = weights[valid_mask] if weights is not None else None
                
                if len(probs_valid) > 0:
                    ax.hist(probs_valid, bins=50, alpha=0.5, label=vote_type, 
                           color=COLORS[vote_idx], weights=weights_valid, density=True)
        
        ax.set_xlabel('Probability', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{demo}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'probability_density_by_demo.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved density plot to {output_file}")
    plt.close()
    
    print(f"\nAll plots saved to '{output_dir}' directory")

def main():
    parser = argparse.ArgumentParser(description="Plot probability distributions by demographic and vote type")
    parser.add_argument("csv_file", nargs='?', default="output/mle_l2_logit_results.csv",
                       help="Input CSV file with probability columns (default: output/mle_l2_logit_results.csv)")
    parser.add_argument("--output-dir", default="output/plots",
                       help="Output directory for plots (default: output/plots)")
    args = parser.parse_args()
    
    plot_distributions(args.csv_file, args.output_dir)

if __name__ == '__main__':
    main()
