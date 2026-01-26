"""
Calculate vote type counts by race/demographic from estimates and demographic data.

This script:
1. Loads probability estimates from ny_estimates.csv
2. Loads CVAP demographic data from 36_data.csv
3. Calculates vote counts for each block group: votes = probability × CVAP population
4. Aggregates totals by vote type (D, R, O, N) and demographic (Wht, His, Blk, Asn, Oth)
5. Outputs summary statistics and a CSV with detailed counts
"""

import argparse
import os
import pandas as pd
import numpy as np

def load_and_prepare_data(estimates_csv, demographic_csv):
    """
    Load and merge estimates and demographic data.
    
    Args:
        estimates_csv: Path to ny_estimates.csv with probability estimates
        demographic_csv: Path to 36_data.csv with CVAP demographic data
        
    Returns:
        merged_df: DataFrame with merged data
    """
    print(f"Loading estimates from {estimates_csv}...")
    estimates_df = pd.read_csv(estimates_csv, dtype={'AFFGEOID': str})
    print(f"  Loaded {len(estimates_df)} block groups from estimates file.")
    
    print(f"Loading demographic data from {demographic_csv}...")
    demo_df = pd.read_csv(demographic_csv, dtype={'AFFGEOID': str})
    print(f"  Loaded {len(demo_df)} block groups from demographic file.")
    
    # Filter to NY state if new_state column exists
    if 'new_state' in demo_df.columns:
        initial_count = len(demo_df)
        demo_df = demo_df[demo_df['new_state'] == 36]
        print(f"  Filtered to NY state: {len(demo_df)} rows (from {initial_count}).")
    
    # Rename CVAP columns to match the standard naming
    column_mapping = {
        'cvap_est_White Alone': 'cvap_Wht',
        'cvap_est_Hispanic or Latino': 'cvap_His',
        'cvap_est_Black or African American Alone': 'cvap_Blk',
        'cvap_est_Asian Alone': 'cvap_Asn',
        'cvap_est_American Indian or Alaska Native Alone': 'cvap_aian',
        'cvap_est_Native Hawaiian or Other Pacific Islander Alone': 'cvap_nhpi',
        'cvap_est_Mixed': 'cvap_sor'
    }
    
    # Only rename columns that exist
    existing_mappings = {k: v for k, v in column_mapping.items() if k in demo_df.columns}
    demo_df.rename(columns=existing_mappings, inplace=True)
    
    # Calculate cvap_Oth as sum of aian, nhpi, sor
    if all(col in demo_df.columns for col in ['cvap_aian', 'cvap_nhpi', 'cvap_sor']):
        demo_df['cvap_Oth'] = (demo_df['cvap_aian'].fillna(0) + 
                               demo_df['cvap_nhpi'].fillna(0) + 
                               demo_df['cvap_sor'].fillna(0))
    elif 'cvap_Oth' not in demo_df.columns:
        print("Warning: Could not calculate cvap_Oth. Setting to 0.")
        demo_df['cvap_Oth'] = 0
    
    # Select only necessary columns from demographic data
    demo_cols = ['AFFGEOID', 'cvap_Wht', 'cvap_His', 'cvap_Blk', 'cvap_Asn', 'cvap_Oth']
    demo_cols = [col for col in demo_cols if col in demo_df.columns]
    demo_df = demo_df[demo_cols]
    
    # Merge on AFFGEOID
    print("Merging estimates and demographic data...")
    merged_df = estimates_df.merge(demo_df, on='AFFGEOID', how='inner')
    print(f"  Merged to {len(merged_df)} block groups.")
    
    if len(merged_df) == 0:
        raise ValueError("No matching block groups found after merge. Check AFFGEOID format.")
    
    return merged_df

def calculate_vote_counts(df):
    """
    Calculate vote counts by vote type and demographic for each block group.
    
    Args:
        df: Merged DataFrame with probabilities and CVAP populations
        
    Returns:
        df: DataFrame with added vote count columns
    """
    print("Calculating vote counts by demographic and vote type...")
    
    demographics = ['Wht', 'His', 'Blk', 'Asn', 'Oth']
    vote_types = ['D', 'R', 'O', 'N']
    
    # Calculate votes for each combination
    for demo in demographics:
        cvap_col = f'cvap_{demo}'
        if cvap_col not in df.columns:
            print(f"Warning: {cvap_col} not found. Skipping {demo} demographic.")
            continue
        
        for vtype in vote_types:
            prob_col = f'{vtype}_{demo}_prob'
            if prob_col not in df.columns:
                print(f"Warning: {prob_col} not found. Skipping {vtype} votes for {demo}.")
                continue
            
            # Calculate votes: probability × CVAP population
            votes_col = f'votes_{vtype}_{demo}'
            df[votes_col] = df[prob_col] * df[cvap_col].fillna(0)
    
    return df

def aggregate_totals(df):
    """
    Aggregate vote counts by vote type and demographic.
    
    Args:
        df: DataFrame with vote count columns
        
    Returns:
        summary_df: DataFrame with aggregated totals
    """
    print("Aggregating vote counts...")
    
    demographics = ['Wht', 'His', 'Blk', 'Asn', 'Oth']
    vote_types = ['D', 'R', 'O', 'N']
    
    # Create summary dictionary
    summary_data = []
    
    for demo in demographics:
        for vtype in vote_types:
            votes_col = f'votes_{vtype}_{demo}'
            if votes_col in df.columns:
                total_votes = df[votes_col].sum()
                summary_data.append({
                    'Demographic': demo,
                    'Vote_Type': vtype,
                    'Total_Votes': total_votes
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Pivot for better readability
    if len(summary_df) > 0:
        pivot_df = summary_df.pivot(index='Demographic', columns='Vote_Type', values='Total_Votes')
        pivot_df = pivot_df.fillna(0)
        pivot_df['Total'] = pivot_df.sum(axis=1)
        pivot_df.loc['Total'] = pivot_df.sum(axis=0)
    else:
        pivot_df = pd.DataFrame()
    
    return summary_df, pivot_df

def main():
    parser = argparse.ArgumentParser(
        description="Calculate vote type counts by race from estimates and demographic data."
    )
    parser.add_argument(
        "estimates_csv",
        help="Path to ny_estimates.csv with probability estimates."
    )
    parser.add_argument(
        "demographic_csv",
        help="Path to 36_data.csv with CVAP demographic data."
    )
    parser.add_argument(
        "--output",
        default="output/vote_counts_by_race.csv",
        help="Path for output CSV file with detailed counts."
    )
    parser.add_argument(
        "--summary-output",
        default="output/vote_counts_summary.csv",
        help="Path for summary CSV file with aggregated totals."
    )
    parser.add_argument(
        "--pivot-output",
        default="output/vote_counts_pivot.csv",
        help="Path for pivot table CSV file."
    )
    args = parser.parse_args()
    
    # Load and merge data
    merged_df = load_and_prepare_data(args.estimates_csv, args.demographic_csv)
    
    # Calculate vote counts
    result_df = calculate_vote_counts(merged_df)
    
    # Aggregate totals
    summary_df, pivot_df = aggregate_totals(result_df)
    
    # Save detailed results
    for p in (args.output, args.summary_output, args.pivot_output):
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)
    print(f"\nSaving detailed results to {args.output}...")
    result_df.to_csv(args.output, index=False)
    print(f"  Saved {len(result_df)} block groups with vote counts.")
    
    # Save summary
    print(f"Saving summary to {args.summary_output}...")
    summary_df.to_csv(args.summary_output, index=False)
    print(f"  Saved {len(summary_df)} vote type/demographic combinations.")
    
    # Save pivot table
    if len(pivot_df) > 0:
        print(f"Saving pivot table to {args.pivot_output}...")
        pivot_df.to_csv(args.pivot_output)
        print("\n--- Vote Counts Summary (Pivot Table) ---")
        print(pivot_df.to_string())
    
    # Print summary statistics
    print("\n--- Summary Statistics ---")
    demographics = ['Wht', 'His', 'Blk', 'Asn', 'Oth']
    vote_types = ['D', 'R', 'O', 'N']
    
    for demo in demographics:
        demo_total = 0
        for vtype in vote_types:
            votes_col = f'votes_{vtype}_{demo}'
            if votes_col in result_df.columns:
                total = result_df[votes_col].sum()
                demo_total += total
                print(f"  {vtype} votes ({demo}): {total:,.0f}")
        print(f"  Total votes ({demo}): {demo_total:,.0f}\n")
    
    print("\n--- Success! ---")
    print(f"Output files:")
    print(f"  - Detailed counts: {args.output}")
    print(f"  - Summary: {args.summary_output}")
    print(f"  - Pivot table: {args.pivot_output}")

if __name__ == "__main__":
    main()

