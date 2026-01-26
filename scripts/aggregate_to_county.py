"""
Aggregate vote counts by demographic to county level.

Loads vote count estimates from optimization output and aggregates them by county,
demographic, and vote type. Outputs a pivot table format.

- Input:
    - CSV file with vote counts (from optimize_gradient_descent.py output)
- Output:
    - CSV file with county-level aggregated vote counts in pivot table format
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np

def extract_county_fips(affgeoid):
    """
    Extract county FIPS code from AFFGEOID.
    
    AFFGEOID format: 1500000US36XXX... where XXX is the county FIPS (3 digits).
    County FIPS is at positions 11-13 (0-indexed: 11, 12, 13).
    
    Args:
        affgeoid: AFFGEOID string (e.g., "1500000US360010001001")
        
    Returns:
        county_fips: County FIPS code as string (e.g., "001")
    """
    if not isinstance(affgeoid, str) or len(affgeoid) < 14:
        return None
    
    # Extract positions 11-13 (0-indexed: 11, 12, 13)
    # Format: 1500000US36XXX... where XXX is county
    try:
        county_fips = affgeoid[11:14]
        return county_fips
    except:
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate vote counts by demographic to county level."
    )
    parser.add_argument(
        "input_csv",
        help="Path to input CSV file with vote counts (from optimize_gradient_descent.py)."
    )
    parser.add_argument(
        "--output",
        default="output/county_vote_counts.csv",
        help="Path for output CSV file with county-level aggregated vote counts."
    )
    parser.add_argument(
        "--state-fips",
        default="36",
        help="State FIPS code (default: 36 for NY)."
    )
    args = parser.parse_args()
    
    # Load input data
    print(f"Loading vote counts from {args.input_csv}...")
    try:
        df = pd.read_csv(args.input_csv, dtype={'AFFGEOID': str})
    except Exception as e:
        print(f"Error loading input file: {e}. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(df)} block groups.")
    
    # Extract county FIPS codes
    print("Extracting county FIPS codes from AFFGEOID...")
    df['county_fips'] = df['AFFGEOID'].apply(extract_county_fips)
    
    # Filter out rows where county extraction failed
    initial_count = len(df)
    df = df[df['county_fips'].notna()]
    if len(df) < initial_count:
        print(f"Warning: Could not extract county FIPS for {initial_count - len(df)} block groups. Excluding them.")
    
    # Create full county FIPS (state + county)
    df['full_county_fips'] = args.state_fips + df['county_fips']
    
    print(f"Found {df['county_fips'].nunique()} unique counties.")
    
    # Identify vote count columns
    vote_count_cols = [col for col in df.columns if col.startswith('votes_')]
    
    if not vote_count_cols:
        print("Error: No vote count columns found (expected columns starting with 'votes_').", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(vote_count_cols)} vote count columns.")
    
    # Parse vote count columns to extract vote type and demographic
    # Format: votes_D_Wht, votes_R_His, etc.
    vote_data = []
    
    for col in vote_count_cols:
        parts = col.split('_')
        if len(parts) >= 3 and parts[0] == 'votes':
            vote_type = parts[1]  # D, R, O, N
            demographic = '_'.join(parts[2:])  # Wht, His, Blk, Asn, Oth
            vote_data.append({
                'column': col,
                'vote_type': vote_type,
                'demographic': demographic
            })
    
    if not vote_data:
        print("Error: Could not parse vote count columns.", file=sys.stderr)
        sys.exit(1)
    
    # Aggregate by county, demographic, and vote type
    print("Aggregating vote counts by county, demographic, and vote type...")
    
    aggregation_data = []
    
    for county_fips in df['full_county_fips'].unique():
        county_df = df[df['full_county_fips'] == county_fips]
        
        for vote_info in vote_data:
            col = vote_info['column']
            vote_type = vote_info['vote_type']
            demographic = vote_info['demographic']
            
            # Sum votes for this county, demographic, vote type
            total_votes = county_df[col].sum()
            
            aggregation_data.append({
                'county_fips': county_fips,
                'demographic': demographic,
                'vote_type': vote_type,
                'votes': total_votes
            })
    
    # Create DataFrame from aggregation data
    agg_df = pd.DataFrame(aggregation_data)
    
    # Create pivot table: County × Demographic with vote type columns
    print("Creating pivot table...")
    
    pivot_df = agg_df.pivot_table(
        index=['county_fips', 'demographic'],
        columns='vote_type',
        values='votes',
        fill_value=0.0
    )
    
    # Reset index to make county_fips and demographic regular columns
    pivot_df = pivot_df.reset_index()
    
    # Rename columns to include 'votes_' prefix for clarity
    vote_types = ['D', 'R', 'O', 'N']
    for vtype in vote_types:
        if vtype in pivot_df.columns:
            pivot_df = pivot_df.rename(columns={vtype: f'votes_{vtype}'})
    
    # Sort by county FIPS and demographic
    pivot_df = pivot_df.sort_values(['county_fips', 'demographic'])
    
    # Save to CSV
    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    print(f"Saving pivot table to {args.output}...")
    pivot_df.to_csv(args.output, index=False)
    
    print(f"\nOutput saved to {args.output}")
    print(f"Final DataFrame shape: {pivot_df.shape}")
    print(f"Counties: {pivot_df['county_fips'].nunique()}")
    print(f"Demographics: {pivot_df['demographic'].nunique()}")
    
    # Print summary statistics
    print("\n--- Summary Statistics ---")
    for demo in pivot_df['demographic'].unique():
        demo_df = pivot_df[pivot_df['demographic'] == demo]
        print(f"\n{demo}:")
        for vtype in vote_types:
            col = f'votes_{vtype}'
            if col in demo_df.columns:
                total = demo_df[col].sum()
                print(f"  {vtype} votes: {total:,.0f}")
    
    print("\n--- Success! ---")

if __name__ == "__main__":
    main()

