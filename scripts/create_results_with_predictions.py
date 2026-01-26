"""
Create a new CSV with all real election data plus predicted margins and turnout by demographic.

Predicted margin for each demo: (D-R)/(D+R+O)
Predicted turnout for each demo: (1-N)
"""
import pandas as pd
import numpy as np
import argparse

def calculate_predictions(df):
    """
    Calculate predicted margin and turnout for each demographic.
    
    Args:
        df: DataFrame with probability columns (D_Wht_prob, R_Wht_prob, etc.)
        
    Returns:
        DataFrame with added prediction columns
    """
    demographics = ['Wht', 'His', 'Blk', 'Asn', 'Oth']
    
    for demo in demographics:
        # Get probability columns for this demographic
        d_col = f'D_{demo}_prob'
        r_col = f'R_{demo}_prob'
        o_col = f'O_{demo}_prob'
        n_col = f'N_{demo}_prob'
        
        # Check if columns exist
        if not all(col in df.columns for col in [d_col, r_col, o_col, n_col]):
            print(f"Warning: Missing probability columns for {demo}")
            continue
        
        # Calculate predicted margin: (D-R)/(D+R+O)
        denominator = df[d_col] + df[r_col] + df[o_col]
        margin = np.where(
            denominator > 0,
            (df[d_col] - df[r_col]) / denominator,
            np.nan
        )
        df[f'pred_margin_{demo}'] = margin
        
        # Calculate predicted turnout: (1-N)
        df[f'pred_turnout_{demo}'] = 1.0 - df[n_col]
    
    return df

def main():
    parser = argparse.ArgumentParser(
        description='Create CSV with real election data plus predicted margins and turnout by demographic'
    )
    parser.add_argument(
        'input_csv',
        help='Input CSV file with MLE probability estimates (e.g., mle_probability_vectors.csv)'
    )
    parser.add_argument(
        'output_csv',
        help='Output CSV file path'
    )
    args = parser.parse_args()
    
    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    
    print("Calculating predicted margins and turnout for each demographic...")
    df = calculate_predictions(df)
    
    print(f"Saving results to {args.output_csv}...")
    df.to_csv(args.output_csv, index=False)
    
    print(f"\nSuccess! Created {args.output_csv}")
    print(f"Added columns:")
    demographics = ['Wht', 'His', 'Blk', 'Asn', 'Oth']
    for demo in demographics:
        print(f"  - pred_margin_{demo}: Predicted margin (D-R)/(D+R+O)")
        print(f"  - pred_turnout_{demo}: Predicted turnout (1-N)")
    
    # Print summary statistics
    print("\nSummary statistics:")
    for demo in demographics:
        margin_col = f'pred_margin_{demo}'
        turnout_col = f'pred_turnout_{demo}'
        if margin_col in df.columns:
            print(f"\n{demo}:")
            print(f"  Margin: mean={df[margin_col].mean():.4f}, std={df[margin_col].std():.4f}")
            print(f"  Turnout: mean={df[turnout_col].mean():.4f}, std={df[turnout_col].std():.4f}")

if __name__ == '__main__':
    main()
