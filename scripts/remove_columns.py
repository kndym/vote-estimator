"""
Remove neighbors column and all _prob columns from CSV file.
"""
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Remove neighbors column and all _prob columns from CSV'
    )
    parser.add_argument(
        'input_csv',
        help='Input CSV file path'
    )
    parser.add_argument(
        'output_csv',
        help='Output CSV file path (default: overwrites input)',
        nargs='?',
        default=None
    )
    args = parser.parse_args()
    
    output_path = args.output_csv if args.output_csv else args.input_csv
    
    print(f"Loading {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    
    print(f"Original columns: {len(df.columns)}")
    
    # Find columns to remove
    neighbors_col = 'neighbors' if 'neighbors' in df.columns else None
    prob_cols = [col for col in df.columns if col.endswith('_prob')]
    
    cols_to_remove = []
    if neighbors_col:
        cols_to_remove.append(neighbors_col)
    cols_to_remove.extend(prob_cols)
    
    print(f"Removing {len(cols_to_remove)} columns:")
    if neighbors_col:
        print(f"  - {neighbors_col}")
    print(f"  - {len(prob_cols)} _prob columns")
    
    # Remove columns
    df_clean = df.drop(columns=cols_to_remove)
    
    print(f"Remaining columns: {len(df_clean.columns)}")
    print(f"Saving to {output_path}...")
    
    df_clean.to_csv(output_path, index=False)
    print(f"Success! Removed {len(cols_to_remove)} columns.")

if __name__ == '__main__':
    main()
