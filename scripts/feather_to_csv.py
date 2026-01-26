import pandas as pd
import os

# Define the directory and file names (simulation_sparse* write to output_dir/final_estimates.feather)
results_dir = 'output/results_sparse_10'
feather_file = 'final_estimates.feather'
csv_file = 'final_estimates_10.csv'

# Construct the full paths
feather_path = os.path.join(results_dir, feather_file)
csv_path = os.path.join(results_dir, csv_file)

# Check if the feather file exists
if not os.path.exists(feather_path):
    print(f"Error: Feather file not found at {feather_path}")
else:
    # Read the feather file
    df = pd.read_feather(feather_path)
    for race in ['Wht', 'Blk', 'His', 'Asn', 'Oth']:
        df[f'margin_{race.lower()}'] = (df[f'D_{race}_prob'] - df[f'R_{race}_prob'])/(df[f'D_{race}_prob'] + df[f'R_{race}_prob'] + 1e-6)

    # Write to a csv file
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    print(f"Successfully converted {feather_path} to {csv_path}")
