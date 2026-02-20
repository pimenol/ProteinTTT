import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def plot_mean_scores_vs_step(logs_dir, output_path=None, metric='plddt', choose_best_plddt=True):
    """
    Create plots showing mean of a chosen metric (plddt, lddt, or tmscore) across all proteins at each step, 
    organized by hyperparameter combinations.
    
    Args:
        logs_dir: Path to directory containing log files (e.g., 'logs_msa_0.0004_4')
        hyperparams: Optional dict with parameter mappings if you want to organize by params
                    e.g., {'lr': [4e-5, 4e-4, 4e-3], 'ags': [4, 8, 16, 32]}
        output_path: Optional path to save the figure
        metric: The metric to plot. Can be 'plddt', 'lddt', or 'tm_score'. Defaults to 'plddt'.
        choose_best_plddt: Whether to choose the best pLDDT for each protein at each step. Defaults to True.
    """
    
    # Read all log files
    log_files = glob.glob(str(Path(logs_dir) / '*_log.tsv'))
    print(len(log_files))
    if not log_files:
        print(f"No log files found in {logs_dir}")
        return
    
    # Collect data from all log files, adding protein identifier
    all_data = []
    for idx, log_file in enumerate(log_files):
        try:
            # Read with quoting to handle multiline pdb column
            df = pd.read_csv(log_file, sep='\t', quoting=1)  # QUOTE_MINIMAL
            # Add protein identifier based on file index
            df['protein_id'] = idx
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
            continue
    
    if not all_data:
        print("No valid log files could be read")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Validate metric
    if metric not in ['plddt', 'lddt', 'tm_score']:
        print(f"Error: Invalid metric '{metric}'. Choose from 'plddt', 'lddt', or 'tm_score'.")
        return None # Return None in case of error
    if choose_best_plddt:
        steps = sorted(combined_df['step'].unique())
        mean_values = []
        
        for current_step in steps:
            step_metrics = []
            
            # For each protein, find the step with highest pLDDT up to current_step
            for protein_id in combined_df['protein_id'].unique():
                protein_data = combined_df[
                    (combined_df['protein_id'] == protein_id) & 
                    (combined_df['step'] <= current_step)
                ]
                
                if len(protein_data) == 0:
                    continue
                
                # Find the row with highest pLDDT
                best_row = protein_data.loc[protein_data['plddt'].idxmax()]
                
                # Take the metric value from that row
                step_metrics.append(best_row[metric])
            
            if step_metrics:
                mean_values.append(np.mean(step_metrics))
            else:
                mean_values.append(np.nan)
        
        mean_metric = pd.Series(mean_values, index=steps)
    else:
        mean_metric = combined_df.groupby('step')[metric].mean()
        
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mean_metric.index, mean_metric.values, 
            'o-', color='green', linewidth=2, markersize=5, label=f'Mean {metric.upper()}')
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel(f'Mean {metric.upper()}', fontsize=12) 
    ax.tick_params(axis='y')
    ax.set_title(f'Mean {metric.upper()} vs Step', fontsize=14) 
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return mean_metric
