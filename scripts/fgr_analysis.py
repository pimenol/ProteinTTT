"""
FGR Analysis and Stabilization Demo

This script analyzes the FGR (Fidelity-Gain Ratio) results from different configurations
and demonstrates how the enhanced FGR implementation stabilizes the stopping criterion.

Key issues with original FGR:
1. Step-wise ratio oscillates wildly due to noisy loss/drift signals
2. Single negative ratio triggers stopping (no patience)
3. No warm-up period - can stop too early before model adapts
4. Division by very small drift_delta causes ratio explosion

Enhanced FGR solutions:
1. Cumulative ratio (total_gain / total_drift) instead of step-wise
2. EMA smoothing for loss and drift
3. Patience mechanism (consecutive negatives required)
4. Warm-up period before stopping is allowed
5. Numerical stability guards
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def compute_enhanced_fgr(df: pd.DataFrame, 
                          ema_decay: float = 0.9,
                          warmup_steps: int = 5,
                          patience: int = 3,
                          ratio_threshold: float = 0.0,
                          drift_threshold: float = 0.1,
                          min_drift: float = 1e-6) -> pd.DataFrame:
    """
    Recompute FGR metrics with enhanced stabilization on existing data.
    
    This simulates what the new FGR implementation would produce.
    """
    df = df.copy()
    
    # Initialize tracking
    initial_loss = None
    ema_loss = None
    negative_count = 0
    
    # New columns
    df['fgr_ratio_cumulative'] = None
    df['fgr_ema_loss'] = None
    df['fgr_ema_ratio'] = None
    df['fgr_stop_ratio_enhanced'] = False
    df['fgr_negative_count'] = 0
    
    for idx, row in df.iterrows():
        step = row['step']
        loss = row['loss']
        drift = row.get('fgr_drift', 0.0)
        
        if step == 0:
            initial_loss = loss
            ema_loss = loss
            df.at[idx, 'fgr_ema_loss'] = ema_loss
            df.at[idx, 'fgr_ratio_cumulative'] = None
            continue
        
        # Update EMA loss
        if loss is not None and ema_loss is not None:
            ema_loss = ema_decay * ema_loss + (1 - ema_decay) * loss
        elif loss is not None:
            ema_loss = loss
        
        df.at[idx, 'fgr_ema_loss'] = ema_loss
        
        # Compute cumulative ratio
        if initial_loss is not None and loss is not None and drift is not None:
            cumulative_gain = initial_loss - loss
            if drift > min_drift:
                cumulative_ratio = cumulative_gain / drift
                df.at[idx, 'fgr_ratio_cumulative'] = cumulative_ratio
                
                # EMA ratio
                if ema_loss is not None:
                    ema_gain = initial_loss - ema_loss
                    df.at[idx, 'fgr_ema_ratio'] = ema_gain / drift
                
                # Update patience counter
                if cumulative_ratio < ratio_threshold:
                    negative_count += 1
                else:
                    negative_count = 0
                
                df.at[idx, 'fgr_negative_count'] = negative_count
                
                # Enhanced stopping: requires warmup AND patience
                should_stop = (step >= warmup_steps and 
                               negative_count >= patience)
                df.at[idx, 'fgr_stop_ratio_enhanced'] = should_stop
    
    return df


def analyze_protein(name: str, data_dirs: dict) -> dict:
    """Analyze FGR results for a single protein across different configurations."""
    results = {}
    
    for config_name, data_dir in data_dirs.items():
        file_path = Path(data_dir) / f"{name}_ttt.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df_enhanced = compute_enhanced_fgr(df)
            
            # Count premature stops with original vs enhanced
            original_stops = df['fgr_stop_ratio'].sum() if 'fgr_stop_ratio' in df.columns else 0
            enhanced_stops = df_enhanced['fgr_stop_ratio_enhanced'].sum()
            
            # Find first stop step
            original_first_stop = df[df['fgr_stop_ratio'] == True]['step'].min() if original_stops > 0 else None
            enhanced_first_stop = df_enhanced[df_enhanced['fgr_stop_ratio_enhanced'] == True]['step'].min() if enhanced_stops > 0 else None
            
            # Get final metrics
            final_row = df.iloc[-1]
            
            results[config_name] = {
                'df': df,
                'df_enhanced': df_enhanced,
                'original_stop_count': original_stops,
                'enhanced_stop_count': enhanced_stops,
                'original_first_stop': original_first_stop,
                'enhanced_first_stop': enhanced_first_stop,
                'final_tm_score': final_row.get('tm_score', None),
                'final_plddt': final_row.get('plddt', None),
                'final_drift': final_row.get('fgr_drift', None),
            }
    
    return results


def print_analysis_summary(results: dict, protein_name: str):
    """Print summary of FGR analysis for a protein."""
    print(f"\n{'='*60}")
    print(f"FGR Analysis: {protein_name}")
    print('='*60)
    
    for config, data in results.items():
        print(f"\n[{config}]")
        print(f"  Original stop triggers: {data['original_stop_count']} steps")
        print(f"  Enhanced stop triggers: {data['enhanced_stop_count']} steps")
        print(f"  Original first stop at step: {data['original_first_stop']}")
        print(f"  Enhanced first stop at step: {data['enhanced_first_stop']}")
        print(f"  Final TM-score: {data['final_tm_score']:.4f}" if data['final_tm_score'] else "  Final TM-score: N/A")
        print(f"  Final pLDDT: {data['final_plddt']:.2f}" if data['final_plddt'] else "  Final pLDDT: N/A")
        print(f"  Final drift: {data['final_drift']:.6f}" if data['final_drift'] else "  Final drift: N/A")
        
        # Show ratio stability comparison
        df_enh = data['df_enhanced']
        if 'fgr_ratio' in df_enh.columns and 'fgr_ratio_cumulative' in df_enh.columns:
            step_ratios = df_enh['fgr_ratio'].dropna()
            cum_ratios = df_enh['fgr_ratio_cumulative'].dropna()
            if len(step_ratios) > 0:
                print(f"  Step-wise ratio std: {step_ratios.std():.2f} (range: {step_ratios.min():.1f} to {step_ratios.max():.1f})")
            if len(cum_ratios) > 0:
                print(f"  Cumulative ratio std: {cum_ratios.std():.2f} (range: {cum_ratios.min():.1f} to {cum_ratios.max():.1f})")


def plot_fgr_comparison(results: dict, protein_name: str, save_path: str = None):
    """Plot comparison of original vs enhanced FGR metrics."""
    n_configs = len(results)
    fig, axes = plt.subplots(n_configs, 3, figsize=(15, 4*n_configs))
    if n_configs == 1:
        axes = [axes]
    
    for ax_row, (config, data) in zip(axes, results.items()):
        df = data['df_enhanced']
        steps = df['step'].values
        
        # Plot 1: Loss and EMA Loss
        ax1 = ax_row[0]
        if 'loss' in df.columns:
            ax1.plot(steps, df['loss'], 'b-', alpha=0.5, label='Loss (raw)')
        if 'fgr_ema_loss' in df.columns:
            ax1.plot(steps, df['fgr_ema_loss'], 'b-', linewidth=2, label='Loss (EMA)')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{config}: Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Step-wise vs Cumulative Ratio
        ax2 = ax_row[1]
        if 'fgr_ratio' in df.columns:
            ratios = df['fgr_ratio'].values
            ax2.plot(steps, ratios, 'r-', alpha=0.5, label='Step-wise ratio')
        if 'fgr_ratio_cumulative' in df.columns:
            cum_ratios = df['fgr_ratio_cumulative'].values
            ax2.plot(steps, cum_ratios, 'g-', linewidth=2, label='Cumulative ratio')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('FGR Ratio')
        ax2.set_title(f'{config}: Ratio Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Limit y-axis for readability
        if 'fgr_ratio_cumulative' in df.columns:
            cum_valid = df['fgr_ratio_cumulative'].dropna()
            if len(cum_valid) > 0:
                y_center = cum_valid.median()
                y_range = max(abs(cum_valid.max() - y_center), abs(cum_valid.min() - y_center)) * 2
                ax2.set_ylim(y_center - y_range, y_center + y_range)
        
        # Plot 3: Drift and Stop Triggers
        ax3 = ax_row[2]
        if 'fgr_drift' in df.columns:
            ax3.plot(steps, df['fgr_drift'], 'purple', linewidth=2, label='Drift')
        
        # Mark original stops
        if 'fgr_stop_ratio' in df.columns:
            orig_stops = df[df['fgr_stop_ratio'] == True]
            ax3.scatter(orig_stops['step'], orig_stops['fgr_drift'], 
                       c='red', s=100, marker='x', label='Original stop', zorder=5)
        
        # Mark enhanced stops
        if 'fgr_stop_ratio_enhanced' in df.columns:
            enh_stops = df[df['fgr_stop_ratio_enhanced'] == True]
            ax3.scatter(enh_stops['step'], enh_stops['fgr_drift'], 
                       c='green', s=100, marker='o', label='Enhanced stop', zorder=5)
        
        ax3.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Drift threshold')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Drift')
        ax3.set_title(f'{config}: Drift & Stop Triggers')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f'FGR Analysis: {protein_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def main():
    """Main analysis function."""
    base_path = Path("/scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT/notebooks/fgr")
    
    data_dirs = {
        'mean_loss_grad': base_path / 'mean_loss_grad',
        'ema_grad': base_path / 'ema_grad',
        'gpt': base_path / 'gpt',
    }
    
    # Get list of all proteins
    proteins = set()
    for dir_path in data_dirs.values():
        if dir_path.exists():
            for f in dir_path.glob('*_ttt.csv'):
                protein_name = f.stem.replace('_ttt', '')
                proteins.add(protein_name)
    
    print("="*70)
    print("FGR (Fidelity-Gain Ratio) Stabilization Analysis")
    print("="*70)
    print("\nKey improvements in enhanced FGR:")
    print("  1. Cumulative ratio (total_gain/total_drift) vs noisy step-wise")
    print("  2. EMA smoothing (decay=0.9) for loss tracking")
    print("  3. Warm-up period (5 steps) before stopping is allowed")
    print("  4. Patience mechanism (3 consecutive negative ratios)")
    print("  5. Numerical stability guards (min_drift=1e-6)")
    
    # Analyze each protein
    all_results = {}
    for protein in sorted(proteins):
        results = analyze_protein(protein, data_dirs)
        if results:
            all_results[protein] = results
            print_analysis_summary(results, protein)
    
    # Summary statistics across all proteins
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    total_original_stops = 0
    total_enhanced_stops = 0
    
    for protein, results in all_results.items():
        for config, data in results.items():
            total_original_stops += data['original_stop_count']
            total_enhanced_stops += data['enhanced_stop_count']
    
    print(f"\nTotal premature stops across all proteins/configs:")
    print(f"  Original FGR: {total_original_stops}")
    print(f"  Enhanced FGR: {total_enhanced_stops}")
    print(f"  Reduction: {total_original_stops - total_enhanced_stops} ({100*(total_original_stops - total_enhanced_stops)/max(1, total_original_stops):.1f}%)")
    
    # Generate plots for a few example proteins
    print("\n" + "="*70)
    print("Generating comparison plots...")
    print("="*70)
    
    plot_dir = base_path / 'analysis_plots'
    plot_dir.mkdir(exist_ok=True)
    
    for protein in list(all_results.keys())[:3]:  # Plot first 3 proteins
        results = all_results[protein]
        save_path = plot_dir / f'{protein}_fgr_analysis.png'
        try:
            plot_fgr_comparison(results, protein, str(save_path))
        except Exception as e:
            print(f"Could not generate plot for {protein}: {e}")
    
    return all_results


if __name__ == "__main__":
    results = main()


