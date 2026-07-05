"""
Step 6: Modality comparison summary + heatmap.
"""
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = r"d:\BTP\btp2\salmamaterials"
MODALITIES = ['VIDEO', 'AUDIO', 'INTERSECT']

def compare():
    print("Building modality comparison...")
    
    # Load all sigma + stats results
    sigma_dfs = {}
    stats_dfs = {}
    for mod in MODALITIES:
        sigma_dfs[mod] = pd.read_csv(os.path.join(BASE_DIR, f"_sigma_results_{mod}.csv"))
        stats_dfs[mod] = pd.read_csv(os.path.join(BASE_DIR, f"_statistical_tests_{mod}.csv"))
    
    tau = 50
    # Get all unique hyp_ids across all modalities
    all_hyps = set()
    for mod in MODALITIES:
        s = sigma_dfs[mod][sigma_dfs[mod]['tau'] == tau]
        all_hyps |= set(s['hyp_id'].unique())
    all_hyps = sorted(all_hyps)
    
    rows = []
    for hyp in all_hyps:
        row = {'hyp_id': hyp, 'tau': tau}
        directions = []
        for mod in MODALITIES:
            s = sigma_dfs[mod][(sigma_dfs[mod]['hyp_id'] == hyp) & (sigma_dfs[mod]['tau'] == tau)]
            st = stats_dfs[mod][(stats_dfs[mod]['hyp_id'] == hyp) & (stats_dfs[mod]['tau'] == tau)]
            
            s_sw = s.iloc[0]['sigma_switching'] if len(s) > 0 else np.nan
            s_ns = s.iloc[0]['sigma_nonswitching'] if len(s) > 0 else np.nan
            sig = st.iloc[0]['significant'] if len(st) > 0 else np.nan
            
            row[f'sw_{mod}'] = s_sw
            row[f'ns_{mod}'] = s_ns
            row[f'sig_{mod}'] = sig
            
            if not pd.isna(s_sw) and not pd.isna(s_ns):
                directions.append('sw>ns' if s_sw > s_ns else 'sw<ns' if s_sw < s_ns else 'eq')
            else:
                directions.append(None)
        
        valid_dirs = [d for d in directions if d is not None]
        row['consistent'] = len(set(valid_dirs)) <= 1 if valid_dirs else np.nan
        rows.append(row)
    
    comp_df = pd.DataFrame(rows)
    out = os.path.join(BASE_DIR, "_modality_comparison_summary.csv")
    comp_df.to_csv(out, index=False)
    print(f"  Saved comparison to: {out}")
    print(f"  {len(comp_df)} hypotheses compared.")
    
    # Heatmap
    plot_dir = os.path.join(BASE_DIR, "_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(all_hyps)*0.4)))
    
    # Panel 1: sigma_switching
    sw_cols = [f'sw_{m}' for m in MODALITIES]
    sw_data = comp_df[sw_cols].copy()
    sw_data.index = comp_df['hyp_id']
    sw_data.columns = MODALITIES
    sns.heatmap(sw_data.astype(float), annot=True, cmap="YlOrRd", fmt='.3f', ax=axes[0],
                linewidths=0.5)
    axes[0].set_title('Sigma Switching')
    
    # Panel 2: sigma_nonswitching
    ns_cols = [f'ns_{m}' for m in MODALITIES]
    ns_data = comp_df[ns_cols].copy()
    ns_data.index = comp_df['hyp_id']
    ns_data.columns = MODALITIES
    sns.heatmap(ns_data.astype(float), annot=True, cmap="YlGnBu", fmt='.3f', ax=axes[1],
                linewidths=0.5)
    axes[1].set_title('Sigma Non-Switching')
    
    fig.suptitle('Modality Comparison at tau=0.50')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, '_modality_comparison_heatmap.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved heatmap to: {plot_dir}/_modality_comparison_heatmap.png")

if __name__ == "__main__":
    compare()
