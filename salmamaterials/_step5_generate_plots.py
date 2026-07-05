"""
Step 5: Generate 12 plots per modality (36 total).
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

def assign_quadrant(val, aro):
    if pd.isna(val) or pd.isna(aro): return "Q0"
    if val == 3.0 or aro == 3.0: return "Q0"
    if val > 3.0 and aro > 3.0: return "Q1"
    if val < 3.0 and aro > 3.0: return "Q2"
    if val < 3.0 and aro < 3.0: return "Q3"
    if val > 3.0 and aro < 3.0: return "Q4"
    return "Q0"

def get_dist(v1, a1, v2, a2):
    return np.sqrt((v1-v2)**2 + (a1-a2)**2)

sns.set_theme(style="whitegrid")

def gen_plots(modality):
    plot_dir = os.path.join(BASE_DIR, "_plots", modality)
    os.makedirs(plot_dir, exist_ok=True)
    
    fpath = os.path.join(BASE_DIR, f"_consecutive_pair_dataset_{modality}.csv")
    sigma_path = os.path.join(BASE_DIR, f"_sigma_results_{modality}.csv")
    stats_path = os.path.join(BASE_DIR, f"_statistical_tests_{modality}.csv")
    
    df = pd.read_csv(fpath)
    sigma_df = pd.read_csv(sigma_path)
    
    tau = "050"
    b_i1 = f'clip2_B_I1_tau{tau}'
    c1_b_i1 = f'clip1_B_I1_tau{tau}'
    b_i2 = f'clip2_B_I2_tau{tau}'
    comb = f'clip2_combined_tau{tau}'
    
    sw_t = df[df['switching'] == True]
    sw_f = df[df['switching'] == False]
    buckets = ['B1','B2','B3','B4','B5']
    quads = ['Q1','Q2','Q3','Q4']
    
    # 1: Markov Switching
    fig, ax = plt.subplots(figsize=(8,6))
    mat = pd.crosstab(sw_t[c1_b_i1], sw_t[b_i1], normalize='index')
    mat = mat.reindex(index=buckets, columns=buckets, fill_value=0)
    sns.heatmap(mat, annot=True, cmap="YlGnBu", fmt='.2f', ax=ax)
    ax.set_title(f'Markov Transition (Switching) [{modality}]')
    ax.set_xlabel('Clip2 Bucket'); ax.set_ylabel('Clip1 Bucket')
    fig.savefig(os.path.join(plot_dir, '_markov_heatmap_switching.png'), bbox_inches='tight')
    plt.close(fig)
    
    # 2: Markov Non-switching
    fig, ax = plt.subplots(figsize=(8,6))
    mat = pd.crosstab(sw_f[c1_b_i1], sw_f[b_i1], normalize='index')
    mat = mat.reindex(index=buckets, columns=buckets, fill_value=0)
    sns.heatmap(mat, annot=True, cmap="YlGnBu", fmt='.2f', ax=ax)
    ax.set_title(f'Markov Transition (Non-Switching) [{modality}]')
    ax.set_xlabel('Clip2 Bucket'); ax.set_ylabel('Clip1 Bucket')
    fig.savefig(os.path.join(plot_dir, '_markov_heatmap_nonswitching.png'), bbox_inches='tight')
    plt.close(fig)
    
    # 3&4: Quadrant transitions
    df['clip2_I2_quad'] = df.apply(lambda r: assign_quadrant(r['clip2_I2_val'], r['clip2_I2_aro']), axis=1)
    for sw_val, name in [(True, 'switching'), (False, 'nonswitching')]:
        fig, ax = plt.subplots(figsize=(8,6))
        sub = df[(df['switching']==sw_val) & df['clip1_E_quad'].isin(quads) & df['clip2_I2_quad'].isin(quads)]
        if len(sub) > 0:
            mat = pd.crosstab(sub['clip1_E_quad'], sub['clip2_I2_quad'], normalize='index')
            mat = mat.reindex(index=quads, columns=quads, fill_value=0)
            sns.heatmap(mat, annot=True, cmap="OrRd", fmt='.2f', ax=ax)
        ax.set_title(f'Quadrant Transition ({name.capitalize()}) [{modality}]')
        ax.set_xlabel('Clip2 I2 Quadrant'); ax.set_ylabel('Clip1 E Quadrant')
        fig.savefig(os.path.join(plot_dir, f'_quadrant_transition_heatmap_{name}.png'), bbox_inches='tight')
        plt.close(fig)
    
    # 5: Sigma bar chart (all hyps at tau=0.50)
    plot_hyps = sigma_df[(sigma_df['tau']=='050') & 
                         sigma_df['hyp_id'].isin(['Hyp-1','Hyp-2_opposing','Hyp-3_relaxed',
                         'Hyp-4_chaos','Hyp-4_baseline','Hyp-5_B4','Hyp-5_B9',
                         'Hyp-6_B3B6','Hyp-7','Hyp-8_full'])]
    if len(plot_hyps) > 0:
        fig, ax = plt.subplots(figsize=(14,6))
        x = range(len(plot_hyps))
        w = 0.35
        ax.bar([i-w/2 for i in x], plot_hyps['sigma_switching'], w, label='Switching', color='#FF7043')
        ax.bar([i+w/2 for i in x], plot_hyps['sigma_nonswitching'], w, label='Non-Switching', color='#42A5F5')
        ax.set_xticks(list(x))
        ax.set_xticklabels(plot_hyps['hyp_id'], rotation=45, ha='right')
        ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.7, label='Support threshold')
        # Annotate n
        for i, (_, r) in enumerate(plot_hyps.iterrows()):
            ax.text(i-w/2, r['sigma_switching']+0.01, f"n={int(r['n_switching'])}", ha='center', fontsize=6)
            ax.text(i+w/2, r['sigma_nonswitching']+0.01, f"n={int(r['n_nonswitching'])}", ha='center', fontsize=6)
        ax.set_ylabel('sigma'); ax.set_title(f'Sigma Comparison [{modality}] tau=0.50')
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, '_sigma_barchart.png'), bbox_inches='tight')
        plt.close(fig)
    
    # 6: Sensitivity curves
    hyps_to_plot = ['Hyp-1','Hyp-7','Hyp-5_B4','Hyp-5_B9']
    fig, axes = plt.subplots(2, 2, figsize=(14,10))
    tau_vals = [0.25, 0.50, 0.75, 1.00]
    for idx, hyp in enumerate(hyps_to_plot):
        ax = axes[idx//2][idx%2]
        h = sigma_df[sigma_df['hyp_id'] == hyp].copy()
        h['tau_f'] = h['tau'].astype(int) / 100
        ax.plot(h['tau_f'], h['sigma_switching'], 'o-', label='Switching')
        ax.plot(h['tau_f'], h['sigma_nonswitching'], 's--', label='Non-Switching')
        # Mark small n
        for _, r in h.iterrows():
            if r['n_switching'] < 10: ax.scatter(r['tau_f'], r['sigma_switching'], marker='x', c='red', s=100)
            if r['n_nonswitching'] < 10: ax.scatter(r['tau_f'], r['sigma_nonswitching'], marker='x', c='red', s=100)
        ax.set_title(hyp); ax.set_xlabel('tau'); ax.set_ylabel('sigma'); ax.legend()
    fig.suptitle(f'Sigma Sensitivity [{modality}]'); fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, '_sigma_sensitivity_curves.png'), bbox_inches='tight')
    plt.close(fig)
    
    # 7: Drift boxplot valence
    fig, ax = plt.subplots(figsize=(10,6))
    sub = df[df['clip1_E_quad'].isin(quads) & df['switching'].notna()]
    if len(sub) > 0:
        sns.boxplot(data=sub, x='clip1_E_quad', y='delta_I1_val', hue='switching', palette="Set2", ax=ax,
                    order=quads)
    ax.set_title(f'Self-Reported Valence Shift [{modality}]'); ax.set_ylabel('Delta I1 Valence')
    fig.savefig(os.path.join(plot_dir, '_drift_boxplot_valence.png'), bbox_inches='tight')
    plt.close(fig)
    
    # 8: Drift boxplot arousal
    fig, ax = plt.subplots(figsize=(10,6))
    if len(sub) > 0:
        sns.boxplot(data=sub, x='clip1_E_quad', y='delta_I1_aro', hue='switching', palette="Set2", ax=ax,
                    order=quads)
    ax.set_title(f'Self-Reported Arousal Shift [{modality}]'); ax.set_ylabel('Delta I1 Arousal')
    fig.savefig(os.path.join(plot_dir, '_drift_boxplot_arousal.png'), bbox_inches='tight')
    plt.close(fig)
    
    # 9: Combined bucket frequency
    fig, ax = plt.subplots(figsize=(12,6))
    valid = df[df['switching'].notna()]
    top10 = valid[comb].value_counts().head(10).index.tolist()
    sub = valid[valid[comb].isin(top10)]
    if len(sub) > 0:
        sns.countplot(data=sub, x=comb, hue='switching', order=top10, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title(f'Top 10 Combined Bucket Frequencies [{modality}] tau=0.50')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, '_combined_bucket_frequency.png'), bbox_inches='tight')
    plt.close(fig)
    
    # 10: I1 vs I2 scatter
    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot(data=df, x='clip2_I1_val', y='clip2_I2_val', hue=b_i1, alpha=0.5, ax=ax)
    ax.plot([1,5],[1,5],'k--',alpha=0.4)
    ax.set_title(f'I1 vs I2 Valence [{modality}]')
    ax.set_xlabel('Survey Valence (I1)'); ax.set_ylabel('Fitbit Valence (I2)')
    fig.savefig(os.path.join(plot_dir, '_I1_I2_scatter.png'), bbox_inches='tight')
    plt.close(fig)
    
    # 11: I1-I2 distance boxplot (full + arousal-only)
    df['i1_i2_dist'] = get_dist(df['clip2_I1_val'], df['clip2_I1_aro'],
                                 df['clip2_I2_val'], df['clip2_I2_aro'])
    df['i1_i2_aro_dist'] = (df['clip2_I1_aro'] - df['clip2_I2_aro']).abs()
    valid = df[df['switching'].notna()]
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    sns.boxplot(data=valid, x='switching', y='i1_i2_dist', hue='switching', ax=axes[0])
    axes[0].set_title('Full Euclidean Distance')
    sns.boxplot(data=valid, x='switching', y='i1_i2_aro_dist', hue='switching', ax=axes[1])
    axes[1].set_title('Arousal-Only Distance')
    fig.suptitle(f'I1-I2 Distance [{modality}]'); fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, '_I1_I2_distance_boxplot.png'), bbox_inches='tight')
    plt.close(fig)
    
    # 12: Maintenance vs Drift proof
    valid = df[df['switching'].notna()]
    b1_q1 = valid[(valid[b_i1] == 'B1') & (valid['clip2_E_quad'] == 'Q1')]
    drift = b1_q1[b1_q1['clip1_E_quad'] != 'Q1']
    maint = b1_q1[b1_q1['clip1_E_quad'] == 'Q1']
    
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    
    # Panel 1: delta_I1_val
    parts = []
    if len(drift) > 0:
        parts.append(pd.DataFrame({'Group': 'Drift', 'value': drift['delta_I1_val'].values}))
    if len(maint) > 0:
        parts.append(pd.DataFrame({'Group': 'Maintenance', 'value': maint['delta_I1_val'].values}))
    if parts:
        melted = pd.concat(parts, ignore_index=True)
        sns.boxplot(data=melted, x='Group', y='value', palette=["#FF9999","#99FF99"], ax=axes[0])
    axes[0].set_title('Delta I1 Valence')
    axes[0].annotate(f'n_drift={len(drift)}, n_maint={len(maint)}', xy=(0.5, 0.95),
                    xycoords='axes fraction', ha='center', fontsize=9)
    
    # Panel 2: delta_I2_val
    parts2 = []
    if len(drift) > 0:
        parts2.append(pd.DataFrame({'Group': 'Drift', 'value': drift['delta_I2_val'].values}))
    if len(maint) > 0:
        parts2.append(pd.DataFrame({'Group': 'Maintenance', 'value': maint['delta_I2_val'].values}))
    if parts2:
        melted2 = pd.concat(parts2, ignore_index=True)
        sns.boxplot(data=melted2, x='Group', y='value', palette=["#FF9999","#99FF99"], ax=axes[1])
    axes[1].set_title('Delta I2 Valence')
    
    fig.suptitle(f'Maintenance vs Drift Proof [{modality}]')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, '_maintenance_vs_drift_proof.png'), bbox_inches='tight')
    plt.close(fig)
    
    print(f"  12 plots saved to {plot_dir}")

if __name__ == "__main__":
    for mod in MODALITIES:
        print(f"\nGenerating plots for {mod}...")
        gen_plots(mod)
    print("\nAll 36 plots generated.")
