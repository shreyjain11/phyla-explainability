"""
Generate High-Quality Publication Figures from Full 3,321 Alignment Dataset

Creates professional figures with biological insights and detailed analysis.
Output: outputs/figures_full_3321/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import torch
from google.cloud import storage
from io import BytesIO
from scipy.stats import gaussian_kde

from integrations.correct_phyla_adapter import CorrectPhylaMambaLRPAnalyzer
from Phyla.phyla.model.model import Phyla


# High-quality publication styling
def apply_publication_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 13,
        'axes.labelsize': 15,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.linewidth': 1.5,
        'grid.linewidth': 1.0,
        'lines.linewidth': 2.0,
        'patch.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.15,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def load_validation_results(filepath):
    """Parse full dataset validation results"""
    print(f"Loading results from: {filepath}")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    stats = {}
    for line in lines[:15]:
        if 'Mean r:' in line:
            stats['mean_r'] = float(line.split(':')[1].strip())
        elif 'Median r:' in line:
            stats['median_r'] = float(line.split(':')[1].strip())
        elif 'Std r:' in line:
            stats['std_r'] = float(line.split(':')[1].strip())
        elif 'Min r:' in line:
            stats['min_r'] = float(line.split(':')[1].strip())
        elif 'Max r:' in line:
            stats['max_r'] = float(line.split(':')[1].strip())
    
    results = []
    start_idx = None
    
    for i, line in enumerate(lines):
        if 'ALL RESULTS' in line or 'Alignment' in line and 'Status' in line:
            start_idx = i + 2
            break
    
    if start_idx:
        for line in lines[start_idx:]:
            line = line.strip()
            if not line or line.startswith('=') or line.startswith('-'):
                continue
            
            parts = line.split()
            if len(parts) >= 3:
                try:
                    alignment = parts[0]
                    n = int(parts[1])
                    r = float(parts[2])
                    
                    if r > 0.90:
                        status = 'excellent'
                    elif r > 0.70:
                        status = 'strong'
                    elif r > 0.50:
                        status = 'moderate'
                    else:
                        status = 'weak'
                    
                    results.append({
                        'alignment': alignment,
                        'n_sequences': n,
                        'r': r,
                        'status': status
                    })
                except (ValueError, IndexError):
                    continue
    
    print(f"Loaded {len(results)} alignments")
    print(f"Statistics: mean={stats.get('mean_r', 'N/A')}, median={stats.get('median_r', 'N/A')}")
    
    return results, stats


def load_fasta_from_gcs(bucket_name, blob_name):
    """Load FASTA from GCS"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_text()
    
    sequences = []
    names = []
    current_seq = []
    current_name = None
    
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('>'):
            if current_name:
                sequences.append(''.join(current_seq))
                names.append(current_name)
            current_name = line[1:]
            current_seq = []
        elif line:
            current_seq.append(line)
    
    if current_name:
        sequences.append(''.join(current_seq))
        names.append(current_name)
    
    return sequences, names


def load_ground_truth_from_gcs(bucket_name, blob_name):
    """Load ground truth distance matrix"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_bytes()
    return np.load(BytesIO(content))


def plot_correlation_distribution_advanced(results, stats, output_path):
    """Advanced correlation distribution with KDE and biological insights"""
    
    r_values = np.array([res['r'] for res in results])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    n, bins, patches = ax1.hist(r_values, bins=50, edgecolor='black', alpha=0.7, 
                                 color='#3498db', density=True, label='Distribution')
    
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center > 0.90:
            patch.set_facecolor('#27ae60')  # Excellent
        elif bin_center > 0.70:
            patch.set_facecolor('#3498db')  # Strong
        elif bin_center > 0.50:
            patch.set_facecolor('#f39c12')  # Moderate
        else:
            patch.set_facecolor('#e74c3c')
    
    kde = gaussian_kde(r_values)
    x_kde = np.linspace(r_values.min(), r_values.max(), 200)
    ax1.plot(x_kde, kde(x_kde), 'k-', linewidth=3, label='Density (KDE)', alpha=0.8)
    
    ax1.axvline(0.90, color='#27ae60', linestyle='--', linewidth=2.5, alpha=0.8, label='Excellent (r>0.90)')
    ax1.axvline(0.70, color='#3498db', linestyle='--', linewidth=2.5, alpha=0.8, label='Strong (r>0.70)')
    ax1.axvline(0.50, color='#f39c12', linestyle='--', linewidth=2.5, alpha=0.8, label='Moderate (r>0.50)')
    
    mean_r = stats['mean_r']
    median_r = stats['median_r']
    ax1.axvline(mean_r, color='darkred', linestyle='-', linewidth=3, label=f'Mean (r={mean_r:.3f})')
    ax1.axvline(median_r, color='darkblue', linestyle=':', linewidth=3, label=f'Median (r={median_r:.3f})')
    
    ax1.set_xlabel('Pearson Correlation (r)', fontweight='bold', fontsize=15)
    ax1.set_ylabel('Density', fontweight='bold', fontsize=15)
    ax1.set_title(f'Phylogenetic Distance Prediction Performance\n(n={len(results)} protein family alignments)', 
                  fontweight='bold', fontsize=18)
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    bp = ax2.boxplot([r_values], vert=False, widths=0.5, patch_artist=True,
                     boxprops=dict(facecolor='#3498db', alpha=0.6),
                     medianprops=dict(color='darkred', linewidth=3),
                     whiskerprops=dict(linewidth=2),
                     capprops=dict(linewidth=2))
    
    ax2.set_xlabel('Pearson Correlation (r)', fontweight='bold', fontsize=15)
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3, axis='x')
    
    q1, q2, q3 = np.percentile(r_values, [25, 50, 75])
    ax2.text(q1, 0.3, f'Q1={q1:.2f}', ha='center', fontsize=11, fontweight='bold')
    ax2.text(q2, 0.3, f'Median={q2:.2f}', ha='center', fontsize=11, fontweight='bold')
    ax2.text(q3, 0.3, f'Q3={q3:.2f}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure 1 saved: {output_path}")


def plot_performance_breakdown_detailed(results, stats, output_path):
    """Detailed performance breakdown with biological context"""
    
    excellent = [r for r in results if r['r'] > 0.90]
    strong = [r for r in results if 0.70 < r['r'] <= 0.90]
    moderate = [r for r in results if 0.50 < r['r'] <= 0.70]
    weak = [r for r in results if r['r'] <= 0.50]
    
    total = len(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Bar chart with counts
    categories = ['Excellent\n(r>0.90)', 'Strong\n(0.70<r≤0.90)', 
                  'Moderate\n(0.50<r≤0.70)', 'Weak\n(r≤0.50)']
    counts = [len(excellent), len(strong), len(moderate), len(weak)]
    percentages = [c/total*100 for c in counts]
    colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c']
    
    bars = ax1.bar(categories, counts, color=colors, edgecolor='black', 
                   linewidth=2, alpha=0.85, width=0.7)
    
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=13)
    
    ax1.set_ylabel('Number of Protein Families', fontweight='bold', fontsize=15)
    ax1.set_title(f'Performance Distribution\n(Total: {total:,} alignments)', 
                  fontweight='bold', fontsize=16)
    ax1.set_ylim(0, max(counts) * 1.18)
    ax1.grid(axis='y', alpha=0.3, linewidth=1.2)
    
    ax2.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'black', 'linewidth': 2, 'alpha': 0.85})
    
    ax2.set_title('Performance Distribution', fontweight='bold', fontsize=16)
    
    interpretation = (
        f"Biological Insights:\n\n"
        f"• Model achieves excellent correlation (r>0.90) on {len(excellent):,} families ({percentages[0]:.1f}%)\n"
        f"  → These families likely have clear phylogenetic signal\n\n"
        f"• Strong performance (r>0.70) on {len(strong):,} additional families\n"
        f"  → Total {len(excellent)+len(strong):,} families ({percentages[0]+percentages[1]:.1f}%) show reliable predictions\n\n"
        f"• Challenges with {len(weak):,} families ({percentages[3]:.1f}%)\n"
        f"  → May indicate high sequence similarity, gaps, or limited diversity"
    )
    
    fig.text(0.5, -0.05, interpretation, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
             wrap=True)
    
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure 2 saved: {output_path}")


def plot_correlation_vs_size_detailed(results, output_path):
    """Detailed scatter with size-based analysis"""
    
    n_seqs = np.array([res['n_sequences'] for res in results])
    r_values = np.array([res['r'] for res in results])
    
    colors = []
    sizes = []
    for r in r_values:
        if r > 0.90:
            colors.append('#27ae60')
            sizes.append(120)
        elif r > 0.70:
            colors.append('#3498db')
            sizes.append(80)
        elif r > 0.50:
            colors.append('#f39c12')
            sizes.append(60)
        else:
            colors.append('#e74c3c')
            sizes.append(50)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    scatter = ax.scatter(n_seqs, r_values, c=colors, s=sizes, alpha=0.6, 
                        edgecolors='black', linewidth=1.2)
    
    z = np.polyfit(n_seqs, r_values, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(n_seqs.min(), n_seqs.max(), 100)
    ax.plot(x_trend, p(x_trend), "k--", alpha=0.6, linewidth=3, 
            label=f'Linear Trend: r = {z[0]:.4f}×n + {z[1]:.3f}')
    
    window_size = 50
    sorted_idx = np.argsort(n_seqs)
    n_sorted = n_seqs[sorted_idx]
    r_sorted = r_values[sorted_idx]
    
    moving_avg_r = []
    moving_avg_n = []
    for i in range(0, len(r_sorted) - window_size, 10):
        moving_avg_r.append(np.mean(r_sorted[i:i+window_size]))
        moving_avg_n.append(np.mean(n_sorted[i:i+window_size]))
    
    ax.plot(moving_avg_n, moving_avg_r, 'purple', linewidth=4, alpha=0.7,
            label=f'Moving Average (window={window_size})')
    
    ax.axhline(0.90, color='#27ae60', linestyle=':', alpha=0.5, linewidth=2)
    ax.axhline(0.70, color='#3498db', linestyle=':', alpha=0.5, linewidth=2)
    ax.axhline(0.50, color='#f39c12', linestyle=':', alpha=0.5, linewidth=2)
    
    ax.set_xlabel('Number of Sequences in Alignment', fontweight='bold', fontsize=15)
    ax.set_ylabel('Pearson Correlation (r)', fontweight='bold', fontsize=15)
    ax.set_title('Prediction Quality vs Alignment Size\n(Biological Complexity Analysis)', 
                 fontweight='bold', fontsize=18)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, linewidth=1.2)
    
    small_mask = n_seqs <= 5
    medium_mask = (n_seqs > 5) & (n_seqs <= 15)
    large_mask = n_seqs > 15
    
    stats_text = (
        f"Size-Based Performance:\n"
        f"Small (≤5):   mean r={np.mean(r_values[small_mask]):.3f} (n={sum(small_mask):,})\n"
        f"Medium (6-15): mean r={np.mean(r_values[medium_mask]):.3f} (n={sum(medium_mask):,})\n"
        f"Large (>15):   mean r={np.mean(r_values[large_mask]):.3f} (n={sum(large_mask):,})"
    )
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure 3 saved: {output_path}")


def plot_top_alignments_comparison(results, phyla_model, analyzer, output_path):
    """Compare model vs ground truth for top 3 performers"""
    
    top_3 = sorted(results, key=lambda x: x['r'], reverse=True)[:3]
    
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)
    
    for idx, result in enumerate(top_3):
        alignment_name = result['alignment']
        r_val = result['r']
        
        print(f"Loading {alignment_name} (r={r_val:.3f})...")
        
        try:
            # Load data
            fasta_path = f"Cleaned_Open_Protein_Set/{alignment_name}_alignment.fasta"
            sequences, names = load_fasta_from_gcs('phyla', fasta_path)
            
            gt_path = f"Cleaned_Open_Protein_Set/{alignment_name}_alignment_dist_matrix.npy"
            gt = load_ground_truth_from_gcs('phyla', gt_path)
            
            # Limit to 10
            n = min(10, len(sequences))
            sequences = sequences[:n]
            names = names[:n]
            gt = gt[:n, :n]
            
            # Analyze
            res = analyzer.analyze_sequences(sequences, names, gt, verbose=False, skip_quality_check=True)
            
            if res:
                model_dist = res.pairwise_distances.numpy()
                
                # Plot
                ax = fig.add_subplot(gs[idx])
                
                # Get upper triangle
                triu_idx = np.triu_indices(n, k=1)
                model_vec = model_dist[triu_idx]
                gt_vec = gt[triu_idx]
                
                ax.scatter(gt_vec, model_vec, alpha=0.7, s=100, color='#3498db', 
                          edgecolors='black', linewidth=1.5)
                
                # Perfect prediction line
                max_val = max(gt_vec.max(), model_vec.max())
                min_val = min(gt_vec.min(), model_vec.min())
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2.5, 
                       label='Perfect Prediction', alpha=0.7)
                
                # Regression line
                z = np.polyfit(gt_vec, model_vec, 1)
                p = np.poly1d(z)
                ax.plot(gt_vec, p(gt_vec), 'r-', linewidth=3, alpha=0.8,
                       label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
                
                ax.set_xlabel('Ground Truth Distance', fontweight='bold', fontsize=13)
                ax.set_ylabel('Model Predicted Distance', fontweight='bold', fontsize=13)
                ax.set_title(f'{alignment_name}\n(r={r_val:.3f}, n={n})', fontweight='bold', fontsize=14)
                ax.legend(frameon=True, fancybox=True, fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')
        
        except Exception as e:
            print(f"  Error loading {alignment_name}: {str(e)[:50]}")
            continue
    
    fig.suptitle('Model vs Ground Truth: Top 3 Performing Alignments', 
                 fontweight='bold', fontsize=20, y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure 4 saved: {output_path}")


def plot_attention_heatmap_detailed(results, phyla_model, analyzer, output_path):
    """High-quality attention heatmap with sequence annotations"""
    
    best = max(results, key=lambda x: x['r'])
    alignment_name = best['alignment']
    
    print(f"Generating attention heatmap for {alignment_name} (r={best['r']:.3f})...")
    
    # Load data
    fasta_path = f"Cleaned_Open_Protein_Set/{alignment_name}_alignment.fasta"
    sequences, names = load_fasta_from_gcs('phyla', fasta_path)
    
    gt_path = f"Cleaned_Open_Protein_Set/{alignment_name}_alignment_dist_matrix.npy"
    gt = load_ground_truth_from_gcs('phyla', gt_path)
    
    # Limit to 10
    n = min(10, len(sequences))
    sequences = sequences[:n]
    names = names[:n]
    gt = gt[:n, :n]
    
    # Analyze
    res = analyzer.analyze_sequences(sequences, names, gt, verbose=False, skip_quality_check=True)
    
    if res:
        relevances = res.relevances.numpy()
        
        fig, ax = plt.subplots(figsize=(20, 8))
        
        relevances_norm = relevances / (relevances.max() + 1e-10)
        
        im = ax.imshow(relevances_norm, cmap='YlOrRd', aspect='auto', 
                      interpolation='bilinear', vmin=0, vmax=1)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('Normalized Attention Weight\n(Position Importance)', 
                      fontweight='bold', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        
        # Labels
        ax.set_xlabel('Position in Alignment', fontweight='bold', fontsize=15)
        ax.set_ylabel('Sequence', fontweight='bold', fontsize=15)
        ax.set_title(f'Position-Level Attention Relevances: {alignment_name}\n' + 
                    f'(r={best["r"]:.3f}, n={n} sequences, phylogenetically important positions highlighted)',
                    fontweight='bold', fontsize=18)
        
        # Sequence labels
        if len(names) <= 15:
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([name[:40] + '...' if len(name) > 40 else name for name in names], 
                              fontsize=10)
        
        # Add grid lines between sequences
        for i in range(len(names)):
            ax.axhline(i - 0.5, color='white', linewidth=1.5, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Figure 5 saved: {output_path}")


def plot_distance_matrices_detailed(results, phyla_model, analyzer, output_path):
    """Side-by-side distance matrices with difference heatmap"""
    
    best = max(results, key=lambda x: x['r'])
    alignment_name = best['alignment']
    
    print(f"Generating distance matrix comparison for {alignment_name}...")
    
    # Load data
    fasta_path = f"Cleaned_Open_Protein_Set/{alignment_name}_alignment.fasta"
    sequences, names = load_fasta_from_gcs('phyla', fasta_path)
    
    gt_path = f"Cleaned_Open_Protein_Set/{alignment_name}_alignment_dist_matrix.npy"
    gt = load_ground_truth_from_gcs('phyla', gt_path)
    
    # Limit to 10
    n = min(10, len(sequences))
    sequences = sequences[:n]
    names = names[:n]
    gt = gt[:n, :n]
    
    # Analyze
    res = analyzer.analyze_sequences(sequences, names, gt, verbose=False, skip_quality_check=True)
    
    if res:
        model_dist = res.pairwise_distances.numpy()
        
        fig = plt.figure(figsize=(20, 6))
        gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.4)
        
        vmin = 0
        vmax = max(model_dist.max(), gt.max())
        
        ax1 = fig.add_subplot(gs[0])
        im1 = ax1.imshow(model_dist, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        ax1.set_title('Model Predictions', fontweight='bold', fontsize=16)
        ax1.set_xlabel('Sequence Index', fontweight='bold', fontsize=13)
        ax1.set_ylabel('Sequence Index', fontweight='bold', fontsize=13)
        
        ax2 = fig.add_subplot(gs[1])
        im2 = ax2.imshow(gt, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        ax2.set_title('Ground Truth', fontweight='bold', fontsize=16)
        ax2.set_xlabel('Sequence Index', fontweight='bold', fontsize=13)
        ax2.set_ylabel('Sequence Index', fontweight='bold', fontsize=13)
        
        ax3 = fig.add_subplot(gs[2])
        diff = model_dist - gt
        max_diff = max(abs(diff.min()), abs(diff.max()))
        im3 = ax3.imshow(diff, cmap='RdBu_r', vmin=-max_diff, vmax=max_diff, aspect='auto')
        ax3.set_title('Prediction Error\n(Model - Ground Truth)', fontweight='bold', fontsize=16)
        ax3.set_xlabel('Sequence Index', fontweight='bold', fontsize=13)
        ax3.set_ylabel('Sequence Index', fontweight='bold', fontsize=13)
        
        cax = fig.add_subplot(gs[3])
        cbar = plt.colorbar(im2, cax=cax)
        cbar.set_label('Phylogenetic Distance', fontweight='bold', fontsize=14)
        
        fig.suptitle(f'Distance Matrix Analysis: {alignment_name} (r={best["r"]:.3f})',
                    fontweight='bold', fontsize=20)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Figure 6 saved: {output_path}")


def main():
    print("="*80)
    print("GENERATING HIGH-QUALITY FIGURES FROM FULL 3,321 ALIGNMENT DATASET")
    print("="*80)
    
    apply_publication_style()
    
    # Create output directory
    output_dir = "outputs/figures_full_3321"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}/")
    
    # Load full dataset results
    results_file = "outputs/full_dataset_validation_results.txt"
    
    if not os.path.exists(results_file):
        print(f"ERROR: Results file not found: {results_file}")
        return
    
    results, stats = load_validation_results(results_file)
    
    if not results:
        print("ERROR: No results loaded!")
        return
    
    print(f"\nDataset: {len(results)} alignments")
    print(f"Mean r: {stats['mean_r']:.3f}")
    print(f"Median r: {stats['median_r']:.3f}")
    
    # FIGURES 1-3: Aggregate statistics
    print("\n" + "="*80)
    print("Generating aggregate statistics figures...")
    print("="*80)
    
    plot_correlation_distribution_advanced(results, stats, 
                                          f"{output_dir}/figure1_correlation_distribution_3321.png")
    
    plot_performance_breakdown_detailed(results, stats,
                                       f"{output_dir}/figure2_performance_breakdown_3321.png")
    
    plot_correlation_vs_size_detailed(results,
                                     f"{output_dir}/figure3_correlation_vs_size_3321.png")
    
    # FIGURES 4-6: Explainability on best alignments
    print("\n" + "="*80)
    print("Generating explainability figures...")
    print("="*80)
    
    # Load Phyla model
    print("Loading Phyla model...")
    phyla_model = Phyla(name='phyla-alpha')
    phyla_model.load()
    phyla_model.eval()
    
    analyzer = CorrectPhylaMambaLRPAnalyzer(phyla_model, device='cuda')
    
    plot_top_alignments_comparison(results, phyla_model, analyzer,
                                   f"{output_dir}/figure4_top3_model_vs_ground_truth.png")
    
    plot_attention_heatmap_detailed(results, phyla_model, analyzer,
                                   f"{output_dir}/figure5_attention_relevance_heatmap.png")
    
    plot_distance_matrices_detailed(results, phyla_model, analyzer,
                                   f"{output_dir}/figure6_distance_matrix_comparison.png")
    
    print("\n" + "="*80)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("="*80)
    print(f"\nOutput directory: {output_dir}/")
    print("\nGenerated figures:")
    print("  1. figure1_correlation_distribution_3321.png - Distribution with KDE and quartiles")
    print("  2. figure2_performance_breakdown_3321.png - Bar chart with biological interpretation")
    print("  3. figure3_correlation_vs_size_3321.png - Scatter with trend analysis")
    print("  4. figure4_top3_model_vs_ground_truth.png - Top 3 alignments comparison")
    print("  5. figure5_attention_relevance_heatmap.png - Position importance visualization")
    print("  6. figure6_distance_matrix_comparison.png - Model vs GT with error heatmap")
    print("="*80)


if __name__ == "__main__":
    main()

