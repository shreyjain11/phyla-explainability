"""
Test Simple Improvements
Compare 3 proven methods on 50 diverse alignments
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from scipy.stats import pearsonr
from google.cloud import storage
from io import BytesIO
import random

from integrations.simple_phyla_adapter import SimplePhylaAnalyzer
from Phyla.phyla.model.model import Phyla


def load_fasta_from_gcs(bucket_name, blob_name):
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
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_bytes()
    return np.load(BytesIO(content))


def get_all_alignments_by_category():
    """Get alignments categorized by size"""
    client = storage.Client()
    bucket = client.bucket('phyla')
    blobs = list(bucket.list_blobs(prefix='Cleaned_Open_Protein_Set/'))
    
    npy_files = [b for b in blobs if b.name.endswith('.npy')]
    
    def estimate_n_sequences(size_bytes):
        n_squared = (size_bytes - 128) / 8
        if n_squared > 0:
            return int(np.sqrt(n_squared))
        return 0
    
    small, medium, large = [], [], []
    
    for blob in npy_files:
        fname = blob.name.split('/')[-1]
        if fname.endswith('_alignment_dist_matrix.npy'):
            alignment_id = fname.replace('_alignment_dist_matrix.npy', '')
            n = estimate_n_sequences(blob.size)
            
            if n <= 5:
                small.append(alignment_id)
            elif n <= 15:
                medium.append(alignment_id)
            else:
                large.append(alignment_id)
    
    return small, medium, large


def test_alignment(alignment_name, analyzer, methods, max_seqs=10):
    """Test one alignment with all methods"""
    
    try:
        # Load data
        fasta_path = f"Cleaned_Open_Protein_Set/{alignment_name}_alignment.fasta"
        sequences, names = load_fasta_from_gcs('phyla', fasta_path)
        
        gt_path = f"Cleaned_Open_Protein_Set/{alignment_name}_alignment_dist_matrix.npy"
        ground_truth = load_ground_truth_from_gcs('phyla', gt_path)
        
        # Limit sequences
        n = min(max_seqs, len(sequences))
        sequences = sequences[:n]
        names = names[:n]
        gt = ground_truth[:n, :n]
        
        results = {}
        
        for method in methods:
            result = analyzer.analyze_sequences(sequences, names, gt, method=method, verbose=False)
            results[method] = result.distance_correlation if result else None
        
        return results
        
    except Exception as e:
        return None


def main():
    print("="*80)
    print("TEST SIMPLE IMPROVEMENTS")
    print("="*80)
    print("\nMethods:")
    print("  1. l2_norm    - L2 normalize embeddings before distance")
    print("  2. cosine     - Cosine distance instead of Euclidean")
    print("  3. last2_avg  - Average last 2 layers + L2 norm")
    print("="*80)
    
    # Load model
    print("\nLoading Phyla model...")
    phyla_model = Phyla(name='phyla-alpha')
    phyla_model.load()
    phyla_model.eval()
    
    print("Initializing simple analyzer...")
    analyzer = SimplePhylaAnalyzer(phyla_model, device='cuda')
    
    # Get alignments
    print("\nSelecting 50 diverse alignments...")
    small, medium, large = get_all_alignments_by_category()
    
    random.seed(42)
    selected = (
        random.sample(small, min(15, len(small))) +
        random.sample(medium, min(20, len(medium))) +
        random.sample(large, min(15, len(large)))
    )
    
    print(f"Testing {len(selected)} alignments...")
    
    methods = ['l2_norm', 'cosine', 'last2_avg']
    method_scores = {m: [] for m in methods}
    
    for i, alignment_id in enumerate(selected, 1):
        results = test_alignment(alignment_id, analyzer, methods, max_seqs=10)
        
        if results is None:
            continue
        
        for method, r in results.items():
            if r is not None:
                method_scores[method].append(r)
        
        if i % 10 == 0:
            print(f"  Processed {i}/{len(selected)}...")
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\n{'Method':<15} {'Mean r':<10} {'Median r':<10} {'Max r':<10} {'Strong %':<10}")
    print("-"*80)
    
    best_mean = -999
    best_method = None
    
    for method in methods:
        scores = method_scores[method]
        if len(scores) > 0:
            mean_r = np.mean(scores)
            median_r = np.median(scores)
            max_r = np.max(scores)
            strong_pct = sum(1 for r in scores if r > 0.70) / len(scores) * 100
            
            if mean_r > best_mean:
                best_mean = mean_r
                best_method = method
            
            status = "✅" if mean_r >= 0.75 else "⚠️" if mean_r >= 0.70 else "❌"
            print(f"{method:<15} {mean_r:>9.3f} {median_r:>9.3f} {max_r:>9.3f} {strong_pct:>9.1f}  {status}")
    
    print("\n" + "="*80)
    print(f"WINNER: {best_method.upper()} (mean r={best_mean:.3f})")
    print("="*80)
    
    # Compare to baseline
    baseline = 0.666  # From comprehensive_validation_100plus
    improvement = best_mean - baseline
    
    print(f"\nBaseline (complex method): r={baseline:.3f}")
    print(f"Best simple method: r={best_mean:.3f}")
    print(f"Improvement: {improvement:+.3f} ({improvement/baseline*100:+.1f}%)")
    
    if best_mean >= 0.75:
        print("\n✅ TARGET ACHIEVED (r >= 0.75)!")
    else:
        print(f"\n⚠️ Need {0.75 - best_mean:.3f} more to reach r=0.75")


if __name__ == "__main__":
    main()

