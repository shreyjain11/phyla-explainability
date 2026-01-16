"""
PHYLA Phase 2: Occlusion-Based Attribution
Measure how masking each MSA column affects CLS embedding distances
"""
import sys
import os
sys.path.insert(0, '/home/shrey/work/Phyla')
sys.path.insert(0, '/home/shrey/work')
os.chdir('/home/shrey/work/Phyla/phyla')

from phyla.model.model import BiMambaWrapper

def forward_with_residual(self, hidden_states, inference_params=None, cpu=None):
    out = self.mamba_fwd(hidden_states, inference_params=inference_params)
    if self.bidirectional:
        out_rev = self.mamba_rev(
            hidden_states.flip(dims=(1,)),
            inference_params=inference_params
        ).flip(dims=(1,))
        if self.bidirectional_strategy == "add":
            out = out + out_rev
        elif self.bidirectional_strategy == "ew_multiply":
            out = out * out_rev
    return hidden_states + out

BiMambaWrapper.forward = forward_with_residual

import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple
import json
from datetime import datetime
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
import wandb
import time

from phyla.model.model import Phyla, Config


class OcclusionAttributionAnalyzer:
    """
    Occlusion-based attribution: mask each column and measure 
    change in CLS distance correlation with ground truth.
    
    High attribution = masking this column hurts phylogenetic signal
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        config = Config()
        self.model = Phyla(config, device=device)
        self.model.load()
        self.model.eval()
    
    def get_cls_embeddings(self, sequences: List[str], names: List[str]) -> np.ndarray:
        """Extract CLS embeddings for sequences"""
        input_ids, cls_mask, seq_mask, _ = self.model.encode(sequences, names)
        input_ids = input_ids.to(self.device)
        cls_mask = cls_mask.to(self.device)
        seq_mask = seq_mask.to(self.device)
        
        final_output = None
        def hook(module, input, output):
            nonlocal final_output
            final_output = output[0].detach().cpu() if isinstance(output, tuple) else output.detach().cpu()
        
        handle = self.model.modul[-1].register_forward_hook(hook)
        
        with torch.no_grad():
            x = self.model.modul[0](input_ids, logits=False, position_ids=None,
                sequence_mask=seq_mask, cls_token_mask=cls_mask)
            for module in self.model.modul[1:]:
                dev = next(module.parameters()).device
                x = module(x.to(dev), hidden_states_given=True, logits=False,
                    position_ids=None, sequence_mask=seq_mask.to(dev),
                    cls_token_mask=cls_mask.to(dev))
        
        handle.remove()
        return final_output[0].numpy()
    
    def compute_gt_distances(self, sequences: List[str]) -> np.ndarray:
        """Compute 1 - sequence identity"""
        n = len(sequences)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                s1, s2 = sequences[i], sequences[j]
                matches = sum(1 for a, b in zip(s1, s2) if a == b and a != '-' and b != '-')
                aligned = sum(1 for a, b in zip(s1, s2) if a != '-' and b != '-')
                identity = matches / aligned if aligned > 0 else 0
                dist[i, j] = dist[j, i] = 1 - identity
        return dist
    
    def mask_column(self, sequences: List[str], col: int, mask_char: str = 'X') -> List[str]:
        """Replace column with mask character"""
        masked = []
        for seq in sequences:
            seq_list = list(seq)
            seq_list[col] = mask_char
            masked.append(''.join(seq_list))
        return masked
    
    def compute_column_attribution(self, sequences: List[str], names: List[str],
                                    alignment_id: str, sample_cols: int = 50) -> dict:
        """
        Compute attribution by measuring correlation drop when masking each column.
        
        Attribution = baseline_r - masked_r (how much masking hurts)
        """
        num_seq = len(sequences)
        msa_len = len(sequences[0])
        
        # Ground truth distances
        gt_dist = self.compute_gt_distances(sequences)
        gt_flat = squareform(gt_dist)
        
        # Baseline correlation
        cls_emb = self.get_cls_embeddings(sequences, names)
        cls_dist = pdist(cls_emb, metric='euclidean')
        baseline_r, _ = spearmanr(gt_flat, cls_dist)
        
        # Sample columns if MSA is long
        if msa_len > sample_cols:
            # Sample uniformly + ensure we get variable positions
            col_indices = sorted(np.random.choice(msa_len, sample_cols, replace=False))
        else:
            col_indices = list(range(msa_len))
        
        # Compute attribution for each sampled column
        attributions = np.zeros(msa_len)
        
        for col in col_indices:
            # Mask this column
            masked_seqs = self.mask_column(sequences, col)
            
            # Get CLS embeddings for masked version
            cls_emb_masked = self.get_cls_embeddings(masked_seqs, names)
            cls_dist_masked = pdist(cls_emb_masked, metric='euclidean')
            
            # Correlation with masked column
            if cls_dist_masked.std() > 1e-10:
                masked_r, _ = spearmanr(gt_flat, cls_dist_masked)
            else:
                masked_r = 0
            
            # Attribution = how much masking hurts correlation
            attributions[col] = baseline_r - masked_r
        
        # For non-sampled columns, interpolate or set to mean
        if msa_len > sample_cols:
            sampled_attr = attributions[col_indices]
            mean_attr = sampled_attr.mean()
            for col in range(msa_len):
                if col not in col_indices:
                    attributions[col] = mean_attr
        
        # Normalize to [0, 1]
        attr_min = attributions.min()
        attr_max = attributions.max()
        if attr_max > attr_min:
            attributions_norm = (attributions - attr_min) / (attr_max - attr_min)
        else:
            attributions_norm = np.zeros_like(attributions)
        
        return {
            'alignment_id': alignment_id,
            'n_sequences': num_seq,
            'msa_length': msa_len,
            'baseline_r': float(baseline_r),
            'column_attribution': attributions_norm.tolist(),
            'column_attribution_raw': attributions.tolist(),
            'top_10_columns': np.argsort(attributions)[-10:][::-1].tolist(),
            'n_sampled': len(col_indices),
        }
    
    def analyze_biological_features(self, sequences: List[str], 
                                     attribution: np.ndarray) -> dict:
        """Correlate attribution with biological features"""
        msa_len = len(sequences[0])
        num_seq = len(sequences)
        
        conservation = []
        gap_fraction = []
        variability = []
        
        for col in range(msa_len):
            residues = [seq[col] for seq in sequences]
            
            gaps = sum(1 for r in residues if r == '-')
            gap_fraction.append(gaps / num_seq)
            
            non_gap = [r for r in residues if r != '-']
            if non_gap:
                unique = len(set(non_gap))
                variability.append(unique / min(len(non_gap), 20))  # Normalize by max AA types
            else:
                variability.append(0)
            
            conservation.append(1 - variability[-1])
        
        conservation = np.array(conservation)
        gap_fraction = np.array(gap_fraction)
        variability = np.array(variability)
        
        # Only correlate on non-gap-dominated columns
        valid_cols = gap_fraction < 0.8
        
        if valid_cols.sum() > 10:
            r_var, p_var = spearmanr(attribution[valid_cols], variability[valid_cols])
            r_cons, p_cons = spearmanr(attribution[valid_cols], conservation[valid_cols])
        else:
            r_var, p_var = 0, 1
            r_cons, p_cons = 0, 1
        
        r_gaps, p_gaps = spearmanr(attribution, gap_fraction)
        
        return {
            'r_attribution_variability': float(r_var),
            'p_variability': float(p_var),
            'r_attribution_conservation': float(r_cons),
            'p_conservation': float(p_cons),
            'r_attribution_gaps': float(r_gaps),
            'p_gaps': float(p_gaps),
            'mean_variability': float(variability.mean()),
            'mean_gap_fraction': float(gap_fraction.mean()),
        }


def load_fasta(path: Path) -> Tuple[List[str], List[str]]:
    names, seqs = [], []
    name, seq = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name:
                    names.append(name)
                    seqs.append(''.join(seq))
                name = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
    if name:
        names.append(name)
        seqs.append(''.join(seq))
    return names, seqs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/shrey/work/Cleaned_OpenProtein_Set/Cleaned_Open_Protein_Set')
    parser.add_argument('--output_dir', default='/home/shrey/work/explainability/results/phase2')
    parser.add_argument('--n_alignments', type=int, default=200)
    parser.add_argument('--min_seqs', type=int, default=6)
    parser.add_argument('--max_seqs', type=int, default=15)
    parser.add_argument('--sample_cols', type=int, default=30)
    parser.add_argument('--wandb_project', default='phyla-explainability')
    parser.add_argument('--wandb_run_name', default=None)
    args = parser.parse_args()
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    run_name = args.wandb_run_name or f"phase2_occlusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project=args.wandb_project, name=run_name,
               config=vars(args), tags=['phase2', 'occlusion', 'attribution'])
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("[INFO] Loading model...", flush=True)
    analyzer = OcclusionAttributionAnalyzer(device='cuda')
    print("[INFO] Model loaded", flush=True)
    
    data_dir = Path(args.data_dir)
    fastas = sorted(list(data_dir.glob("*_alignment.fasta")))
    
    all_results = []
    bio_corrs = {'variability': [], 'conservation': [], 'gaps': []}
    
    processed = 0
    start_time = time.time()
    
    print(f"\n{'='*70}", flush=True)
    print(f"PHASE 2: Occlusion Attribution ({args.n_alignments} alignments)", flush=True)
    print(f"{'='*70}\n", flush=True)
    
    for fasta in fastas:
        if processed >= args.n_alignments:
            break
        
        aid = fasta.stem.replace("_alignment", "")
        
        try:
            names, seqs = load_fasta(fasta)
            
            if len(seqs) < args.min_seqs:
                continue
            
            if len(seqs) > args.max_seqs:
                idxs = sorted(np.random.choice(len(seqs), args.max_seqs, replace=False))
                names = [names[i] for i in idxs]
                seqs = [seqs[i] for i in idxs]
            
            clean = [''.join(c for c in s if c.isalpha() or c == '-') for s in seqs]
            
            # Skip very long alignments for speed
            if len(clean[0]) > 500:
                continue
            
            result = analyzer.compute_column_attribution(
                clean, names, aid, sample_cols=args.sample_cols)
            
            attribution = np.array(result['column_attribution'])
            bio_features = analyzer.analyze_biological_features(clean, attribution)
            result.update(bio_features)
            
            all_results.append(result)
            
            bio_corrs['variability'].append(bio_features['r_attribution_variability'])
            bio_corrs['conservation'].append(bio_features['r_attribution_conservation'])
            bio_corrs['gaps'].append(bio_features['r_attribution_gaps'])
            
            wandb.log({
                'idx': processed,
                'baseline_r': result['baseline_r'],
                'r_attr_variability': bio_features['r_attribution_variability'],
                'r_attr_conservation': bio_features['r_attribution_conservation'],
                'r_attr_gaps': bio_features['r_attribution_gaps'],
            })
            
            processed += 1
            
            if processed % 10 == 0 or processed <= 5:
                elapsed = time.time() - start_time
                rate = processed / elapsed * 60
                mean_var = np.mean(bio_corrs['variability'])
                print(f"[{processed:3d}/{args.n_alignments}] {aid:<18} "
                      f"base_r={result['baseline_r']:.3f} "
                      f"r(attr,var)={bio_features['r_attribution_variability']:.3f} "
                      f"| mean={mean_var:.3f} | {rate:.0f}/min", flush=True)
        
        except Exception as e:
            print(f"[ERROR] {aid}: {str(e)[:50]}", flush=True)
            continue
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}", flush=True)
    print(f"PHASE 2 COMPLETE ({processed} alignments in {elapsed/60:.1f}min)", flush=True)
    print(f"{'='*70}", flush=True)
    
    mean_var = np.mean(bio_corrs['variability'])
    std_var = np.std(bio_corrs['variability'])
    mean_cons = np.mean(bio_corrs['conservation'])
    mean_gaps = np.mean(bio_corrs['gaps'])
    
    print(f"\n📊 ATTRIBUTION vs BIOLOGICAL FEATURES:")
    print(f"   Attribution ↔ Variability:   r = {mean_var:.4f} ± {std_var:.4f}")
    print(f"   Attribution ↔ Conservation:  r = {mean_cons:.4f}")
    print(f"   Attribution ↔ Gap fraction:  r = {mean_gaps:.4f}")
    
    print(f"\n🎯 BIOLOGICAL INTERPRETATION:")
    if mean_var > 0.2:
        print(f"   ✓ Model focuses on VARIABLE positions (phylogenetically informative)")
        print(f"     Columns with more amino acid diversity contribute more to phylogenetic signal")
    elif mean_var < -0.2:
        print(f"   ✗ Model focuses on CONSERVED positions (unexpected)")
    else:
        print(f"   ~ Model shows weak preference (attribution spread across columns)")
    
    if mean_gaps < -0.1:
        print(f"   ✓ Model down-weights GAPPED regions")
    
    print(f"\n{'='*70}", flush=True)
    
    wandb.log({
        'final/n_processed': processed,
        'final/mean_r_variability': mean_var,
        'final/std_r_variability': std_var,
        'final/mean_r_conservation': mean_cons,
        'final/mean_r_gaps': mean_gaps,
        'final/time_minutes': elapsed/60,
    })
    
    out_file = Path(args.output_dir) / f"phase2_occlusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        json.dump({'config': vars(args), 'results': all_results,
                   'summary': {'mean_r_variability': mean_var, 'std_r_variability': std_var,
                              'mean_r_conservation': mean_cons, 'mean_r_gaps': mean_gaps}}, f)
    print(f"\n✓ Saved: {out_file}", flush=True)
    
    wandb.finish()


if __name__ == "__main__":
    main()
