"""
PHYLA Phase 1: CLS Embedding Phylogenetic Correlation Analysis
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
print("✓ Applied BiMambaWrapper residual fix", flush=True)

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json
from datetime import datetime
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
import wandb
import time

from phyla.model.model import Phyla, Config


class PhylaCLSAnalyzer:
    def __init__(self, device='cuda'):
        self.device = device
        config = Config()
        self.model = Phyla(config, device=device)
        self.model.load()
        self.model.eval()
        self.num_modules = len(self.model.modul)
    
    def compute_sequence_distances(self, sequences: List[str]) -> np.ndarray:
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
    
    def analyze_alignment(self, sequences: List[str], names: List[str], alignment_id: str) -> Dict:
        num_seq = len(sequences)
        msa_len = len(sequences[0])
        
        gt_dist = self.compute_sequence_distances(sequences)
        gt_flat = squareform(gt_dist)
        gt_std = gt_flat.std()
        
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
        
        cls_emb = final_output[0].numpy()
        cls_dist_euc = pdist(cls_emb, metric='euclidean')
        cls_dist_cos = pdist(cls_emb, metric='cosine')
        
        result = {
            'alignment_id': alignment_id,
            'n_sequences': num_seq,
            'msa_length': msa_len,
            'gt_distance_mean': float(gt_flat.mean()),
            'gt_distance_std': float(gt_std),
            'cls_norm_mean': float(np.linalg.norm(cls_emb, axis=1).mean()),
            'dist_euc_mean': float(cls_dist_euc.mean()),
            'dist_euc_std': float(cls_dist_euc.std()),
        }
        
        if gt_std > 1e-10 and cls_dist_euc.std() > 1e-10:
            r_euc, p_euc = spearmanr(gt_flat, cls_dist_euc)
            r_euc_p, _ = pearsonr(gt_flat, cls_dist_euc)
            result['spearman_euc'] = float(r_euc)
            result['pearson_euc'] = float(r_euc_p)
            result['pvalue_euc'] = float(p_euc)
        
        if gt_std > 1e-10 and cls_dist_cos.std() > 1e-10:
            r_cos, p_cos = spearmanr(gt_flat, cls_dist_cos)
            result['spearman_cos'] = float(r_cos)
            result['pvalue_cos'] = float(p_cos)
        
        return result


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
    parser.add_argument('--output_dir', default='/home/shrey/work/explainability/results/phase1')
    parser.add_argument('--min_seqs', type=int, default=6)
    parser.add_argument('--max_seqs', type=int, default=50)
    parser.add_argument('--wandb_project', default='phyla-explainability')
    parser.add_argument('--wandb_run_name', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    data_dir = Path(args.data_dir)
    fastas = sorted(list(data_dir.glob("*_alignment.fasta")))
    total = len(fastas)
    print(f"[INFO] Found {total} alignments", flush=True)
    
    run_name = args.wandb_run_name or f"phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project=args.wandb_project, name=run_name,
               config={**vars(args), 'total_alignments': total, 'residual_patch': True},
               tags=['phase1', 'cls', 'full-dataset'])
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("[INFO] Loading model...", flush=True)
    analyzer = PhylaCLSAnalyzer(device='cuda')
    print(f"[INFO] Model loaded with {analyzer.num_modules} modules", flush=True)
    
    all_results = []
    correlations_euc = []
    correlations_cos = []
    
    processed = skipped = errors = 0
    start_time = time.time()
    
    print(f"\n{'='*70}", flush=True)
    print(f"PHASE 1: Processing {total} alignments", flush=True)
    print(f"{'='*70}\n", flush=True)
    
    for idx, fasta in enumerate(fastas):
        aid = fasta.stem.replace("_alignment", "")
        
        try:
            names, seqs = load_fasta(fasta)
            
            if len(seqs) < args.min_seqs:
                skipped += 1
                continue
            
            if len(seqs) > args.max_seqs:
                idxs = sorted(np.random.choice(len(seqs), args.max_seqs, replace=False))
                names = [names[i] for i in idxs]
                seqs = [seqs[i] for i in idxs]
            
            clean = [''.join(c for c in s if c.isalpha() or c == '-') for s in seqs]
            
            result = analyzer.analyze_alignment(clean, names, aid)
            all_results.append(result)
            
            r_euc = result.get('spearman_euc')
            r_cos = result.get('spearman_cos')
            
            if r_euc is not None:
                correlations_euc.append(r_euc)
            if r_cos is not None:
                correlations_cos.append(r_cos)
            
            wandb.log({
                'idx': processed,
                'n_seq': result['n_sequences'],
                'msa_len': result['msa_length'],
                'spearman_euc': r_euc,
                'spearman_cos': r_cos,
                'cls_norm': result['cls_norm_mean'],
            })
            
            processed += 1
            
            if processed % 25 == 0 or processed <= 5:
                elapsed = time.time() - start_time
                rate = processed / elapsed * 60
                eta = (total - idx) / (processed / elapsed) / 60 if processed > 0 else 0
                mean_r = np.mean(correlations_euc) if correlations_euc else 0
                r_str = f"{r_euc:.3f}" if r_euc is not None else "N/A"
                
                print(f"[{processed:4d}/{total}] {aid:<18} n={result['n_sequences']:2d} "
                      f"r={r_str} | mean={mean_r:.3f} | "
                      f"{rate:.0f}/min | ETA:{eta:.0f}m", flush=True)
        
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"[ERROR] {aid}: {str(e)[:50]}", flush=True)
            continue
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}", flush=True)
    print(f"COMPLETE: {processed} processed | {skipped} skipped | {errors} errors", flush=True)
    print(f"Time: {elapsed/60:.1f}min ({processed/(elapsed/60):.0f}/min)", flush=True)
    print(f"\nSpearman (Euclidean): {np.mean(correlations_euc):.4f} +/- {np.std(correlations_euc):.4f}", flush=True)
    print(f"Spearman (Cosine):    {np.mean(correlations_cos):.4f} +/- {np.std(correlations_cos):.4f}", flush=True)
    print(f"{'='*70}", flush=True)
    
    wandb.log({
        'final/processed': processed,
        'final/mean_spearman_euc': np.mean(correlations_euc),
        'final/std_spearman_euc': np.std(correlations_euc),
        'final/mean_spearman_cos': np.mean(correlations_cos),
        'final/time_minutes': elapsed/60,
    })
    
    out_file = Path(args.output_dir) / f"phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        json.dump({'config': vars(args), 'results': all_results,
                   'summary': {'mean_euc': np.mean(correlations_euc), 
                              'std_euc': np.std(correlations_euc),
                              'n': processed}}, f)
    print(f"\n✓ Saved: {out_file}", flush=True)
    
    wandb.finish()


if __name__ == "__main__":
    main()
