"""CLS Embedding Analysis for Phylogenetic Signal Detection"""
import torch
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

class PhylaCLSAnalyzer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def compute_sequence_distances(self, sequences):
        n = len(sequences)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                s1, s2 = sequences[i], sequences[j]
                matches = sum(1 for a, b in zip(s1, s2) if a == b and a != '-' and b != '-')
                aligned = sum(1 for a, b in zip(s1, s2) if a != '-' and b != '-')
                dist[i, j] = dist[j, i] = 1 - (matches / aligned if aligned > 0 else 0)
        return dist
    
    def extract_cls_embeddings(self, sequences, names):
        input_ids, cls_mask, seq_mask, _ = self.model.encode(sequences, names)
        input_ids, cls_mask, seq_mask = [x.to(self.device) for x in [input_ids, cls_mask, seq_mask]]
        
        final_output = None
        def hook(m, i, o): 
            nonlocal final_output
            final_output = (o[0] if isinstance(o, tuple) else o).detach().cpu()
        
        handle = self.model.modul[-1].register_forward_hook(hook)
        with torch.no_grad():
            x = self.model.modul[0](input_ids, logits=False, position_ids=None,
                sequence_mask=seq_mask, cls_token_mask=cls_mask)
            for module in self.model.modul[1:]:
                dev = next(module.parameters()).device
                x = module(x.to(dev), hidden_states_given=True, logits=False,
                    position_ids=None, sequence_mask=seq_mask.to(dev), cls_token_mask=cls_mask.to(dev))
        handle.remove()
        return final_output[0].numpy()
    
    def analyze_alignment(self, sequences, names):
        gt_dist = self.compute_sequence_distances(sequences)
        gt_flat = squareform(gt_dist)
        if gt_flat.std() < 1e-10: return {'error': 'No variance'}
        
        cls_emb = self.extract_cls_embeddings(sequences, names)
        cls_dist = pdist(cls_emb, metric='euclidean')
        
        r, p = spearmanr(gt_flat, cls_dist) if cls_dist.std() > 1e-10 else (0, 1)
        return {'n_sequences': len(sequences), 'spearman_euclidean': float(r), 'pvalue': float(p)}
