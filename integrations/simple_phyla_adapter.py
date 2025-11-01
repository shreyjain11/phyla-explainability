"""
SIMPLE PHYLA ADAPTER - Proven Improvements Only

Based on debug results showing complex methods hurt performance.
Focus: CLS-only with normalization and smart distance metrics.

Target: r = 0.75+ (from baseline 0.66)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class SimplePhylaResult:
    sequence_embeddings: torch.Tensor
    pairwise_distances: torch.Tensor
    ground_truth_distances: Optional[torch.Tensor]
    sequence_names: List[str]
    distance_correlation: Optional[float]
    metadata: Dict[str, Any]


class SimplePhylaAnalyzer:
    """
    Simple, proven approach for phylogenetic distance prediction
    
    Based on debug findings:
    - CLS-only beats complex ensemble
    - L2 normalization helps
    - Last 2 layers better than all layers
    """
    
    def __init__(self, phyla_model, device='cuda'):
        self.phyla_model = phyla_model
        self.device = device
        self.phyla_model.to(device)
        self.phyla_model.eval()
    
    def analyze_sequences(
        self,
        sequences: List[str],
        sequence_names: List[str],
        ground_truth_distances: Optional[np.ndarray] = None,
        method: str = 'l2_norm',  # Options: 'l2_norm', 'cosine', 'last2_avg'
        verbose: bool = False
    ) -> SimplePhylaResult:
        """
        Simple CLS-based distance prediction
        
        Args:
            method: 'l2_norm' (normalize before distance), 
                   'cosine' (cosine distance),
                   'last2_avg' (average last 2 layers)
        """
        
        if verbose:
            print(f"[SIMPLE] Method: {method}")
            print(f"[SIMPLE] Sequences: {len(sequences)}")
        
        # Encode
        input_ids, cls_token_mask, sequence_mask, _ = self.phyla_model.encode(sequences, sequence_names)
        input_ids = input_ids.to(self.device)
        cls_token_mask = cls_token_mask.to(self.device)
        sequence_mask = sequence_mask.to(self.device)
        
        with torch.no_grad():
            if method == 'last2_avg':
                # Extract from last 2 modules
                x = self.phyla_model.modul[0](
                    input_ids,
                    logits=False,
                    position_ids=None,
                    sequence_mask=sequence_mask,
                    cls_token_mask=cls_token_mask
                )
                
                # Forward through all but last 2
                for module in self.phyla_model.modul[1:-2]:
                    correct_device = next(module.parameters()).device
                    x = module(
                        x.to(correct_device),
                        hidden_states_given=True,
                        logits=False,
                        position_ids=None,
                        sequence_mask=sequence_mask.to(correct_device),
                        cls_token_mask=cls_token_mask.to(correct_device)
                    )
                
                # Get second-to-last layer
                if len(self.phyla_model.modul) >= 2:
                    module = self.phyla_model.modul[-2]
                    correct_device = next(module.parameters()).device
                    x = module(
                        x.to(correct_device),
                        hidden_states_given=True,
                        logits=False,
                        position_ids=None,
                        sequence_mask=sequence_mask.to(correct_device),
                        cls_token_mask=cls_token_mask.to(correct_device)
                    )
                    
                    # Extract CLS from second-to-last
                    cls_second_last = x[cls_token_mask.to(correct_device)].view(
                        cls_token_mask.shape[0], cls_token_mask.sum(dim=1)[0], -1
                    ).to(self.device)
                else:
                    cls_second_last = None
                
                # Get last layer
                module = self.phyla_model.modul[-1]
                x = module(
                    x.to(self.device),
                    hidden_states_given=True,
                    logits=False,
                    position_ids=None,
                    sequence_mask=sequence_mask,
                    cls_token_mask=cls_token_mask
                )
                
                # Extract CLS from last
                if x.dim() == 3:
                    cls_last = x.squeeze(0)
                else:
                    cls_last = x
                
                # Average last 2 layers
                if cls_second_last is not None:
                    if cls_second_last.dim() == 3:
                        cls_second_last = cls_second_last.squeeze(0)
                    sequence_rep = 0.5 * cls_second_last + 0.5 * cls_last
                    if verbose:
                        print(f"[SIMPLE] Averaged last 2 layers")
                else:
                    sequence_rep = cls_last
                    if verbose:
                        print(f"[SIMPLE] Using only last layer (single module model)")
            
            else:
                # Standard: just forward through model
                x = self.phyla_model(input_ids, sequence_mask, cls_token_mask)
                
                # Extract CLS tokens
                if x.dim() == 3:
                    sequence_rep = x.squeeze(0)
                else:
                    sequence_rep = x
            
            if verbose:
                print(f"[SIMPLE] Embedding shape: {sequence_rep.shape}")
            
            # Compute distances based on method
            if method == 'cosine':
                # Cosine distance
                normed = F.normalize(sequence_rep, p=2, dim=-1)
                cos_sim = torch.mm(normed, normed.T)
                pairwise_distances = 1.0 - cos_sim
                
                if verbose:
                    print(f"[SIMPLE] Using cosine distance")
            
            elif method in ['l2_norm', 'last2_avg']:
                # L2 normalize then Euclidean
                normed = F.normalize(sequence_rep, p=2, dim=-1)
                euclidean = torch.cdist(normed, normed, compute_mode='donot_use_mm_for_euclid_dist')
                pairwise_distances = euclidean.max() - euclidean
                
                if verbose:
                    print(f"[SIMPLE] Using L2-normalized Euclidean (inverted)")
            
            else:
                # Fallback: regular inverted Euclidean
                euclidean = torch.cdist(sequence_rep, sequence_rep, compute_mode='donot_use_mm_for_euclid_dist')
                pairwise_distances = euclidean.max() - euclidean
                
                if verbose:
                    print(f"[SIMPLE] Using regular inverted Euclidean")
            
            if verbose:
                print(f"[SIMPLE] Distance range: [{pairwise_distances.min():.3f}, {pairwise_distances.max():.3f}]")
        
        # Compute correlation
        distance_correlation = None
        if ground_truth_distances is not None:
            gt_tensor = torch.from_numpy(ground_truth_distances).float().to(self.device)
            n = min(pairwise_distances.shape[0], gt_tensor.shape[0])
            
            model_vec = pairwise_distances[:n, :n][torch.triu_indices(n, n, offset=1)[0],
                                                    torch.triu_indices(n, n, offset=1)[1]]
            gt_vec = gt_tensor[:n, :n][torch.triu_indices(n, n, offset=1)[0],
                                        torch.triu_indices(n, n, offset=1)[1]]
            
            if len(model_vec) > 1:
                model_centered = model_vec - model_vec.mean()
                gt_centered = gt_vec - gt_vec.mean()
                corr = (model_centered * gt_centered).sum() / (
                    torch.sqrt((model_centered ** 2).sum() * (gt_centered ** 2).sum()) + 1e-8
                )
                distance_correlation = float(corr.item())
                
                if verbose:
                    print(f"[SIMPLE] Correlation: r={distance_correlation:.3f}")
        
        return SimplePhylaResult(
            sequence_embeddings=sequence_rep.detach().cpu(),
            pairwise_distances=pairwise_distances.detach().cpu(),
            ground_truth_distances=torch.from_numpy(ground_truth_distances) if ground_truth_distances is not None else None,
            sequence_names=sequence_names,
            distance_correlation=distance_correlation,
            metadata={
                "method": method,
                "num_sequences": len(sequences),
                "device": self.device
            }
        )

