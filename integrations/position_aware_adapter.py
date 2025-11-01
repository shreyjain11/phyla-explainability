import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class PositionAwareAttributionResult:
    relevances: torch.Tensor
    sequence_embeddings: torch.Tensor
    pairwise_distances: torch.Tensor
    ground_truth_distances: Optional[torch.Tensor]
    sequence_names: List[str]
    objective_value: float
    distance_correlation: Optional[float]
    conservation_correlation: Optional[float]
    metadata: Dict[str, Any]


def compute_position_phylogenetic_signal(
    hidden_states: torch.Tensor,
    cls_token_mask: torch.Tensor,
    sequence_mask: torch.Tensor,
    ground_truth_distances: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    POSITION-AWARE objective: Compute phylogenetic signal at each position
    
    Key idea: Positions that vary between distant sequences should produce
    different embeddings, while positions that are same should produce similar embeddings.
    
    Args:
        hidden_states: [batch, total_tokens, dim] all token embeddings
        cls_token_mask: [batch, total_tokens] mask for CLS tokens
        sequence_mask: [batch, total_tokens] which sequence each token belongs to
        ground_truth_distances: [num_seqs, num_seqs] optional phylogenetic distances
    
    Returns:
        objective: scalar to maximize
    """
    
    position_embeddings = hidden_states[~cls_token_mask]
    position_seq_ids = sequence_mask[~cls_token_mask]
    
    if ground_truth_distances is not None:
        num_seqs = cls_token_mask.sum().item()
        seq_ids = torch.unique(position_seq_ids)
        tokens_per_seq = (~cls_token_mask[0]).sum().item() // num_seqs
        
        position_objectives = []
        
        for pos_idx in range(min(tokens_per_seq, 100)):
            pos_embeddings = []
            for seq_id in range(num_seqs):
                seq_tokens = (position_seq_ids == seq_id).nonzero(as_tuple=True)[0]
                if pos_idx < len(seq_tokens):
                    pos_embeddings.append(position_embeddings[seq_tokens[pos_idx]])
            
            if len(pos_embeddings) == num_seqs:
                pos_embeddings = torch.stack(pos_embeddings)
                
                pos_distances = torch.cdist(pos_embeddings.unsqueeze(0), 
                                           pos_embeddings.unsqueeze(0))[0]
                pos_distances_inv = pos_distances.max() - pos_distances
                gt_device = ground_truth_distances.to(pos_distances.device)
                
                n = min(num_seqs, gt_device.shape[0])
                pos_vec = pos_distances_inv[:n, :n][torch.triu_indices(n, n, offset=1)[0],
                                                     torch.triu_indices(n, n, offset=1)[1]]
                gt_vec = gt_device[:n, :n][torch.triu_indices(n, n, offset=1)[0],
                                            torch.triu_indices(n, n, offset=1)[1]]
                
                pos_obj = -F.mse_loss(pos_vec / (pos_vec.max() + 1e-8), 
                                      gt_vec / (gt_vec.max() + 1e-8))
                position_objectives.append(pos_obj)
        
        if position_objectives:
            return torch.stack(position_objectives).mean()
        else:
            return torch.tensor(0.0, device=hidden_states.device, requires_grad=True)
    
    else:
        num_seqs = cls_token_mask.sum().item()
        tokens_per_seq = (~cls_token_mask[0]).sum().item() // num_seqs
        
        position_variances = []
        
        for pos_idx in range(min(tokens_per_seq, 100)):
            pos_embeddings = []
            for seq_id in range(num_seqs):
                seq_tokens = (position_seq_ids == seq_id).nonzero(as_tuple=True)[0]
                if pos_idx < len(seq_tokens):
                    pos_embeddings.append(position_embeddings[seq_tokens[pos_idx]])
            
            if len(pos_embeddings) >= 2:
                pos_embeddings = torch.stack(pos_embeddings)
                pos_var = torch.var(pos_embeddings, dim=0).mean()
                position_variances.append(pos_var)
        
        if position_variances:
            return torch.stack(position_variances).mean()
        else:
            return torch.tensor(0.0, device=hidden_states.device, requires_grad=True)


def grad_x_input_position_aware(
    embeddings: torch.Tensor,
    objective: torch.Tensor,
    sequence_lengths: List[int],
    normalize: bool = True,
) -> torch.Tensor:
    """Compute gradient × input with per-sequence normalization"""
    
    grads = torch.autograd.grad(
        outputs=objective,
        inputs=embeddings,
        retain_graph=True,
        allow_unused=False,
        create_graph=False
    )[0]
    
    if grads is None:
        raise RuntimeError("Gradients are None")
    
    with torch.no_grad():
        R = (embeddings * grads).sum(dim=2)
        
        if normalize:
            start = 0
            for L in sequence_lengths:
                end = start + L + 1
                if end <= R.shape[1]:
                    seg = R[:, start:end]
                    max_abs = seg.abs().max()
                    if max_abs > 1e-8:
                        R[:, start:end] = seg / max_abs
                start = end
    
    return R


class PositionAwarePhylaMambaLRPAnalyzer:
    """
    POSITION-AWARE analyzer for biologically truthful attributions
    
    Key innovation: Uses position-level phylogenetic signal, not just sequence-level
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
        ground_truth_distances: Optional[np.ndarray] = None
    ) -> PositionAwareAttributionResult:
        """
        Analyze with POSITION-AWARE objective
        """
        
        input_ids, cls_token_mask, sequence_mask, _ = self.phyla_model.encode(sequences, sequence_names)
        input_ids = input_ids.to(self.device)
        cls_token_mask = cls_token_mask.to(self.device)
        sequence_mask = sequence_mask.to(self.device)
        
        gt_tensor = None
        if ground_truth_distances is not None:
            gt_tensor = torch.from_numpy(ground_truth_distances).float()
            n = len(sequences)
            if gt_tensor.shape[0] > n:
                gt_tensor = gt_tensor[:n, :n]
        
        print(f"[POSITION-AWARE] Analyzing {len(sequences)} sequences...")
        print(f"[POSITION-AWARE] Using position-level phylogenetic objective")
        
        seq_lengths = [len(s) for s in sequences]
        
        with torch.enable_grad():
            embedding_layer = self.phyla_model.modul[0].backbone.embedding
            embeddings = embedding_layer(input_ids)
            embeddings = embeddings.requires_grad_(True)
            
            hidden_states = embeddings
            
            hidden_states = self.phyla_model.modul[0].backbone(
                hidden_states,
                inference_params=None,
                position_ids=None,
                hidden_states_given=True
            )
            
            for module in self.phyla_model.modul[1:-1]:
                correct_device = next(module.parameters()).device
                hidden_states = module(
                    hidden_states.to(correct_device),
                    hidden_states_given=True,
                    logits=False,
                    position_ids=None,
                    sequence_mask=sequence_mask.to(correct_device),
                    cls_token_mask=cls_token_mask.to(correct_device)
                )
            
            output = self.phyla_model.modul[-1](
                hidden_states.to(self.device),
                hidden_states_given=True,
                logits=False,
                position_ids=None,
                sequence_mask=sequence_mask,
                cls_token_mask=cls_token_mask
            )
            
            if output.dim() == 3 and output.size(1) == cls_token_mask.sum(dim=1)[0].item():
                sequence_rep = output
            else:
                sequence_rep = output[cls_token_mask].view(1, cls_token_mask.sum().item(), -1)
            
            objective = compute_position_phylogenetic_signal(
                hidden_states,
                cls_token_mask,
                sequence_mask,
                gt_tensor
            )
            
            print(f"[POSITION-AWARE] Objective: {objective.item():.6f}")
            
            relevances = grad_x_input_position_aware(embeddings, objective, seq_lengths, normalize=True)
        
        with torch.no_grad():
            euclidean_dist = torch.cdist(sequence_rep[0], sequence_rep[0])
            pairwise_distances = euclidean_dist.max() - euclidean_dist
        
        distance_correlation = None
        if gt_tensor is not None:
            with torch.no_grad():
                gt_device = gt_tensor.to(pairwise_distances.device)
                n = min(pairwise_distances.shape[0], gt_device.shape[0])
                
                model_vec = pairwise_distances[:n, :n][torch.triu_indices(n, n, offset=1)[0],
                                                        torch.triu_indices(n, n, offset=1)[1]]
                gt_vec = gt_device[:n, :n][torch.triu_indices(n, n, offset=1)[0],
                                            torch.triu_indices(n, n, offset=1)[1]]
                
                if len(model_vec) > 1:
                    model_centered = model_vec - model_vec.mean()
                    gt_centered = gt_vec - gt_vec.mean()
                    corr = (model_centered * gt_centered).sum() / (
                        torch.sqrt((model_centered ** 2).sum() * (gt_centered ** 2).sum()) + 1e-8
                    )
                    distance_correlation = float(corr.item())
        
        print(f"[POSITION-AWARE] Relevances: norm={torch.norm(relevances).item():.6f}")
        if distance_correlation:
            print(f"[POSITION-AWARE] Distance correlation: r={distance_correlation:.4f}")
        
        return PositionAwareAttributionResult(
            relevances=relevances.detach().cpu(),
            sequence_embeddings=sequence_rep.detach().cpu(),
            pairwise_distances=pairwise_distances.detach().cpu(),
            ground_truth_distances=gt_tensor.cpu() if gt_tensor is not None else None,
            sequence_names=sequence_names,
            objective_value=float(objective.item()),
            distance_correlation=distance_correlation,
            conservation_correlation=None,  # Computed externally
            metadata={
                "device": self.device,
                "num_sequences": len(sequences),
                "sequence_lengths": seq_lengths,
                "objective_type": "position_aware_phylogenetic"
            }
        )

