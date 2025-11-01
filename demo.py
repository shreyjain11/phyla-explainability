#!/usr/bin/env python3
"""
Simple demonstration of Phyla-MambaLRP Integration

This script provides a minimal example that users can run immediately
to see the integration in action.

Usage:
    python demo.py
"""

import torch
import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("Phyla-MambaLRP Integration Demo")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        from Phyla.phyla.model.model import Phyla
        from integrations.phyla_mambalrp_adapter import PhylaMambaLRPAnalyzer
        print("[OK] Successfully imported Phyla and MambaLRP modules")
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        print("Please ensure Phyla and MambaLRP are properly installed.")
        return False
    
    print("\nLoading Phyla model...")
    try:
        model = Phyla(name='phyla-alpha', device=device)
        model.eval()
        print("[OK] Phyla model initialized")
        
        try:
            model.load()
            print("[OK] Loaded pretrained weights")
        except Exception as e:
            print(f"[WARNING] Using random weights (pretrained not available): {e}")
            
    except Exception as e:
        print(f"[ERROR] Failed to initialize Phyla: {e}")
        return False
    
    print("\nInitializing MambaLRP analyzer...")
    try:
        analyzer = PhylaMambaLRPAnalyzer(model, device=device)
        print("[OK] MambaLRP analyzer ready")
    except Exception as e:
        print(f"[ERROR] Failed to initialize analyzer: {e}")
        return False
    
    print("\nAnalyzing demo sequences...")
    sequences = [
        "AADD",    # seq1: Different ending
        "AAAA",    # seq2: Same ending as seq3
        "AAAC"     # seq3: Similar to seq2
    ]
    names = ["seq1_AADD", "seq2_AAAA", "seq3_AAAC"]
    
    print("Input sequences:")
    for name, seq in zip(names, sequences):
        print(f"  {name}: {seq}")
    
    try:
        result = analyzer.analyze_sequences(sequences, names)
        print("[OK] Analysis completed successfully!")
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        return False
    
    print("\nResults:")
    print("-" * 30)
    
    # Pairwise distances
    print("Pairwise Distances (evolutionary similarity):")
    distances = result.pairwise_distances.numpy()
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i < j:  # Only print upper triangle
                dist = distances[i, j]
                print(f"  {name_i} ↔ {name_j}: {dist:.4f}")
    
    print("\nExpected: seq2 and seq3 should be most similar (smallest distance)")
    most_similar_pair = np.unravel_index(np.argmin(distances + np.eye(len(names)) * 999), distances.shape)
    print(f"Most similar: {names[most_similar_pair[0]]} ↔ {names[most_similar_pair[1]]}")
    
    # Per-position relevances
    print("\nPer-Position Relevance Scores:")
    relevances = result.relevances[0].numpy()  # First batch
    
    # Calculate sequence boundaries (accounting for CLS tokens)
    seq_lengths = [len(seq) for seq in sequences]
    seq_boundaries = []
    current_pos = 0
    for seq_len in seq_lengths:
        seq_boundaries.append((current_pos + 1, current_pos + seq_len + 1))  # Skip CLS
        current_pos += seq_len + 1
    
    for i, (name, seq, (start, end)) in enumerate(zip(names, sequences, seq_boundaries)):
        seq_relevance = relevances[start:end]
        # Normalize for display
        if seq_relevance.max() > 0:
            seq_relevance_norm = seq_relevance / seq_relevance.max()
        else:
            seq_relevance_norm = seq_relevance
        
        print(f"  {name} ({seq}):")
        for pos, (aa, rel) in enumerate(zip(seq, seq_relevance_norm)):
            print(f"    Position {pos+1} ({aa}): {rel:.3f}")
    
    print("\nExpected: Later positions should have higher relevance scores")
    print("(since they distinguish between similar/different sequences)")
    
    print(f"\nAnalysis Summary:")
    print(f"  Objective value: {result.objective_value:.4f}")
    print(f"  Number of sequences: {result.metadata['num_sequences']}")
    print(f"  Device used: {result.metadata['device']}")
    print(f"  MambaLRP available: {result.metadata['mambalrp_available']}")
    
    print(f"\nBiological Interpretation:")
    d12 = distances[0, 1]
    d13 = distances[0, 2]
    d23 = distances[1, 2]
    
    if d23 < d12 and d23 < d13:
        print("[OK] Expected clustering: seq2 (AAAA) and seq3 (AAAC) are most similar")
        print("     This makes biological sense - they differ by only one amino acid")
    else:
        print("[WARNING] Unexpected clustering pattern - model may need more training")
    
    avg_final_relevance = np.mean([seq_relevance[-1] for _, _, (start, end) in zip(names, sequences, seq_boundaries) 
                                  for seq_relevance in [relevances[start:end]]])
    avg_first_relevance = np.mean([seq_relevance[0] for _, _, (start, end) in zip(names, sequences, seq_boundaries) 
                                  for seq_relevance in [relevances[start:end]]])
    
    if avg_final_relevance > avg_first_relevance:
        print("[OK] Expected relevance pattern: Final positions are more important")
        print("     This suggests the model focuses on distinguishing sequence endings")
    else:
        print("[WARNING] Unexpected relevance pattern - may indicate model artifacts")
    
    print(f"\nDemo completed successfully!")
    print("Ready to analyze your own biological sequences!")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
