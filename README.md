# Phyla Explainability Integration

Integration of explainability methods into the Phyla phylogenetic inference model, enabling attribution analysis and distance prediction validation on protein family alignments.

## Overview

This project integrates attention-based attribution methods with the Phyla model (a hybrid Mamba-transformer architecture for phylogenetic inference) to provide interpretable predictions of evolutionary relationships between protein sequences.

## Key Features

- **Attention-based Attribution**: Extract position-level importance scores from Phyla's tree_head attention mechanism
- **Phylogenetic Distance Prediction**: Predict pairwise evolutionary distances between protein sequences
- **Multi-Strategy Ensemble**: Adaptive combination of embedding strategies based on alignment characteristics
- **Large-Scale Validation**: Tested on 3,321 protein family alignments from OpenProtein dataset
- **Publication-Quality Visualizations**: Automated generation of professional figures for analysis

## Performance

Validated on **3,321 protein family alignments** from the OpenProtein dataset:

- **Mean Pearson correlation**: r = 0.655
- **Median correlation**: r = 0.719
- **Excellent performance** (r > 0.90): 373 alignments (11.2%)
- **Strong performance** (r > 0.70): 1,412 alignments (42.5%)

## Installation

```bash
# Clone repository
git clone https://github.com/shreyjain11/phyla-explainability.git
cd phyla-explainability

# Install dependencies
pip install -r requirements.txt

# Install Phyla and MambaLRP submodules
cd Phyla && pip install -e . && cd ..
cd MambaLRP && pip install -e . && cd ..
```

## Usage

### Basic Usage

```python
from Phyla.phyla.model.model import Phyla
from integrations.correct_phyla_adapter import CorrectPhylaMambaLRPAnalyzer

# Load Phyla model
phyla_model = Phyla(name='phyla-alpha')
phyla_model.load()

# Initialize analyzer
analyzer = CorrectPhylaMambaLRPAnalyzer(phyla_model, device='cuda')

# Analyze sequences
result = analyzer.analyze_sequences(
    sequences=your_sequences,
    sequence_names=your_names,
    ground_truth_distances=ground_truth_matrix  # Optional
)

# Access results
print(f"Distance correlation: {result.distance_correlation}")
print(f"Attention relevances shape: {result.relevances.shape}")
print(f"Pairwise distances shape: {result.pairwise_distances.shape}")
```

### Generate Figures

```bash
python3 scripts/generate_figures_full_dataset.py
```

Generates 6 publication-quality figures:
1. Correlation distribution histogram
2. Performance breakdown bar chart
3. Correlation vs alignment size analysis
4. Model vs ground truth scatter plots
5. Attention relevance heatmaps
6. Distance matrix comparisons

## Architecture

The integration provides three adapter implementations:

1. **`correct_phyla_adapter.py`** (Recommended)
   - Multi-layer fusion of CLS embeddings
   - Position-wise gap weighting
   - Adaptive ensemble of 4 embedding strategies
   - Data-driven strategy selection based on alignment characteristics

2. **`simple_phyla_adapter.py`** (Baseline)
   - CLS-only approach with L2 normalization
   - Cosine distance option
   - Last-2-layer averaging

3. **`position_aware_adapter.py`** (Experimental)
   - Position-level phylogenetic signal computation
   - Conservation-aware attribution

## Key Innovations

1. **Inverted Euclidean Distance**: Corrects for the fact that Phyla embeds similar sequences close together, while phylogenetic distance measures dissimilarity
   
2. **Adaptive Embedding Strategy**: Switches between CLS-only, attention-only, or combined approaches based on embedding quality (cosine similarity)

3. **Gap-Aware Weighting**: Adjusts for high-gap alignments by weighting positions based on information content

4. **Multi-Layer Fusion**: Combines embeddings from multiple Phyla modules with depth-based weighting

## Results

See `outputs/figures_full_3321/` for detailed visualizations from the full 3,321-alignment validation.

Key findings:
- Model performs excellently on diverse, well-represented protein families
- Performance degrades with high gap content (>50%) and low sequence diversity
- Attention patterns reveal phylogenetically informative positions
- Distance predictions capture evolutionary relationships with moderate-to-strong correlation

---

## Phase 1 Results: CLS Embedding Analysis

### Key Finding
CLS embeddings encode phylogenetic signal with **mean Spearman r = 0.54 ± 0.23** across 2,367 protein families.

### Critical Bug Discovery
Identified missing residual connections in `BiMambaWrapper.forward()` - see `docs/RESIDUAL_BUG_FIX.md`

### Results Summary
| Metric | Value |
|--------|-------|
| Alignments analyzed | 2,367 |
| Mean Spearman r | 0.54 ± 0.23 |
| Strong correlation (r ≥ 0.7) | 24.1% |
| p-value | < 0.001 |
