# Critical Bug Fix: Missing Residual Connections

## Discovery: January 15, 2026

The `BiMambaWrapper` in Phyla was missing residual connections, causing signal collapse.

| Layer | Before Fix | After Fix |
|-------|-----------|-----------|
| embedding | 10.70 | 13.82 |
| layer_1 | 0.0008 | 13.82 |
| layer_3+ | 0.0 | 13.82 |

**Result**: r = 0.54 ± 0.23 correlation with phylogenetic distances (n=2,367)
