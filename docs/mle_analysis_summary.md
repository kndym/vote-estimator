# MLE Analysis Summary

## Problem Identified
The model was producing uniform probabilities (~0.25 each) instead of matching actual vote proportions:
- Actual: D=38.48%, R=14.63%, O=0.57%, N=46.32%
- Model output: D≈25%, R≈25%, O≈25%, N≈25%

## Root Causes

1. **Spatial Term Dominance**: The Dirichlet spatial smoothing term was 2-13x larger than the DirMult data-fitting term, pulling probabilities toward uniform distributions through neighbor averaging.

2. **Preprocessing Issue**: Setting minimum vote totals to 1 BEFORE scaling inflated small vote types (especially O votes), distorting proportions.

3. **Uniform Initialization**: Starting from flat Dirichlet (uniform) initialization made it hard to escape the uniform solution.

## Fixes Implemented

1. **Fixed Preprocessing** (lines 125-176):
   - Scale vote totals FIRST to match demo totals
   - Set minimums AFTER scaling to preserve proportions
   - Re-normalize to maintain consistency

2. **Added DirMult Weighting** (parameter `--dir-mult-weight`):
   - Allows balancing data fit vs spatial smoothing
   - Tested weights from 1 to 5000

3. **Data-Based Initialization** (lines 178-217):
   - Initialize from aggregate vote proportions instead of uniform
   - Uses Dirichlet with concentration parameters based on actual vote shares
   - Provides better starting point for optimization

4. **Enhanced Diagnostics**:
   - Print U vs V proportions every 10 iterations
   - Show DirMult vs Dirichlet term magnitudes and ratio
   - Track neighbor similarity and average probabilities

## Results

With weight=5000 and data-based initialization:
- O votes: Improved from ~20% to ~0.5% (target: 0.57%) ✓
- D votes: Still low (~28% vs target 38%) ✗
- R votes: Too low (~4% vs target 14%) ✗
- N votes: Too high (~68% vs target 46%) ✗

## Remaining Issues

1. **Optimization Instability**: Model diverges over iterations, moving away from target
2. **Spatial Term Still Too Strong**: Even with weight=5000, spatial term dominates
3. **Identification Problem**: Many p configurations produce same aggregate U, making optimization ambiguous
4. **Gradient Structure**: Spatial gradients may be pushing in wrong direction when neighbors have different patterns

## Recommendations

1. **Much Higher Weight**: Try weight=50000+ to truly dominate spatial term
2. **Reduce Spatial Smoothing**: Make spatial term adaptive or divide by neighbor count
3. **Alternative Model**: Consider model that directly constrains demographic differences
4. **Different Optimization**: Try constrained optimization or different likelihood structure
