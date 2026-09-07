## EI Model Theory and Technical Details (LapEnt Neighbor-Weighted)

This document describes the current ecological inference (EI) model implemented in
`scripts/mle_agent_kv_lapent.py`. The model estimates per-precinct, per-demographic
vote probability vectors using a Laplacian smoothing term, an entropy term, and a
data-fit term, optimized with Adam in probability space.

---

## 1. Model Entities and Shapes

Let:
- `i` index precincts (block groups)
- `d` index demographic groups in `DEMOGRAPHICS`
- `k` index vote types in `VOTE_TYPES`

Key tensors:
- `D[i, d]` : demographic population (CVAP)
- `V[i, k]` : observed vote totals (scaled)
- `p[i, d, k]` : unknown probability of vote type `k` for demographic `d` in precinct `i`

Constraints:
- `p[i, d, k] >= 0` for all `i,d,k`
- `sum_k p[i, d, k] = 1` for all `i,d`


## 2. Data Preparation and Scaling

The model works with:
- `votes_D`, `votes_R`, `votes_O`, and `votes_N` (non-votes)
- demographic CVAP columns (White, Hispanic, Black, Asian, Other)

Votes are scaled to match demographic totals:
- Let `vote_totals[i] = sum_k V_raw[i, k]`
- Let `demo_totals[i] = sum_d D_raw[i, d]`
- Scaling: `V_scaled[i, k] = V_raw[i, k] * demo_totals[i] / vote_totals[i]`

Then:
- `V_scaled` is renormalized to match demo totals.
- `D` is protected from zeros with:
  - `D = D + EPSILON`
  - `D = max(D, 1.0)`


## 3. Forward Model (Aggregation)

Predicted vote totals per precinct are computed by:

```
U[i, k] = sum_d p[i, d, k] * D[i, d]
```

This ties the demographic probabilities to the observed totals.


## 4. Objective (Loss) Overview

The optimizer uses a sum of three components:

1) **Data fit (L2)**  
   Penalizes discrepancy between predicted and observed vote totals:
   ```
   L_data = sum_i ||U[i, :] - V[i, :]||^2
   ```

2) **Spatial smoothing (neighbor-weighted Laplacian)**  
   Encourages each precinct's demographic probability vector to resemble a
   population-weighted neighbor average:
   ```
   neighbor_avg[i, d, :] =
       (sum_j A[i, j] * p[j, d, :] * D[j, d]) / (sum_j A[i, j] * D[j, d] + EPSILON)
   L_spatial = sum_i,d ||p[i, d, :] - neighbor_avg[i, d, :]||^2
   ```

3) **Entropy regularization**  
   Encourages non-degenerate probabilities:
   ```
   L_entropy = sum_i,d,k p[i, d, k] * log(p[i, d, k] + EPSILON)
   ```

Total loss (implicit in gradients):
```
L = L_data + spatial_weight * L_spatial + entropy_weight * L_entropy
```


## 5. Gradient Components (Implementation Details)

### 5.1 Data fit gradient
```
grad_data[i, d, k] = 2 * (U[i, k] - V[i, k]) * D[i, d]
```

### 5.2 Spatial gradient (neighbor-weighted)
For each demographic:
```
diff[i, d, :] = p[i, d, :] - neighbor_avg[i, d, :]
grad_spatial[i, d, :] = 2 * diff[i, d, :]
```

### 5.3 Entropy gradient
```
grad_entropy[i, d, k] = log(p[i, d, k] + EPSILON) + 1
```

Total gradient:
```
grads = grad_data + spatial_weight * grad_spatial + entropy_weight * grad_entropy
```


## 6. Low-Population Handling

### 6.1 Noise (optional)
If enabled, small noise is injected for low-pop cells to prevent lock-in:
```
low_pop_weight = exp(-D / (low_pop_scale + EPSILON))
grads += noise * low_pop_weight
```

### 6.2 Low-pop blending (current behavior)
After each update and projection, the model blends low-pop cells toward the
population-weighted neighbor average (not uniform):

```
low_pop_weight = exp(-D / (low_pop_scale + EPSILON))
p = (1 - low_pop_weight) * p + low_pop_weight * neighbor_avg
```

This ensures:
- Low-pop precinct/demos inherit signal from neighbors.
- High-pop precinct/demos remain mostly unchanged.


## 7. Optimization and Constraints

The model is optimized in probability space with Adam:
```
p <- p - lr * Adam(grads)
```

After every step:
```
p = clip(p, EPSILON, +inf)
p = p / sum_k p
```

This enforces the simplex constraint per precinct and demographic.


## 8. Interpretation

The model is a structured EI estimator:
- **Data fit** enforces consistency with observed totals.
- **Neighbor smoothing** propagates high-pop signal across the graph,
  weighted by demographic population.
- **Entropy** prevents extreme, brittle distributions.
- **Low-pop blending** prevents collapse by steering low-pop demos toward
  neighbor-weighted averages rather than uniform probabilities.


## 9. Key Implementation References

- Data scaling and CVAP preparation:
  - `preprocess_counts()` in `scripts/mle_agent_kv_lapent.py`
- Spatial gradient (neighbor-weighted):
  - `run_optimization()` in `scripts/mle_agent_kv_lapent.py`
- Low-pop blending (neighbor-weighted):
  - `run_optimization()` in `scripts/mle_agent_kv_lapent.py`


## 10. Practical Notes

- The model is sensitive to `spatial_weight`, `entropy_weight`, and
  `low_pop_scale`. These control smoothing strength, distribution sharpness,
  and low-pop influence respectively.
- Population-weighted neighbors make low-pop neighbors carry less influence,
  allowing high-pop precincts to dominate smoothing and signal propagation.
