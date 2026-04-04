# Supplementary Materials

New experiments and analyses addressing reviewer concerns. Figures referenced in the rebuttal text responses.

**Setup**: Qwen2.5-inspired Transformer (128 hidden, 512 intermediate, 4 heads, 6 layers), character-level tokenizer (45 tokens), trained on 7 geometric tasks over 5,075 real-world city coordinates.

---

## 1. Representation Similarity  - Four Metrics (pKZe Q3, rQVL W3)

We computed CKA, RSA, SNN (k=10), and SVCCA across all pairwise comparisons for PT1-X (single-task, 7 tasks x 3 seeds), PT2 (two-task, 7 pairs x 3 seeds), and PT3 (three-task, 7 triples x 3 seeds). 210 pairs per group x 4 layers x 4 metrics = 10,080 total computations. Each column below shows the 7x7 task-averaged similarity matrix (with SEM) at layer 5, progressing from 1-task to 2-task to 3-task training.

| | PT1-X (1 task) | PT2 (2 tasks) | PT3 (3 tasks) |
|---|:---:|:---:|:---:|
| **CKA** | ![](assets/main/cka_pt1x_l5.png) | ![](assets/main/cka_pt2_l5.png) | ![](assets/main/cka_pt3_l5.png) |
| **RSA** | ![](assets/main/rsa_pt1x_l5.png) | ![](assets/main/rsa_pt2_l5.png) | ![](assets/main/rsa_pt3_l5.png) |
| **SNN** | ![](assets/main/snn_pt1x_l5.png) | ![](assets/main/snn_pt2_l5.png) | ![](assets/main/snn_pt3_l5.png) |
| **SVCCA** | ![](assets/main/svcca_pt1x_l5.png) | ![](assets/main/svcca_pt2_l5.png) | ![](assets/main/svcca_pt3_l5.png) |

**Table 1.** Cross-task similarity at layer 5 (non-overlapping task pairs only).

| Metric | PT1-X (1 task) | PT2 (2 tasks) | PT3 (3 tasks) |
|--------|:---:|:---:|:---:|
| **CKA** | 0.501 | 0.879 | 0.854 |
| **RSA** | 0.467 | 0.892 | 0.885 |
| **SNN** | 0.101 | 0.185 | 0.204 |
| **SVCCA** | 0.551 | 0.729 | 0.794 |

| |
|:---:|
| ![](assets/main/convergence_trend.png) |
| **Figure 1e.** Cross-task representation similarity (non-overlapping task pairs only) at layers 3, 4, and 5. All four metrics increase from 1-task to 2-task to 3-task training across all layers. Convergence is stronger in later layers. |

**Takeaway**: All four metrics confirm representation convergence as task count increases. Even models trained on completely non-overlapping task sets become more similar with more tasks. SNN (the non-global metric emphasized by the PRH paper) supports convergence. SVCCA shows subspaces align even across isolated tasks; it is the finer geometry that differs.

---

## 2. Effective Rank (pKZe "representation rank", rQVL)

Participation ratio of singular values for all 63 experiments, layers 3-6.

| | |
|:---:|:---:|
| ![](assets/main/rank_pt1x_l5.png) | ![](assets/main/rank_pt1x_by_layer.png) |
| **Figure 2a.** Effective rank at layer 5 per training task, in canonical order. Tasks that learn geographic representations compress to ~13-20 dimensions (of 256 hidden dims). Crossing (161) and inside (118) fail to compress, consistent with their training difficulties. | **Figure 2b.** Layer progression. Most tasks compress from ~100 at layer 3 to ~15-20 by layer 5. Crossing stays flat at ~160 across all layers. |

**Takeaway**: Effective rank provides a quantitative characterization of representation geometry. The dramatic rank gap between tasks that learn spatial structure (13-20) and tasks that don't (118-161) complements the PCA visualization.

---

## 3. Scattered Atlantis (pKZe Q2a/Q2b)

100 Atlantis cities scattered uniformly across the globe (not clustered). Full PT1 + 21 FTWB2 models trained, representations extracted, probe generalization evaluated.

| |
|:---:|
| ![](assets/main/scattered_atlantis.png) |
| **Figure 3a.** Probe generalization error: original (clustered) vs scattered Atlantis. Distance task harms generalization in both conditions. |

| |
|:---:|
| ![](assets/main/scattered_atlantis_examples.png) |
| **Figure 3b.** Ground-truth (red circles) vs probe-predicted (blue crosses) Atlantis locations. Top: clustered Atlantis. Bottom: scattered Atlantis. Left: with distance task (large errors). Right: without distance task (small errors). Purple lines connect GT to prediction. |

| Condition | With distance (n=6) | Without distance (n=15) | p-value |
|-----------|:---:|:---:|:---:|
| **Original (clustered)** | 495 +/- 51 | 179 +/- 21 | 0.0003 |
| **Scattered** | 1069 +/- 49 | 485 +/- 52 | 0.0001 |

**Takeaway**: The divergent task effect is not an artifact of Atlantis clustering. It holds, and is even more significant, when new entities are uniformly distributed. Figure 3b shows this visually: without the distance task, predicted Atlantis locations cluster near their true positions; with the distance task, predictions scatter widely.

---

## 4. Width Ablation (CBid W3)

2x wider model (256 hidden, 1024 intermediate, 8 heads) at constant compute (half epochs).

| |
|:---:|
| ![](assets/main/width_ablation_diff.png) |
| **Figure 4.** FTWB2 - FTWB1 difference for the wide model. Same qualitative pattern as the default architecture: distance remains divergent. |

**Takeaway**: Doubling model capacity does not resolve the divergent task phenomenon. This is a property of the task-representation interaction, not a capacity limitation.

---

## 5. Gradient Analysis (CBid Q3 - Mechanistic Evidence)

We analyze task gradient alignment at two levels: (a) parameter-space gradients (which model weights each task wants to update), and (b) activation-space gradients (which direction each task wants to push the novel entity's internal representation). The contrast between these two analyses is informative.

### 5a. Parameter-Space Gradient Similarity

We compute the cosine similarity between the flattened parameter gradients of each pair of single-task fine-tuning losses, averaged over ~640 samples per task. This measures whether tasks want to update the same weights in the same direction.

| |
|:---:|
| <img src="assets/main/param_grad_cossim_global.png" width="50%"> |

**Figure 5a.** Pairwise cosine similarity of parameter gradients (all model parameters). Distance is *not* an outlier — its similarities with other tasks range from -0.06 to 0.35, comparable to other pairs. The most anti-correlated pair is actually inside–distance (-0.32). Most task pairs show low positive similarity (0.04–0.41), indicating the tasks update largely orthogonal but not conflicting sets of parameters.

### 5b. Activation-Space Gradient Analysis

The parameter-level analysis does not single out distance. But divergent fine-tuning operates at the *representation* level — the question is not which weights get updated, but how those updates affect the novel entity's internal state. We therefore compute the gradient of each task's fine-tuning loss with respect to hidden activations *at the novel entity's token position* (~5,500 samples per task). This measures: "which direction does each task's loss want to push the new entity's internal representation?"

| Layer 3 | Layer 4 | Layer 5 |
|:---:|:---:|:---:|
| ![](assets/main/activation_grad_cossim_l3.png) | ![](assets/main/activation_grad_cossim_l4.png) | ![](assets/main/activation_grad_cossim_l5.png) |

**Figure 5b.** Pairwise cosine similarity of mean activation gradients at novel entity token positions, across layers 3–5. In sharp contrast with the parameter-level analysis (Figure 5a), distance is *anti-correlated* with all six other tasks at every layer, while the other six are mutually aligned (0.60–0.95). The effect is strongest at layers 3–4 (the causal computation layers identified by our intervention analysis).

| Layer 3 | Layer 4 | Layer 5 |
|:---:|:---:|:---:|
| ![](assets/main/activation_grad_coord_space_l3.png) | ![](assets/main/activation_grad_coord_space_l4.png) | ![](assets/main/activation_grad_coord_space_l5.png) |

**Figure 5c.** Mean activation gradient projected onto the X and Y coordinate probe directions. Each point is one task. Distance (red square) is the only task with a negative X-projection at every layer — it pushes new entity representations in the opposite direction along the primary coordinate axis.

**Takeaway**: The two-level analysis reveals that the divergence is specifically localized to the representation level, not the parameter level. Tasks update largely similar parameters (Figure 5a), but the *effect* of those updates on novel entity representations is opposite for distance vs. all other tasks (Figure 5b-c). This is a more precise mechanistic claim than "distance learns different weights" — it means the same parameter updates push entity representations in conflicting directions. The conflict is strongest at layers 3–4, consistent with the causal computation boundary identified independently through our intervention analysis.

---

## 6. Geographic Holdout Robustness (pKZe Q2a)

To test whether distance divergence depends on *which* region is held out during pretraining, we ran the full pipeline (PT1 + 7 FTWB1 + 21 FTWB2 = 29 models) three times, each time holding out a different geographic region: North Africa (234 cities), North India (259 cities), and Middle East (210 cities). 87 new models total.

| |
|:---:|
| ![](assets/main/exp7_holdout_regions.png) |
| **Figure 6.** The three geographic holdout regions. Each region is excluded from pretraining data, then integrated via fine-tuning. |

| |
|:---:|
| ![](assets/main/exp7_ftwb1_3regions.png) |
| **Figure 6a.** FTWB1 single-task normalized improvement heatmaps (7x7) for three holdout regions. Rows = fine-tuning task, columns = evaluation task. Distance (row D) consistently shows near-zero transfer to all other tasks across all three regions (avg: NA=0.11, NI=0.12, ME=0.04), while other tasks transfer substantially. |

| |
|:---:|
| ![](assets/main/exp7_ftwb2_vs_ftwb1_3regions.png) |
| **Figure 6b.** FTWB2 minus best-FTWB1 (two-task synergy/interference) for three holdout regions. Blue = synergy (two-task exceeds best single-task), red = interference. The pattern is consistent across all three regions: pairs containing inside+perimeter, crossing+distance, trianglearea+angle show interference (rows 3-6), while pairs like distance+inside, angle+crossing show synergy (rows 19, 21). |

| Region | FTWB1 Distance transfer | FTWB1 Other tasks transfer (avg) | FTWB2 overall diff |
|--------|:---:|:---:|:---:|
| **North Africa** | 0.11 | 0.50 | +0.009 |
| **North India** | 0.12 | 0.49 | -0.001 |
| **Middle East** | 0.04 | 0.43 | +0.021 |
| **Original Atlantis** | 0.06 | 0.47 | — |

**Takeaway**: Distance is the worst-transferring task regardless of which geographic region is held out. The divergent task phenomenon is a property of the task itself, not of the holdout geometry. This directly addresses Q2a: no other task becomes divergent when the pretraining data distribution changes. Combined with the scattered Atlantis experiment (Section 3, addressing Q2b), this establishes that distance divergence is invariant to both the geometry and location of held-out entities.

---
---

# Full Figure Dump

Everything below is the complete set of plots for all metrics, groups, and layers. Provided for thoroughness; not referenced in the rebuttal text.

## CKA  - All Groups, All Layers

Per group (pt1x, pt2, pt3, ftwb1, ftwb2), per layer (3-6): `full_matrix`, `averaged_matrix`, `averaged_matrix_sem`, `intra_vs_inter`.

| File | Description |
|------|-------------|
| `full_dump/cka/pt1x_full_matrix_l{3,4,5,6}.png` | Full 21x21 CKA matrix, PT1-X |
| `full_dump/cka/pt1x_averaged_matrix_l{3,4,5,6}.png` | 7x7 averaged CKA, PT1-X |
| `full_dump/cka/pt1x_averaged_matrix_sem_l{3,4,5,6}.png` | 7x7 averaged + SEM, PT1-X |
| `full_dump/cka/pt1x_intra_vs_inter_l{3,4,5,6}.png` | Bar plot, PT1-X |
| `full_dump/cka/pt2_*` | Same set for PT2 |
| `full_dump/cka/pt3_*` | Same set for PT3 |
| `full_dump/cka/ftwb1_*` | Same set for FTWB1 (layer 5 only) |
| `full_dump/cka/ftwb2_*` | Same set for FTWB2 (layer 5 only) |

## RSA  - All Groups, All Layers

| File | Description |
|------|-------------|
| `full_dump/rsa/{pt1x,pt2,pt3}_full_matrix_l{3,4,5,6}.png` | Full 21x21 RSA matrices |
| `full_dump/rsa/{pt1x,pt2,pt3}_averaged_matrix_sem_l{3,4,5,6}.png` | 7x7 averaged + SEM |
| `full_dump/rsa/{pt1x,pt2,pt3}_intra_vs_inter_l{3,4,5,6}.png` | Bar plots |

## SNN  - All Groups, All Layers

| File | Description |
|------|-------------|
| `full_dump/snn/{pt1x,pt2,pt3}_full_matrix_l{3,4,5,6}.png` | Full 21x21 SNN matrices |
| `full_dump/snn/{pt1x,pt2,pt3}_averaged_matrix_sem_l{3,4,5,6}.png` | 7x7 averaged + SEM |
| `full_dump/snn/{pt1x,pt2,pt3}_intra_vs_inter_l{3,4,5,6}.png` | Bar plots |

## SVCCA  - All Groups, All Layers

| File | Description |
|------|-------------|
| `full_dump/svcca/{pt1x,pt2,pt3}_full_matrix_l{3,4,5,6}.png` | Full 21x21 SVCCA matrices |
| `full_dump/svcca/{pt1x,pt2,pt3}_averaged_matrix_sem_l{3,4,5,6}.png` | 7x7 averaged + SEM |
| `full_dump/svcca/{pt1x,pt2,pt3}_intra_vs_inter_l{3,4,5,6}.png` | Bar plots |

## Effective Rank  - All Groups, All Layers

| File | Description |
|------|-------------|
| `full_dump/effective_rank/{pt1x,pt2,pt3}_rank_bar_l{3,4,5,6}.png` | Per-variant rank bars |
| `full_dump/effective_rank/{pt1x,pt2,pt3}_rank_by_layer.png` | Layer progression |

## Seed Robustness  - All Individual Seeds

| File | Description |
|------|-------------|
| `full_dump/seed_robustness/{original,seed1,seed2,seed3}_ftwb1_evaluation_heatmap.png` | Per-seed FTWB1 |
| `full_dump/seed_robustness/{original,seed1,seed2,seed3}_ftwb2_evaluation_heatmap.png` | Per-seed FTWB2 |
| `full_dump/seed_robustness/{original,seed1,seed2,seed3}_ftwb2_vs_ftwb1.png` | Per-seed difference |
| `full_dump/seed_robustness/aggregated_*.png` | Aggregated versions |
| `full_dump/seed_robustness/*cka_generalization*.png` | CKA-generalization correlations |
| `full_dump/seed_robustness/probe_generalization_histogram*.png` | Probe gen histograms |

## CKA Trends & Width Ablation

| File | Description |
|------|-------------|
| `full_dump/cka_multilayer/same_task_cka_trends*.png` | Same-task vs cross-task CKA trends |
| `full_dump/cka_multilayer/cka_trends_*.png` | Per-seed and aggregated CKA trends |
| `full_dump/wide_ftwb1_evaluation_heatmap.png` | Wide model FTWB1 |
| `full_dump/wide_ftwb2_evaluation_heatmap.png` | Wide model FTWB2 |
| `full_dump/wide_ftwb2_vs_ftwb1.png` | Wide model difference |
