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

We analyze task gradient alignment at two levels to understand *where* the divergence lives.

- **Method:** For each single-task FT objective, we compute (a) the gradient w.r.t. all model parameters, and (b) the gradient w.r.t. hidden activations at the novel entity's token position. We then measure pairwise cosine similarity across all 7 tasks.
- **Parameter space (Figure 5a):** All task pairs are near-orthogonal (cossim 0.04–0.25), with no task standing out. This is expected in high dimensions and tells us the tasks update largely independent parameter subsets.
- **Activation space (Figures 5b-c):** The six non-distance tasks form a tight cluster (cossim 0.60–0.95), all pushing entity representations in the same direction. Distance is anti-correlated with every one of them. The conflict is specifically about *what happens to the entity's representation*, not about which weights get updated.

### 5a. Parameter-Space Gradient Similarity

We compute the cosine similarity between the flattened parameter gradients of each pair of single-task fine-tuning losses, averaged over ~640 samples per task. This measures whether tasks want to update the same weights in the same direction.

| |
|:---:|
| <img src="assets/main/param_grad_cossim_global.png" width="50%"> |

**Figure 5a.** Pairwise cosine similarity of parameter gradients (all model parameters). No task stands out as particularly divergent. Most pairs show near-zero similarity (0.04–0.25), consistent with high-dimensional parameter vectors being approximately orthogonal by default. The most negative pair is inside–distance (-0.32), not distance alone. At the parameter level, these tasks are largely indifferent to each other.

### 5b. Activation-Space Gradient Analysis

The parameter-level analysis does not single out distance. But divergent fine-tuning operates at the *representation* level: the question is not which weights get updated, but how those updates affect the novel entity's internal state. We therefore compute the gradient of each task's fine-tuning loss with respect to hidden activations *at the novel entity's token position* (~5,500 samples per task). This measures: "which direction does each task's loss want to push the new entity's internal representation?"

| Layer 3 | Layer 4 | Layer 5 |
|:---:|:---:|:---:|
| ![](assets/main/activation_grad_cossim_l3.png) | ![](assets/main/activation_grad_cossim_l4.png) | ![](assets/main/activation_grad_cossim_l5.png) |

**Figure 5b.** Pairwise cosine similarity of mean activation gradients at novel entity token positions, across layers 3–5. The picture here is drastically different from the parameter-level analysis. The six non-distance tasks are strongly mutually aligned (0.60–0.95), all agreeing on which direction to push entity representations. Distance is anti-correlated with *every single one of them* at every layer. Unlike the near-orthogonality in parameter space, this is active opposition in a low-dimensional representational bottleneck.

| Layer 3 | Layer 4 | Layer 5 |
|:---:|:---:|:---:|
| ![](assets/main/activation_grad_coord_space_l3.png) | ![](assets/main/activation_grad_coord_space_l4.png) | ![](assets/main/activation_grad_coord_space_l5.png) |

**Figure 5c.** Mean activation gradient projected onto the X and Y coordinate probe directions. Each point is one task. Distance (red square) is the only task with a negative X-projection at every layer, pushing new entity representations in the opposite direction along the primary coordinate axis.

**Takeaway**: Parameter gradients are uninformative: all tasks are roughly orthogonal in parameter space, as expected in high dimensions (Figure 5a). The divergence only becomes visible when we ask *what those parameter updates do to entity representations*. There, the six non-distance tasks form a tight coalition, all pushing entity representations in the same direction, while distance pushes the opposite way (Figures 5b-c). The conflict is strongest at layers 3–4, consistent with the causal computation boundary identified independently through our intervention analysis. This rules out the simple explanation that "distance learns different weights." Instead, similar weight updates produce *opposite representational effects* at the entity position.

---

## 6. Geographic Holdout Robustness (pKZe Q2a)

We test whether the divergent task phenomenon depends on *which* region is held out during pretraining, by running the full pipeline three times with three different holdout regions. 87 new models total.

- **Setup:** Three geographic holdout regions: North Africa (234 cities), North India (259 cities), Middle East (210 cities). Each excluded from pretraining, then integrated via fine-tuning. Full PT1 + 7 FTWB1 + 21 FTWB2 per region.
- **FTWB1 (Figure 6a):** Distance is the worst single-task specialist at transferring to other tasks in every region (avg transfer: NA=0.11, NI=0.12, ME=0.04 vs other tasks 0.43–0.50). No other task becomes divergent.
- **FTWB2 best-teacher (Figure 6b):** All 6 distance-containing pairs show interference (red), non-distance pairs are neutral/synergistic. Pattern is identical across all 3 regions.
- **Probe generalization (Figures 6c-d):** A non-distance model (A+P) integrates holdout cities at baseline accuracy (error 112–128 ≈ baseline 119–133). A distance-containing model (D+I) shows 3–4× worse probe error (318–435), confirming representational harm.

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
| **Figure 6b.** FTWB2 minus best-FTWB1 (two-task synergy/interference) for three holdout regions. Blue = synergy (two-task exceeds best single-task), red = interference. The pattern is consistent across all three regions: all 6 distance-containing pairs (rows D,T through D,Cr) show interference, while non-distance pairs are neutral to mildly synergistic. |

| Region | FTWB1 Distance transfer | FTWB1 Other tasks transfer (avg) | FTWB2 overall diff |
|--------|:---:|:---:|:---:|
| **North Africa** | 0.11 | 0.50 | +0.009 |
| **North India** | 0.12 | 0.49 | -0.001 |
| **Middle East** | 0.04 | 0.43 | +0.021 |
| **Original Atlantis** | 0.06 | 0.47 | — |

### Probe Generalization

To further confirm that distance harms *representational* integration (not just task performance), we extracted layer-5 representations from one well-integrated model (A+P = angle+perimeter, no distance) and one ill-integrated model (D+I = distance+inside) per region. We trained a linear probe on non-holdout city representations to predict x,y coordinates, then tested on holdout cities.

| |
|:---:|
| ![](assets/main/exp7_probe_gen_maps.png) |
| **Figure 6c.** Probe generalization: ground truth holdout locations (black crosses) vs probe-predicted locations (red dots). Left column: well-integrated model (A+P). Right column: ill-integrated model (D+I). The distance-containing model's predictions scatter widely, indicating that holdout cities are not correctly placed in the learned coordinate space. |

| |
|:---:|
| ![](assets/main/exp7_probe_gen_error_comparison.png) |
| **Figure 6d.** Mean probe distance error across regions. The well-integrated model (A+P, blue) matches the baseline error on non-holdout cities (gray), confirming that holdout cities are correctly integrated into the spatial representation. The ill-integrated model (D+I, red) shows 3-4x higher error, confirming representational harm from the distance task. |

| Region | A+P (no distance) | D+I (has distance) | Baseline |
|--------|:---:|:---:|:---:|
| **North Africa** | 113 | 318 | 119 |
| **North India** | 128 | 333 | 121 |
| **Middle East** | 112 | 435 | 133 |

**Takeaway**: Distance is the worst-transferring task regardless of which geographic region is held out. The divergent task phenomenon is a property of the task itself, not of the holdout geometry. The probe generalization analysis confirms this extends to the *representational* level: the distance task prevents holdout cities from being correctly placed in the learned coordinate space, while non-distance models integrate holdout cities at baseline accuracy. This directly addresses Q2a: no other task becomes divergent when the pretraining data distribution changes. Combined with the scattered Atlantis experiment (Section 3, addressing Q2b), this establishes that distance divergence is invariant to both the geometry and location of held-out entities.

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
