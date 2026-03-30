# Supplementary Materials

New experiments and analyses addressing reviewer concerns. Figures referenced in the rebuttal text responses.

**Setup**: Qwen2.5-inspired Transformer (128 hidden, 512 intermediate, 4 heads, 6 layers), character-level tokenizer (45 tokens), trained on 7 geometric tasks over 5,075 real-world city coordinates.

---

## 1. Representation Similarity — Four Metrics (pKZe Q3, rQVL W3)

We computed CKA, RSA, SNN (k=10), and SVCCA across all pairwise comparisons for PT1-X (single-task, 7 tasks x 3 seeds), PT2 (two-task, 7 pairs x 3 seeds), and PT3 (three-task, 7 triples x 3 seeds). 210 pairs per group x 4 layers x 4 metrics = **10,080 total computations**. Each column below shows the 7x7 task-averaged similarity matrix (with SEM) at layer 5, progressing from 1-task to 2-task to 3-task training.

| | PT1-X (1 task) | PT2 (2 tasks) | PT3 (3 tasks) |
|---|:---:|:---:|:---:|
| **CKA** | ![](assets/main/cka_pt1x_l5.png) | ![](assets/main/cka_pt2_l5.png) | ![](assets/main/cka_pt3_l5.png) |
| **RSA** | ![](assets/main/rsa_pt1x_l5.png) | ![](assets/main/rsa_pt2_l5.png) | ![](assets/main/rsa_pt3_l5.png) |
| **SNN** | ![](assets/main/snn_pt1x_l5.png) | ![](assets/main/snn_pt2_l5.png) | ![](assets/main/snn_pt3_l5.png) |
| **SVCCA** | ![](assets/main/svcca_pt1x_l5.png) | ![](assets/main/svcca_pt2_l5.png) | ![](assets/main/svcca_pt3_l5.png) |

**Table 1.** Intra-task vs inter-task similarity at layer 5 (Welch's t-test).

| Metric | | PT1-X (1 task) | PT2 (2 tasks) | PT3 (3 tasks) |
|--------|---|:---:|:---:|:---:|
| **CKA** | Intra | 0.736 (p=0.001\*\*) | 0.912 (p=0.022\*) | 0.906 (p=0.013\*) |
| | Inter | 0.501 | 0.883 | 0.865 |
| **RSA** | Intra | 0.719 (p=0.002\*\*) | 0.917 (p=0.226) | 0.923 (p=0.065) |
| | Inter | 0.467 | 0.895 | 0.891 |
| **SNN** | Intra | 0.176 (p=0.005\*\*) | 0.211 (p=0.155) | 0.218 (p=0.373) |
| | Inter | 0.101 | 0.190 | 0.208 |
| **SVCCA** | Intra | 0.609 (p=0.366) | 0.612 (p=0.073) | 0.742 (p=0.459) |
| | Inter | 0.551 | 0.703 | 0.777 |

**Takeaway**: All four metrics confirm representation convergence (1-task → 2-task → 3-task). CKA, RSA, and SNN significantly distinguish intra from inter for PT1-X (p < 0.01); by PT2/PT3, intra ≈ inter — representations converge regardless of task identity. SNN (the non-global metric emphasized by the PRH paper) supports convergence strongly. SVCCA shows subspaces align even across isolated tasks; it is the finer geometry that differs.

---

## 2. Effective Rank (pKZe "representation rank", rQVL)

Participation ratio of singular values for all 63 experiments, layers 3–6.

| | |
|:---:|:---:|
| ![](assets/main/rank_pt1x_l5.png) | ![](assets/main/rank_pt1x_by_layer.png) |
| **Figure 2a.** Per-task effective rank at layer 5. Successful tasks compress to ~13–20 dims (of 256). Crossing (161) and inside (118) fail to compress. | **Figure 2b.** Layer progression. Most tasks: ~100 at L3 → ~15–20 by L5. Crossing flat at ~160. |

**Takeaway**: Effective rank provides a quantitative characterization of representation geometry. The dramatic rank gap between tasks that learn spatial structure (13–20) and tasks that don't (118–161) offers a simple diagnostic complementing PCA inspection.

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

**Takeaway**: The divergent task effect is not an artifact of Atlantis clustering. It holds — and is even more significant — when new entities are uniformly distributed. Figure 3b shows this visually: without the distance task, predicted Atlantis locations cluster near their true positions; with the distance task, predictions scatter widely.

---

## 4. Width Ablation (CBid W3)

2x wider model (256 hidden, 1024 intermediate, 8 heads) at constant compute (half epochs).

| |
|:---:|
| ![](assets/main/width_ablation_diff.png) |
| **Figure 4.** FTWB2 - FTWB1 difference for the wide model. Same qualitative pattern as the default architecture — distance remains divergent. |

**Takeaway**: Doubling model capacity does not resolve the divergent task phenomenon. This is a property of the task–representation interaction, not a capacity limitation.

---

## 5. Seed Robustness (all reviewers)

All experiments replicated across 4 random seeds.

| | |
|:---:|:---:|
| ![](assets/main/seed_ftwb2_aggregated.png) | ![](assets/main/probe_gen_histogram.png) |
| **Figure 5a.** FTWB2 performance aggregated across 4 seeds. Trained=0.897, Transfer=0.601. | **Figure 5b.** Probe generalization. Distance causes ~5x worse OOD error. Training with Atlantis (Exp5) eliminates the effect. |

**Takeaway**: All phenomena are robust across seeds. The aggregated 4-seed results strengthen every claim from the original single-seed analysis.

---
---

# Full Figure Dump

Everything below is the complete set of plots for all metrics, groups, and layers. Provided for thoroughness; not referenced in the rebuttal text.

## CKA — All Groups, All Layers

Per group (pt1x, pt2, pt3, ftwb1, ftwb2), per layer (3–6): `full_matrix`, `averaged_matrix`, `averaged_matrix_sem`, `intra_vs_inter`.

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

## RSA — All Groups, All Layers

| File | Description |
|------|-------------|
| `full_dump/rsa/{pt1x,pt2,pt3}_full_matrix_l{3,4,5,6}.png` | Full 21x21 RSA matrices |
| `full_dump/rsa/{pt1x,pt2,pt3}_averaged_matrix_sem_l{3,4,5,6}.png` | 7x7 averaged + SEM |
| `full_dump/rsa/{pt1x,pt2,pt3}_intra_vs_inter_l{3,4,5,6}.png` | Bar plots |

## SNN — All Groups, All Layers

| File | Description |
|------|-------------|
| `full_dump/snn/{pt1x,pt2,pt3}_full_matrix_l{3,4,5,6}.png` | Full 21x21 SNN matrices |
| `full_dump/snn/{pt1x,pt2,pt3}_averaged_matrix_sem_l{3,4,5,6}.png` | 7x7 averaged + SEM |
| `full_dump/snn/{pt1x,pt2,pt3}_intra_vs_inter_l{3,4,5,6}.png` | Bar plots |

## SVCCA — All Groups, All Layers

| File | Description |
|------|-------------|
| `full_dump/svcca/{pt1x,pt2,pt3}_full_matrix_l{3,4,5,6}.png` | Full 21x21 SVCCA matrices |
| `full_dump/svcca/{pt1x,pt2,pt3}_averaged_matrix_sem_l{3,4,5,6}.png` | 7x7 averaged + SEM |
| `full_dump/svcca/{pt1x,pt2,pt3}_intra_vs_inter_l{3,4,5,6}.png` | Bar plots |

## Effective Rank — All Groups, All Layers

| File | Description |
|------|-------------|
| `full_dump/effective_rank/{pt1x,pt2,pt3}_rank_bar_l{3,4,5,6}.png` | Per-variant rank bars |
| `full_dump/effective_rank/{pt1x,pt2,pt3}_rank_by_layer.png` | Layer progression |

## Seed Robustness — All Individual Seeds

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
