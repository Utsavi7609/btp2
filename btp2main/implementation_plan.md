# BTP-2: Switching Analysis Pipeline — Implementation Plan

## Overview

This plan covers the complete next phase after physiological inference (I2) is done.
The goal is to build a **consecutive clip-pair dataset**, assign the B1–B10 bucket labels, compute σ values for 10 hypotheses, run statistical tests, and generate all plots.

**New files must all start with `_` or `(`.**

---

## Data Sources (Verified)

| Variable | File | Column |
|---|---|---|
| E (Expressed) | [salmamaterials/final_expressed_valence_VIDEO.csv](file:///d:/BTP/btp2/salmamaterials/final_expressed_valence_VIDEO.csv) + [_AUDIO.csv](file:///d:/BTP/btp2/salmamaterials/final_expressed_valence_AUDIO.csv) | `clip_title` → valence/arousal |
| P (Perceived) | [salmamaterials/perceived_valence_summary.csv](file:///d:/BTP/btp2/salmamaterials/perceived_valence_summary.csv) + arousal | rows = clip_title |
| I1 (Self-Reported) | [salmamaterials/induced_valence_summary.csv](file:///d:/BTP/btp2/salmamaterials/induced_valence_summary.csv) + arousal | rows = clip_title |
| I2 (Fitbit-Inferred) | [salmamaterials/inferred_valence_summary_qwen_FS.csv](file:///d:/BTP/btp2/salmamaterials/inferred_valence_summary_qwen_FS.csv) + arousal | rows = clip_title |
| Watch order + raw E/P/I1 | [salmamaterials/BTP_Phase1/clip_responses_export.xlsx](file:///d:/BTP/btp2/salmamaterials/BTP_Phase1/clip_responses_export.xlsx) | `participant`, `clip_title`, `created_at`, `clip_valence/arousal` = E, `impact_valence/arousal` = P, `participant_valence/arousal` = I1 |

> [!IMPORTANT]
> [_chronological_clip_orders.csv](file:///d:/BTP/btp2/_chronological_clip_orders.csv) does NOT yet exist. It must be generated first.
> The 6 participants in the Excel are named: `Nishant`, `debjit2001`, `pranjal123`, and 3 others — **verify all unique participant names** when running Step 0.

---

## Step 0: Generate [_chronological_clip_orders.csv](file:///d:/BTP/btp2/_chronological_clip_orders.csv)

**Script:** `salmamaterials/_step0_build_watch_order.py`

- Read [BTP_Phase1/clip_responses_export.xlsx](file:///d:/BTP/btp2/salmamaterials/BTP_Phase1/clip_responses_export.xlsx)
- Group by `participant`
- Sort each participant's rows by `created_at` (timestamp)
- Assign `watch_position` = 1, 2, 3 ... (chronological order)
- Save as `salmamaterials/_chronological_clip_orders.csv`

**Output columns:** `user_id`, `clip_title`, `watch_position`

> [!NOTE]
> `clip_order` column already exists in the Excel but verify it matches the `created_at` sort — use `created_at` as the ground truth for ordering.

---

## Step 1: Build Consecutive Pair Dataset

**Script:** `salmamaterials/_step1_build_pair_dataset.py`

**Logic:**
- Load [_chronological_clip_orders.csv](file:///d:/BTP/btp2/_chronological_clip_orders.csv)
- For each user, sort clips by `watch_position`
- Slide window of size 2 → 273 pairs × 6 users = **1,638 rows**
- For each pair, look up E, P, I1, I2 values for both clips from their respective summary CSVs
- E: Use VIDEO source first; fall back to AUDIO if clip not in VIDEO file

**Quadrant Assignment** (midpoint = 3.0):
```
Q1: val > 3 AND aro > 3
Q2: val < 3 AND aro > 3
Q3: val < 3 AND aro < 3
Q4: val > 3 AND aro < 3
Q0: val == 3.0 OR aro == 3.0 (boundary — flag, exclude from quadrant analyses)
```

**Output columns:**
```
clip1_id, clip2_id, user_id,
clip1_E_val, clip1_E_aro, clip1_P_val, clip1_P_aro,
clip1_I1_val, clip1_I1_aro, clip1_I2_val, clip1_I2_aro,
clip2_E_val, clip2_E_aro, clip2_P_val, clip2_P_aro,
clip2_I1_val, clip2_I1_aro, clip2_I2_val, clip2_I2_aro,
clip1_E_quad, clip2_E_quad,
switching,  (True if clip1_E_quad != clip2_E_quad)
delta_I1_val, delta_I1_aro, delta_I2_val, delta_I2_aro
```

**Save:** `salmamaterials/_consecutive_pair_dataset.csv`

---

## Step 2: Assign B1–B10 Bucket Labels

**Script:** `salmamaterials/_step2_assign_buckets.py`

**Equality rule** at threshold τ: `|A_val − B_val| ≤ τ AND |A_aro − B_aro| ≤ τ`

Test at **τ ∈ {0.25, 0.5, 0.75, 1.0}**. Primary τ = 0.5.

**Bucket definitions** (compute separately for I1 and I2):
```
B1 / B6:  P == E AND P == I    (all three agree)
B2 / B7:  P == I AND P != E    (projection — feel what you perceive, not what clip says)
B3 / B8:  P == E AND P != I    (correct perception, wrong self-report)
B4 / B9:  E == I AND P != E    (body+clip agree, perception wrong)
B5 / B10: none of the above    (complete chaos)
```

**New columns to add** (for each clip, for each τ):
```
clip1_B_I1_tau025, clip1_B_I1_tau050, clip1_B_I1_tau075, clip1_B_I1_tau100
clip1_B_I2_tau025, clip1_B_I2_tau050, clip1_B_I2_tau075, clip1_B_I2_tau100
clip2_B_I1_tau025 ... clip2_B_I2_tau100
clip2_combined_tau050  (e.g. "B3+B6" if clip2 is B3 by I1 AND B6 by I2)
```

**Update:** `salmamaterials/_consecutive_pair_dataset.csv`

---

## Step 3: Compute σ for All 10 Hypotheses

**Script:** `salmamaterials/_step3_compute_sigma.py`

`σ = (rows satisfying hypothesis condition) / (total rows in filtered subset)`

Computed **separately** for `switching=True` and `switching=False`. At all 4 τ values.

| Hypothesis | Filter | Measure |
|---|---|---|
| Hyp-1 | clip2_B_I1 = B1 | σ, split by switching |
| Hyp-2 | switching=True, opposing E-quads | rate of clip2_combined = "B3+B6" |
| Hyp-3 | clip1_E_quad∈{Q1,Q3}, clip2 E near-neutral | σ(clip2_B_I1=B2), split by switching |
| Hyp-4 | clip1_B_I1=B5 OR clip1_B_I2=B10 | σ(clip2 in aligned bucket) vs B1-Clip1 baseline |
| Hyp-5 | all pairs | σ(clip2_B_I1=B4) and σ(clip2_B_I2=B9) — expect NO difference |
| Hyp-6 | switching pairs only | σ("B3+B6"), adjacent switch vs diagonal switch |
| Hyp-7 | all pairs | σ(clip2_B_I2=B9), switching vs non-switching |
| Hyp-8 | all pairs | Euclidean distance(I1,I2) for clip2, switching vs non-switching |
| Hyp-9 | clip2 near-neutral E, non-switching, clip2∈B2 | check if P₂ closer to I1₁ than E₂ |
| Hyp-10 | clip1_B_I2=B10 | mean I2_arousal of clip2, vs clip1_B_I2=B6 group |

**Save:** `salmamaterials/_sigma_results.csv`

Columns: `hyp_id, tau, sigma_switching, sigma_nonswitching, n_switching, n_nonswitching`

---

## Step 4: Statistical Tests

**Script:** `salmamaterials/_step4_statistical_tests.py`

- For each hypothesis: **independent samples t-test** on switching vs non-switching groups
- Flag significant if **p < 0.05**
- For Hyp-5: expect p > 0.05 (null retained)

**Save:** `salmamaterials/_statistical_tests.csv`

Columns: `hyp_id, tau, t_stat, p_value, significant, sigma_switching, sigma_nonswitching, direction_correct`

---

## Step 5: Maintenance vs Drift Proof

**Script:** (part of `_step4_statistical_tests.py` or standalone)

- Take all **B1(I1) clips in Q1** (happy)
- **Group A** = switching pairs (clip1_E_quad ≠ Q1) — these are drift attempts
- **Group B** = non-switching pairs (clip1_E_quad = Q1) — these are maintenance
- Compute `σ(B1 | Group A)` and `σ(B1 | Group B)` → expect A ≈ 0.30–0.40, B ≈ 0.80+
- Run t-test, generate box plot of `delta_I1_val` for A vs B

**Save:** `salmamaterials/_plots/_maintenance_vs_drift_proof.png`

---

## Step 6: Generate All Plots

**Script:** `salmamaterials/_step5_generate_plots.py`

Save all to `salmamaterials/_plots/`:

| Plot File | What It Shows |
|---|---|
| `_markov_heatmap_switching.png` | 5×5 B1–B5 bucket transition (switching pairs only) |
| `_markov_heatmap_nonswitching.png` | Same, non-switching pairs |
| `_quadrant_transition_heatmap_switching.png` | 4×4 clip1_E_quad → clip2_I2_quad (switching) |
| `_quadrant_transition_heatmap_nonswitching.png` | Same, non-switching |
| `_sigma_barchart.png` | σ per hypothesis, bars per τ, colour = switching/non-switching |
| `_sigma_sensitivity_curves.png` | σ vs τ line chart per hypothesis |
| `_drift_boxplot_valence.png` | delta_I1_val distribution: switching vs non-switching, per quadrant |
| `_drift_boxplot_arousal.png` | Same for arousal |
| `_I1_I2_scatter.png` | I1 val vs I2 val for clip2, coloured by B-bucket |
| `_I1_I2_distance_boxplot.png` | Euclidean I1–I2 distance per switching type |

---

## Execution Order

```
Step 0 → _step0_build_watch_order.py
Step 1 → _step1_build_pair_dataset.py
Step 2 → _step2_assign_buckets.py   (updates Step 1 output in-place)
Step 3 → _step3_compute_sigma.py
Step 4 → _step4_statistical_tests.py
Step 5 → _step5_generate_plots.py
```

Each step is independent and saves its own file — if one crashes, you can rerun just that step.

---

## I2 Data Integration Note

The I2 we use here is from [inferred_valence_summary_qwen_FS.csv](file:///d:/BTP/btp2/salmamaterials/inferred_valence_summary_qwen_FS.csv) (already in `salmamaterials/`).
This is the **K-EmoCon validated Qwen few-shot model** — not the new Llama Fitbit file.

> [!NOTE]
> The new [inferred_emotions_enhanced_llama.csv](file:///d:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/inferred_emotions_enhanced_llama.csv) file (Fitbit-based, 368 windows) is Ground Truth B for the **Affective Drift** calculation (comparing inferred vs self-reported). It is a separate analysis from the switching analysis which operates on the 274 K-EmoCon validated clips.

---

## Verification Plan

### Automated
- After Step 1: verify row count = 1,638 (or close, depending on missing clips)
- After Step 2: verify every row has a non-null B-bucket for all 4 τ values
- After Step 3: verify σ values are in [0, 1] for all hypotheses
- After Step 4: print full statistical test table and inspect direction of effects

### Manual
- Inspect `_markov_heatmap_switching.png` vs `_markov_heatmap_nonswitching.png` — the two should look meaningfully different in high-hypothesis-density cells
- Inspect `_maintenance_vs_drift_proof.png` — Group B box should be visually higher than Group A
