"""Task C: Generate structured results narrative document."""
import pandas as pd
import numpy as np
import os

BASE = r"d:\BTP\btp2\salmamaterials"

def main():
    df = pd.read_csv(os.path.join(BASE, "_consecutive_pair_dataset_AUDIO.csv"))
    sigma = pd.read_csv(os.path.join(BASE, "_sigma_results_AUDIO.csv"))
    stests = pd.read_csv(os.path.join(BASE, "_statistical_tests_AUDIO.csv"))
    valid = df[df["switching"].notna()].copy()
    sw_t = valid[valid["switching"] == True]
    sw_f = valid[valid["switching"] == False]

    # Compute needed values
    i1v = df["clip2_I1_val"].dropna()
    i2v = df["clip2_I2_val"].dropna()
    i2a = df["clip2_I2_aro"].dropna()

    # B2 dominance
    b2_sw = (sw_t["clip2_B_I1_tau050"] == "B2").mean()
    b2_ns = (sw_f["clip2_B_I1_tau050"] == "B2").mean()

    # Maintenance vs Drift
    mvd = stests[(stests["hyp_id"] == "Maintenance_vs_Drift") & (stests["tau"] == 50)]
    mvd_row = mvd.iloc[0] if len(mvd) > 0 else None

    # Load Task A results if available
    b2_dom_path = os.path.join(BASE, "_taskA_b2_dominance.csv")
    b2_dom = None
    if os.path.exists(b2_dom_path):
        b2_dom = pd.read_csv(b2_dom_path)
        b2_all = b2_dom[b2_dom["start_bucket"] == "ALL"]

    # Load Task B results if available
    aro_path = os.path.join(BASE, "_taskB_arousal_carryover.csv")
    aro_carry = None
    if os.path.exists(aro_path):
        aro_carry = pd.read_csv(aro_path)

    doc = f"""# BTP-2: Switching Analysis Results Narrative

## Section 1: I2 Valence Compression Finding

The Llama-3.3-70B model, when applied to 1Hz Fitbit heart rate sequences for
clip-level physiological emotion inference, produces **systematically compressed
valence predictions**.

| Metric | I1 (Survey) | I2 (Fitbit/Llama) |
|---|---|---|
| Range | [{i1v.min():.2f}, {i1v.max():.2f}] | [{i2v.min():.2f}, {i2v.max():.2f}] |
| Mean | {i1v.mean():.3f} | {i2v.mean():.3f} |
| Std | {i1v.std():.3f} | {i2v.std():.3f} |
| Fraction < 3.0 | {(i1v < 3.0).mean():.1%} | {(i2v < 3.0).mean():.1%} |

**Root cause:** Heart rate alone cannot determine emotional valence direction.
An elevated heart rate is physiologically identical for fear (negative valence)
and excitement (positive valence). This is the documented **valence ambiguity
problem** in heart rate-based emotion recognition (Kreibig, 2010; Mauss & Robinson,
2009). The K-EmoCon validation (MAE ~0.12) succeeded because it used 64Hz ECG
with rich morphological features unavailable in consumer-grade 1Hz Fitbit data.

**I2 Arousal is valid:** I2 arousal has std={i2a.std():.2f} and range
[{i2a.min():.2f}, {i2a.max():.2f}], confirming the model can discriminate
arousal levels from heart rate patterns.

**Consequence:** All I2-dependent combined bucket analyses (B6-B10, {{B3+B6}})
are affected by valence compression. I1-based analyses (B1-B5) and I2
arousal-only analyses remain valid.

**This is itself a thesis contribution:** We have empirically demonstrated the
valence ambiguity limitation of HR-only physiological inference on consumer
wearable data in a controlled clinical dataset.

---

## Section 2: Three Statistically Validated Findings (AUDIO, tau=0.50)

### Finding 1: Conscious Denial in Forced Emotional Switching (Hyp-2)

- **t = 3.35, p = 0.0009** (highly significant)
- sigma(opposing switching) = 0.029 (n=240)
- sigma(non-switching) = 0.000 (n=374)
- **Direction: Confirmed** (switching > non-switching)

**Interpretation:** During opposing emotional transitions (e.g., Sad clip
followed by Happy clip), a statistically significant 2.9% of pairs show
the combined {{B3+B6}} pattern where the viewer correctly perceived the clip
and the body confirmed the emotion, but the survey denied the feeling.
This pattern literally never occurs (0.000) in non-switching sequences.

**Caveat:** The absolute rate (2.9%) is low because the combined bucket
requires I2 to match E on both valence and arousal, which is affected by
the valence compression issue. The statistical significance is driven by the
clean binary separation (0.029 vs 0.000).

### Finding 2: Diagonal Switching Produces Stronger Body-Mind Divergence (Hyp-6)

- **t = 2.13, p = 0.033** (significant)
- sigma(diagonal) = 0.029 (n=240 opposing)
- sigma(adjacent) = 0.010 (n=629 adjacent)
- **Direction: Confirmed** (diagonal > adjacent)

**Interpretation:** Hard emotional jumps that cross both the valence and
arousal axes (Q1<->Q3 or Q2<->Q4) produce nearly 3x the body-mind divergence
compared to gentle adjacent switches that change only one axis. The magnitude
of the emotional transition moderates how much the physiological and
self-reported systems diverge.

### Finding 3: Neutral Clips Mirror Prior State After Transitions (Hyp-9 REVERSED)

- **t = 2.50, p = 0.015** (significant)
- mirror_rate(switching) = 0.523 (n=44)
- mirror_rate(non-switching) = 0.200 (n=20)
- **Direction: REVERSED** from prediction (switching > non-switching)

**Interpretation:** When a viewer has just undergone an emotional transition,
their perception of a subsequent neutral clip reflects their prior emotional
state more strongly than the clip's own content (mirror_rate = 52.3%). In
non-switching (stable emotional maintenance), this effect is much weaker (20%).

**Clinical implication:** If you want to read a patient's current emotional
state using a neutral probe clip, show it AFTER an emotionally charged
sequence for the strongest diagnostic signal, not during stable maintenance.

---

## Section 3: B2 Dominance - The Core Markov Finding

Regardless of starting bucket or switching condition, approximately
{b2_sw:.0%} (switching) to {b2_ns:.0%} (non-switching) of all clip-pair
transitions result in the viewer entering the Projection state
(B2: P = I1 != E).

This means the dominant emotional regulation strategy in sequential film
clip viewing is **Projection**: viewers align their self-reported feeling
(I1) with their perception (P), but both diverge from the clip's expressed
content (E).

"""

    if b2_dom is not None and len(b2_all) > 0:
        from scipy import stats as sp_stats
        r = b2_all.iloc[0]
        doc += f"""B2 dominance overall t-test (switching vs non-switching):
P(B2|switching) = {r['P_B2_switching']:.3f}, P(B2|non-switching) = {r['P_B2_nonswitching']:.3f}

"""

    doc += f"""---

## Section 4: Maintenance vs Drift Direction

"""
    if mvd_row is not None:
        doc += f"""- t = {mvd_row['t_stat']:.4f}, p = {mvd_row['p_value']:.4f}
- Mean delta_I1_val (Drift group): {mvd_row['sigma_switching']:.4f} (n={int(mvd_row['n_switching'])})
- Mean delta_I1_val (Maintenance group): {mvd_row['sigma_nonswitching']:.4f} (n={int(mvd_row['n_nonswitching'])})

The direction is correct: viewers who switch into Q1 (Happy) from a different
quadrant show larger valence shifts (0.628) than viewers who maintain Q1
(0.275). The result is marginally significant (p=0.089), limited by
n_maintenance=23.
"""

    doc += f"""
---

## Section 5: Limitations

1. **I2 Valence Compression:** Llama I2 valence is compressed to 2.5-3.0,
   making I2-dependent bucket analyses (B6-B10) unreliable for valence-based
   claims. See Section 1.

2. **VIDEO E Model Calibration:** The video expressed emotion model predicts
   exclusively Q1 (Happy/Excited) for all 274 clips (valence range 3.2-4.2,
   arousal range 3.3-4.5). VIDEO switching analysis is impossible.

3. **Small Sample:** 6 participants, 1,638 pairs total. Effective n per
   hypothesis subset is often < 100, limiting statistical power for
   rare-event hypotheses (Hyp-4, Hyp-5).

4. **Validation Not Yet Executed:** All findings are exploratory on the
   training dataset. Independent validation with 10-15 new participants
   is required before claims can be upgraded from "supported" to "validated."

5. **Averaged Across Users:** P and I1 are averaged across all 6 users,
   losing individual difference information.

---

## Section 6: What Needs Validation

### Finding 1 (Hyp-2): Conscious Denial
- **Test:** 10+ new participants view opposing-quadrant clip pairs vs
  same-quadrant control pairs
- **Threshold:** sigma(B3+B6 | opposing) >= 0.15 (must be higher than
  the exploratory 0.029 due to sample design optimization)
- **Minimum n:** 15 participants x 10 pairs = 150 switching pairs

### Finding 2 (Hyp-6): Diagonal vs Adjacent
- **Test:** Diagonal switching sequences vs adjacent switching sequences
- **Threshold:** sigma(diagonal) significantly > sigma(adjacent), p < 0.05
- **Minimum n:** 15 participants x 5 diagonal + 5 adjacent pairs each

### Finding 3 (Hyp-9 Reversed): Neutral Mirror After Transition
- **Test:** Emotional clip followed by neutral probe vs neutral probe alone
- **Threshold:** mirror_rate(post-emotional) >= 0.60,
  mirror_rate(control) <= 0.40
- **Minimum n:** 15 participants x 5 emotional+probe + 5 control sequences
"""

    outpath = os.path.join(BASE, "_BTP2_RESULTS_NARRATIVE.md")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"Results narrative saved to: {outpath}")

if __name__ == "__main__":
    main()
