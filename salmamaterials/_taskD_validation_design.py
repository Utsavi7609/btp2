"""Task D: Validation phase design document."""
import pandas as pd
import numpy as np
import os

BASE = r"d:\BTP\btp2\salmamaterials"
OPPOSING = {("Q1","Q3"),("Q3","Q1"),("Q2","Q4"),("Q4","Q2")}
ADJACENT = {("Q1","Q2"),("Q2","Q1"),("Q1","Q4"),("Q4","Q1"),
            ("Q2","Q3"),("Q3","Q2"),("Q3","Q4"),("Q4","Q3")}

def main():
    df = pd.read_csv(os.path.join(BASE, "_consecutive_pair_dataset_AUDIO.csv"))
    valid = df[df["switching"].notna()].copy()
    sw_t = valid[valid["switching"] == True]

    # Find clips for Design 1: opposing switching pairs in B3+B6
    opp = sw_t[sw_t.apply(lambda r: (r["clip1_E_quad"], r["clip2_E_quad"]) in OPPOSING, axis=1)]
    b3b6 = opp[opp["clip2_combined_tau050"] == "B3+B6"]

    # Find Q1->Q1 non-switching B1 control pairs
    ns = valid[valid["switching"] == False]
    ctrl_b1 = ns[(ns["clip1_E_quad"] == "Q1") & (ns["clip2_E_quad"] == "Q1") &
                 (ns["clip2_B_I1_tau050"] == "B1")]

    # Design 2: diagonal vs adjacent
    diag_pairs = sw_t[sw_t.apply(lambda r: (r["clip1_E_quad"], r["clip2_E_quad"]) in OPPOSING, axis=1)]
    adj_pairs = sw_t[sw_t.apply(lambda r: (r["clip1_E_quad"], r["clip2_E_quad"]) in ADJACENT, axis=1)]

    # Design 3: neutral clips after emotional clip (relaxed neutral definition)
    neut = df[(df["clip2_E_val"] - 3.0).abs().le(0.5) & (df["clip2_E_aro"] - 3.0).abs().le(0.5)]
    neut_b2 = neut[neut["clip2_B_I1_tau050"] == "B2"]
    neut_sw = neut_b2[neut_b2["switching"] == True]

    doc = f"""# BTP-2: Validation Phase Design

## Overview

This document specifies the experimental design to validate the three
statistically supported findings from the AUDIO switching analysis.
All designs use the same 274 movie clips shown in randomized sequences
to 10-15 new participants.

---

## Design 1: Validating Conscious Denial (Hyp-2)

**Finding to validate:** Opposing-quadrant emotional transitions produce
the combined {{B3+B6}} pattern (correct perception + body confirmation +
survey denial) at a significantly higher rate than same-quadrant transitions.

### Existing Evidence Clips

{b3b6[["clip1_title", "clip2_title", "clip1_E_quad", "clip2_E_quad", "user_id"]].head(15).to_string(index=False) if len(b3b6) > 0 else "No B3+B6 opposing pairs found."}

Total opposing switching B3+B6 pairs in AUDIO data: {len(b3b6)}

### Control Clips (Q1->Q1, B1)

{ctrl_b1[["clip1_title", "clip2_title", "user_id"]].head(10).to_string(index=False) if len(ctrl_b1) > 0 else "No control B1 pairs found."}

### Protocol
1. **Block A (Test):** 5 pairs of opposing-quadrant clips per participant
   - Select pairs from the opposing list above
   - Prioritize Q3->Q1 and Q1->Q3 transitions
2. **Block B (Control):** 5 pairs of same-quadrant clips (Q1->Q1)
3. **Measurement:** After each Clip 2, collect P survey + I1 survey
4. **Success criteria:**
   - sigma(B3+B6 | Block A) >= 0.15
   - sigma(B3+B6 | Block B) <= 0.05
   - p < 0.05 (Fisher exact test or chi-squared)
5. **Minimum participants:** 15 (producing 75 test pairs + 75 control pairs)

---

## Design 2: Validating Diagonal vs Adjacent Switching (Hyp-6)

**Finding to validate:** Diagonal switches (Q1<->Q3, Q2<->Q4) produce
stronger body-mind divergence (B3+B6 rate) than adjacent switches.

### Diagonal Transition Clips (sample)

{diag_pairs[["clip1_title", "clip2_title", "clip1_E_quad", "clip2_E_quad", "user_id"]].drop_duplicates(["clip1_title", "clip2_title"]).head(10).to_string(index=False) if len(diag_pairs) > 0 else "No diagonal pairs found."}

Total diagonal pairs: {len(diag_pairs)}

### Adjacent Transition Clips (sample)

{adj_pairs[["clip1_title", "clip2_title", "clip1_E_quad", "clip2_E_quad", "user_id"]].drop_duplicates(["clip1_title", "clip2_title"]).head(10).to_string(index=False) if len(adj_pairs) > 0 else "No adjacent pairs found."}

Total adjacent pairs: {len(adj_pairs)}

### Protocol
1. **Block A (Diagonal):** 5 diagonal switching pairs per participant
2. **Block B (Adjacent):** 5 adjacent switching pairs per participant
3. **Measurement:** P + I1 surveys after each Clip 2
4. **Success criteria:**
   - sigma(B3+B6 | diagonal) significantly > sigma(B3+B6 | adjacent)
   - p < 0.05 (t-test or Fisher exact)
5. **Minimum participants:** 15

---

## Design 3: Validating Neutral Mirror Effect (Hyp-9 Reversed)

**Finding to validate:** Neutral clips reflect the viewer's prior emotional
state more strongly after an emotional transition than during stable maintenance.

### Emotional-Then-Neutral Pairs (sample)

{neut_sw[["clip1_title", "clip2_title", "clip1_E_quad", "clip2_E_quad", "user_id"]].head(10).to_string(index=False) if len(neut_sw) > 0 else "No emotional->neutral switching pairs found."}

Total emotional->neutral switching pairs: {len(neut_sw)}

### Protocol
1. **Block A (Post-Emotional):** [Emotional clip from Q2/Q3] then [Neutral probe]
   - 5 sequences per participant
2. **Block B (Control):** [Neutral probe clip shown after another neutral clip]
   - 5 sequences per participant
3. **Measurement:** P and I1 surveys for the neutral probe clip ONLY
4. **Analysis:**
   - Compute d(P, prior_clip_I1) vs d(P, probe_E) for each neutral probe
   - mirror_rate = fraction where d(P, prior_I1) < d(P, probe_E)
5. **Success criteria:**
   - mirror_rate(Block A) >= 0.60
   - mirror_rate(Block B) <= 0.40
   - p < 0.05 (Fisher exact)
6. **Minimum participants:** 15

---

## General Requirements

| Field | Value |
|---|---|
| New participants | 10-15 (target: 15) |
| Clips per participant | ~30 (5 pairs per condition x 3 designs) |
| Session duration | ~45 minutes |
| Survey instrument | Same P and I1 questionnaire used in training data |
| Physiological | Optional: Fitbit for I2 (but not required for validation) |
| Randomization | Block order randomized across participants |
| Exclusion | Participants who skip >20% of surveys |
"""

    outpath = os.path.join(BASE, "_BTP2_VALIDATION_DESIGN.md")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"Validation design saved to: {outpath}")

if __name__ == "__main__":
    main()
