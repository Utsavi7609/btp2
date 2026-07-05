# BTP-2: Validation Phase Design

## Overview

This document specifies the experimental design to validate the three
statistically supported findings from the AUDIO switching analysis.
All designs use the same 274 movie clips shown in randomized sequences
to 10-15 new participants.

---

## Design 1: Validating Conscious Denial (Hyp-2)

**Finding to validate:** Opposing-quadrant emotional transitions produce
the combined {B3+B6} pattern (correct perception + body confirmation +
survey denial) at a significantly higher rate than same-quadrant transitions.

### Existing Evidence Clips

                  clip1_title                   clip2_title clip1_E_quad clip2_E_quad       user_id
   TheFaultInOurStars_016.mp4 TheTheoryOfEverything_041.mp4           Q1           Q3       Nishant
TheTheoryOfEverything_034.mp4          TheBlindSide_065.mp4           Q4           Q2       Nishant
   TheFaultInOurStars_036.mp4 TheTheoryOfEverything_041.mp4           Q1           Q3    debjit2001
             LadyBird_003.mp4             AboutTime_013.mp4           Q2           Q4    pranjal123
         TheBlindSide_080.mp4 TheTheoryOfEverything_041.mp4           Q1           Q3 sayantankuila
         TheBlindSide_081.mp4             AboutTime_013.mp4           Q2           Q4 sayantankuila
   TheFaultInOurStars_043.mp4          TheBlindSide_065.mp4           Q4           Q2      ajaycc17

Total opposing switching B3+B6 pairs in AUDIO data: 7

### Control Clips (Q1->Q1, B1)

               clip1_title                clip2_title    user_id
         AboutTime_005.mp4          AboutTime_052.mp4    Nishant
      TheBlindSide_054.mp4       TheBlindSide_092.mp4    Nishant
      TheBlindSide_069.mp4 TheFaultInOurStars_036.mp4    Nishant
TheFaultInOurStars_018.mp4       TheBlindSide_005.mp4    Nishant
      TheBlindSide_091.mp4       TheBlindSide_059.mp4    Nishant
TheFaultInOurStars_016.mp4       TheBlindSide_092.mp4 debjit2001
          LadyBird_104.mp4 TheFaultInOurStars_036.mp4 debjit2001
TheFaultInOurStars_039.mp4       TheBlindSide_059.mp4 debjit2001
      TheBlindSide_038.mp4       TheBlindSide_080.mp4 debjit2001
      TheBlindSide_089.mp4       TheBlindSide_005.mp4 pranjal123

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

                  clip1_title                clip2_title clip1_E_quad clip2_E_quad user_id
            AboutTime_069.mp4 TheFaultInOurStars_069.mp4           Q1           Q3 Nishant
         TheBlindSide_028.mp4       TheBlindSide_061.mp4           Q3           Q1 Nishant
               Gifted_041.mp4           LadyBird_011.mp4           Q4           Q2 Nishant
         TheBlindSide_045.mp4       TheBlindSide_082.mp4           Q4           Q2 Nishant
         TheBlindSide_075.mp4          AboutTime_005.mp4           Q3           Q1 Nishant
   TheFaultInOurStars_000.mp4          AboutTime_035.mp4           Q4           Q2 Nishant
            AboutTime_035.mp4       TheBlindSide_033.mp4           Q2           Q4 Nishant
    TheSpectacularNow_068.mp4       TheBlindSide_056.mp4           Q3           Q1 Nishant
   TheFaultInOurStars_027.mp4           LadyBird_013.mp4           Q4           Q2 Nishant
TheTheoryOfEverything_037.mp4 TheFaultInOurStars_072.mp4           Q2           Q4 Nishant

Total diagonal pairs: 240

### Adjacent Transition Clips (sample)

                  clip1_title                   clip2_title clip1_E_quad clip2_E_quad user_id
   TheFaultInOurStars_069.mp4    TheFaultInOurStars_064.mp4           Q3           Q4 Nishant
   TheFaultInOurStars_046.mp4    TheFaultInOurStars_045.mp4           Q4           Q3 Nishant
         TheBlindSide_014.mp4             AboutTime_013.mp4           Q3           Q4 Nishant
TheTheoryOfEverything_042.mp4          TheBlindSide_021.mp4           Q3           Q4 Nishant
   TheFaultInOurStars_031.mp4          TheBlindSide_085.mp4           Q4           Q3 Nishant
         TheBlindSide_085.mp4             AboutTime_019.mp4           Q3           Q4 Nishant
TheTheoryOfEverything_030.mp4          TheBlindSide_028.mp4           Q4           Q3 Nishant
             LadyBird_011.mp4             AboutTime_022.mp4           Q2           Q1 Nishant
   TheFaultInOurStars_058.mp4             AboutTime_046.mp4           Q4           Q1 Nishant
            AboutTime_046.mp4 TheTheoryOfEverything_026.mp4           Q1           Q4 Nishant

Total adjacent pairs: 629

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

                  clip1_title                   clip2_title clip1_E_quad clip2_E_quad    user_id
TheTheoryOfEverything_041.mp4 TheTheoryOfEverything_031.mp4           Q3           Q1    Nishant
            AboutTime_038.mp4              LadyBird_056.mp4           Q2           Q1    Nishant
            AboutTime_011.mp4          TheBlindSide_003.mp4           Q1           Q3    Nishant
   TheFaultInOurStars_051.mp4          TheBlindSide_042.mp4           Q1           Q4    Nishant
         TheBlindSide_010.mp4          TheBlindSide_057.mp4           Q4           Q2    Nishant
         TheBlindSide_057.mp4    TheFaultInOurStars_073.mp4           Q2           Q4    Nishant
            AboutTime_068.mp4    TheFaultInOurStars_073.mp4           Q1           Q4 debjit2001
         TheBlindSide_035.mp4          TheBlindSide_071.mp4           Q4           Q2 debjit2001
         TheBlindSide_071.mp4          TheBlindSide_009.mp4           Q2           Q4 debjit2001
            AboutTime_071.mp4          TheBlindSide_057.mp4           Q1           Q2 debjit2001

Total emotional->neutral switching pairs: 44

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
