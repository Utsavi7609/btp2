# BTP-2 Final Presentation: Slide Content

**Time Limit:** 12 Minutes (Approx. 15 Slides)
**Faculty Guidelines followed:** Quick focus on objective, 1 lit-review slide, real hardware photos included, every results slide has observation + plot, low text density.

---

## Slide 1: Title Slide
*   **Title:** Targeted Emotion Regulation: Validating Affective Drift and Maintenance through Self-Reports, Physiological Signals, and LLMs
*   **Student:** Utsavi Gohil (22CS30058)
*   **Supervisor:** Professor Bivas Mitra

---

## Slide 2: The Core Problem (The "Big Picture")
*   **Background:** Media interventions (music/film) are widely prescribed to regulate emotion, but success is usually measured using subjective self-report surveys.
*   **Problem 1 (Affective Drift):** Humans are unreliable narrators of their own feelings. What the body physiologically experiences often diverges from what the user consciously reports (due to social desirability or memory decay).
*   **Problem 2 (Sequential Context):** A sad movie clip works differently if you watch it while you are currently *Happy* vs. currently *Sad*. Prior emotional momentum is almost always ignored in standard literature.

---

## Slide 3: Objectives & Key Contributions
*   **Core Objective:** Use consumer-grade physiological signals (Fitbit HR) and Large Language Models to mathematically prove the existence of complete mind-body divergence, and map how sequential emotional transitions alter human affect.
*   **Key Technical Contributions:**
    1. Built a clinically validated LLM biometric inference pipeline ($0.177$ MAE).
    2. Engineered a $-120s$ Autonomic Trailing Window pipeline to capture true physiological emotion ($I_2$).
    3. Analysed 1,638 clip-pairs to discover 3 statistically significant emotional switching phenomena.
    4. Proposed actionable clinical protocols for emotional "Maintenance" vs "Drift".

---

## Slide 4: Literature Review & Limitations
*   **The Mismatch:** Tian et al. (2017) and Hsu et al. (2022) proved self-reported perceptions and induced feelings often conflict. *Gap: No one has compared subjective surveys against objective physiological inferences (Body).*
*   **Physiology Limits (Kreibig, 2010):** Heart rate suffers from the "Valence Ambiguity" problem (elevated HR means both fear AND excitement). Latency is $\sim$20-30s.
*   **Wearable LLMs:** *Health-LLM* (Kim 2024) and *StressLLM* (Thapa 2025) prove LLMs can read wearables. *Our Novelty: Few-Shot In-Context learning on 1Hz consumer Fitbit sequences using clinical anchors.*

---

## Slide 5: Data Collection & Real Hardware
*   **Visual Check:** *(Insert `image1.png` - `image5.png` data collection screens side-by-side with `empatica.png` watch picture)*
*   **Participants:** 6 core participants (from 10 initially recruited)
*   **Dataset:** 1,495 robust clip-viewing instances combining subjective survey grids ($P$, $I_1$) with continuous 1Hz Fitbit HR metrics. 
*   **Stimuli:** 274 Hollywood movie clips spanning 1-3 minutes.

---

## Slide 6: Methodology Overview (The 8-Phase Pipeline)
*   **Visual Check:** *(Create a simple flow-chart box diagram in your PPT here)*
    1. Data Collection ($P$, $I_1$, HR)
    2. Audio/Video Expressed Emotion Extraction ($E$)
    3. Clinical Validation Mapping (K-EmoCon)
    4. Autonomic Trailing Extraction (-120s)
    5. Few-Shot LLM Execution ($I_2$)
    6. Multi-Threshold Bucket Generation
    7. Switching Pair Dataset Generation (1,638 pairs)
    8. Statistical Hypothesis Testing

---

## Slide 7: The "Shifted Window" (Autonomic Extraction)
*   **The Technique:** Because the autonomic nervous system has a massive lag, looking at heart rate *during* the clip is wrong. We shifted the extraction window backwards by $120$ seconds trailing the survey timestamp.
*   **Feature Engineering:** Injected rolling RMSSD using $\sqrt{\frac{1}{N-1}\sum (HR_{i+1} - HR_i)^2}$ to bypass the valence-ambiguity problem.
*   **Execution:** Passed raw 120s JSON sequences to `Llama-3.3-70B`. Achieved $96.4\%$ valid JSON retention across all participants.

---

## Slide 8: K-EmoCon Clinical Validation (Scientific Anchor)
*   **Visual Check:** *(Insert Table 5.2 showing 0.177 MAE \& 85.7% accuracy)*
*   **The Test:** Before trusting Fitbit, we ran our LLM pipeline on the ultra-clean 64Hz K-EmoCon medical ECG dataset (excluding sweat metrics to perfectly mimic Fitbit limitations).
*   **Observation:** The LLM successfully achieved a Valence MAE of $0.177$ (Pearson $r = 0.980$) and Arousal MAE of $0.548$.
*   **Inference:** This definitively proved our prompt pipeline turns LLMs into highly accurate "Biological Proxies".

---

## Slide 9: Result 1 — The "Explosion of Drift"
*   **Visual Check:** *(Show a side-by-side comparison of Table 5.3 vs Table 5.4 --- just the $\tau=0.50$ AUDIO row)* 
*   **Self-Report (Surveys):** Viewers reported only 12 clips failing into the "Drift/Denial" bucket. They claimed the clips didn't change their state. 
*   **Body Truth (Fitbit):** When we swapped surveys for Llama-inferred physiology, "Drift" exploded to **103 clips** ($8.6\times$ increase). 
*   **Observation:** The autonomic nervous system actively captured the emotional tension that the viewers subconsciously suppressed or ignored.

---

## Slide 10: The Switching Phenomenon
*   **Visual Check:** *(Insert `tab:switching-def` or a generic 2x2 graphic showing Q3$\rightarrow$Q1 vs Q1$\rightarrow$Q1)*
*   **The Core Flaw in Literature:** If Clip 1 is Happy and Clip 2 is Happy, we call this **Non-Switching**. You cannot use this sequence to prove a clip "caused" happiness because the user was already happy.
*   **The Fix:** You must exclusively analyze **Switching Pairs** where Clip 1 and Clip 2 cross quadrants (e.g., Sad $\rightarrow$ Happy) to prove causal emotional drift.

---

## Slide 11: Result 2 — Conscious Denial
*   **Visual Check:** *(Insert `_sigma_barchart.png` highlighting Hyp-2)*
*   **Observation:** When a user is forced across large, opposing emotional quadrants (e.g., Sad to Happy), they subconsciously resist the change in surveys.
*   **Significance:** We logged statistically significant "Conscious Denial" (where the body drifts but the survey explicitly denies it) occurring exclusively in switching sequences ($p = 0.0009$).

---

## Slide 12: Result 3 — Diagonal Dominance & Neutral Mirroring
*   **Visual Check:** *(Insert `_drift_boxplot_valence.png`)*
*   **Diagonal Jumps ($p=0.033$):** Subjecting a user to a diagonal cross-axis transition (Q1 $\leftrightarrow$ Q3) produces $3\times$ higher body-mind dissociation than adjacent transitions.
*   **Neutral Mirroring ($p=0.015$):** If you show a viewer an ambiguous/neutral clip right after a powerful emotional sequence, the viewer's survey will "mirror" their internal baseline 52.3% of the time, hijacking the neutral clip.
*   **Observation:** A clip classified historically as "Calm" does totally distinct physiological work depending on if a viewer is already calm, or merely transitioning toward calm.

---

## Slide 13: Phase 5 — Independent Clinical Validation
*   **Visual Check:** *(Insert `Screenshot 2026-04-11 004334.png` and `Screenshot 2026-04-10 091624.png` showing the validation app)*
*   **Experiment:** Built an entirely independent web portal with forced counter-balanced sequences across $\sim34$ distinct analytical user sessions.
*   **Observation:** Successfully reproduced the analytical findings. Users in a "Forced Drift" sequence exhibited a violently higher internal valence shift ($|\Delta I_1| = 1.406$) than users placed in a "Maintenance" sequence ($p = 0.014$).

---

## Slide 14: Recommended Clinical Protocols
*   **Visual Check:** *(Display Table 5.18 "Lab Baseline Induction" flowchart)*
*   **Application:** Clinicians and Affective Computing labs can use our sequence tables to reliably induce mental states.
*   **Example Rule:** If a psychologist needs to forcefully drag a patient out of a Q3 (Sad) state into a Q1 (Happy) state, they must strictly monitor $I_2$ (Fitbit), NEVER $I_1$ (surveys), because the user will reliably deny the emotional uplift due to conscious resistance logic shown in our statistics.

---

## Slide 15: Conclusion & Future Works
*   **Conclusion:** The $-120s$ Autonomic pipeline proves 'Affective Drift' exists. Relying on isolated self-reports is scientifically insufficient.
*   **Future Scope 1:** Sensor fusion—integrating Galvanic Skin Response (EDA) and 64Hz medical ECG for higher arousal confidence.
*   **Future Scope 2:** Real-Time Architecture—Deploying our LLM inference directly to a smartphone app to actively switch film/music media in real-time as a user attempts to drift into negative states. 

---

### Tips for Speaker (Utsavi)
*   **Time Management:** Slides 2-5 should take exactly 3 minutes. Spend the massive bulk of your 12 minutes talking about Slide 7 (-120s shift), Slide 9 (Explosion of Drift), and Slide 11/12 (Switching Phenomenon charts).
*   **Bivas Sir's specific request:** When pointing to `_drift_boxplot_valence.png`, literally say out loud: "As observed in this plot, the distribution drastically separates when switching..." (He specifically requested observation text to accompany every plot visually).
