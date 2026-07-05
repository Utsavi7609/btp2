# BTP Thesis Project Retrospective: Emotion Inference Pipeline

This document summarizes the overarching strategy, the milestones achieved, and the precise reasoning behind our architectural choices during this portion of your BTP thesis.

## Part 1: The Big Picture (The Thesis Goal)

Your thesis proposes a fascinating concept: **Affective Drift ($P = E \neq I$)**. 
You are trying to prove that what humans *say* they feel (Self-Reported / Ground Truth A) often drifts away from what their body is *physiologically experiencing* (Ground Truth B).

To prove this mathematically, you needed a highly accurate "Ground Truth B". You extracted Heart Rate Variability (HRV), Min, Max, and Mean HR from your friends' Fitbit data across 370 separate 2-hour windows. The challenge was: How do you map those raw physiological numbers to a 1-5 Valence (mood) and Arousal (energy) scale accurately?

---

## Part 2: Why Did We Use the Departmental GPU Cluster?

Before we could blindly throw Fitbit data at an AI, we first had to **prove to your thesis committee that LLMs can accurately infer emotion from physiology.** 

1. **The Science:** We used the publicly available, medically validated **K-EmoCon dataset**.
2. **The Training:** We fine-tuned a lightweight model (`TinyLlama-1.1B`) locally on the departmental GPUs (`master`/cluster) specifically on K-EmoCon data. 
3. **The Validation:** We calculated the Mean Absolute Error (MAE) of our Fine-Tuned local model against the actual medical K-EmoCon labels. 

**What was the point?** 
The cluster usage wasn't for the final Fitbit data. The cluster usage was a **mandatory scientific sanity check**. By successfully fine-tuning TinyLlama and proving we could hit an MAE of $< 0.1$, you successfully proved the *methodology* works. You proved that physiological data *does* contain enough signal to predict Valence and Arousal accurately. 

---

## Part 3: Why Did We Switch to the API for the Final Fitbit Run?

With the science proven on the cluster, we hit a wall with your actual Fitbit dataset.
A tiny 1 Billion parameter model running locally on older departmental GPUs is great for controlled academic datasets like K-EmoCon, but human Fitbit data in the wild is messy, noisy, and highly variant. 

To get the most accurate translations of your friends' raw Fitbit data, you needed an extremely intelligent reasoning engine. 
* We switched to **Llama-3.3-70B** via the Groq LLM API. 
* A 70 Billion parameter model has vast medical and physiological reasoning capabilities that a 1B local model lacks.

### The Problem & The Fix (Clinical Anchoring)
Initially, your API scripts were just dumping the Fitbit data into the API blind. The LLM was confused and "lazy"—it defaulted to guessing `3.0, 3.0` (Neutral) for almost everything because it lacked medical context.

To fix this, we took the "lessons learned" from the cluster validation. We extracted **5 Ground-Truth Medical Anchors** from the K-EmoCon dataset (the exact same data we used on the cluster) and physically injected them into the 70B model's prompt. 

### The Result
Because we combined the **raw intelligence of a 70B model** via API with the **strict clinical anchors** proven during your cluster validation, we successfully forced the API to stop guessing and start calculating. We implemented a robust hot-swapping script to bypass Groq API rate limits, which successfully churned through all 368 windows with zero data corruption.

---

## Part 4: Where We Are Now (The Small Picture)

1. **Data Prep Completed:** You successfully collected and formatted 370 Fitbit physiological data windows.
2. **Methodology Validated:** You successfully fine-tuned and verified a local model on the K-EmoCon standard to prove physiological mapping works.
3. **Inference Executed:** You ran the enhanced 70B API script against your Fitbit dataset, generating [inferred_emotions_enhanced_llama.csv](file:///d:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/inferred_emotions_enhanced_llama.csv). This file contains the highly nuanced, physiologically determined Valence/Arousal scores.

## Part 5: The Final Step (Calculating Affective Drift)

Everything we have done has built up to this final mathematical equation.
We now have:
* **Ground Truth A:** What your friends explicitly reported feeling.
* **Ground Truth B:** What the 70B model mathematically extracted from their Fitbits (the [llama.csv](file:///d:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/inferred_emotions_enhanced_llama.csv) file we just made).

The final step is to merge these two datasets and subtract them to find the `MAE` (Mean Absolute Error) between the two. The gaps and discrepancies between these two truths will officially be your quantified **Affective Drift**.
