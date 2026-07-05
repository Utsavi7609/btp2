# LLM Emotion Inference Validation Report

We have successfully repaired the validation pipeline and executed the LLM physiological-to-emotion evaluations on both your **In-House Fitbit Dataset** and the **K-EmoCon E4 Dataset**.

## Executive Summary

Based on the evaluation criteria (Mean Absolute Error, Pearson Correlation, and Accuracy within ±1 point on a 1-5 scale), **the zero-shot LLM approach struggles to extract reliable emotions exclusively from resting heart-rate and HR variability features without contextualizing text.** 

The correlations ($r$) are either slightly negative or very near zero across both datasets, and while the "Accuracy ±1" metric looks decent (up to 75%), this is primarily because valence/arousal annotations naturally cluster around the median (3), making it statistically easier to be within ±1 point of the truth by guessing near the average.

**Recommendation:** Yes, **fine-tuning or few-shot in-context learning is highly necessary** if you intend to map basic HR/IBI metrics to 2D Valence/Arousal spaces accurately. The current foundational LLMs (Llama-3.3, Qwen-32B, Llama-4) do not possess the precise physiological decoding capabilities out-of-the-box.

---

## 1. In-House Fitbit Data Results

Evaluated on 1,493 clips/samples against self-reported annotations.

| Model | Valence MAE | Arousal MAE | Val. Pearson $r$ | Aro. Pearson $r$ | Val. Acc ±1 | Aro. Acc ±1 |
| --- | --- | --- | --- | --- | --- | --- |
| **Qwen3-32B** | 1.056 | 1.031 | -0.071 | 0.000 | 70.2% | 72.6% |
| **Llama-4-Scout-17B** | 1.054 | 1.104 | -0.112 | -0.008 | 72.8% | 70.2% |

**Observations:**
- MAE hovers tightly around ~1.05, meaning the model is typically off by 1 full scale point.
- The Pearson correlations are essentially 0 (or slightly negative), indicating the models' inferred directionality does not track with the users' self-reported increases or decreases in arousal/valence.

---

## 2. K-EmoCon Cross-Dataset Results

Evaluated on 28 participants with physiological data derived from clinical-grade E4 wristbands against observer annotations to eliminate self-reporting bias. Note that `mixtral-8x7b-32768` was decommissioned on Groq, so `llama-3.1-8b-instant` was used as a fallback.

| Model | Valence MAE | Arousal MAE | Val. Pearson $r$ | Aro. Pearson $r$ | Val. Acc ±1 | Aro. Acc ±1 |
| --- | --- | --- | --- | --- | --- | --- |
| **Llama-3.3-70B-Versatile** | 0.763 | 1.113 | -0.183 | 0.235 | 75.0% | 39.3% |
| **Llama-3.1-8B-Instant** | 1.013 | 1.154 | N/A | 0.215 | 65.4% | 46.2% |

**Observations:**
- Llama-3.3-70B achieved a visibly better Valence MAE (0.763), showing that larger parameter models are slightly more capable of guessing the baseline.
- However, Arousal accuracy is very poor (39.3% - 46.2% within ±1).
- We see a slight positive correlation for Arousal ($r$ ≈ 0.23), suggesting the LLMs recognize massive shifts in heart-rate variance as Arousal spikes, but fail entirely at tracking Valence ($r$ = -0.18).

---

## File Deliverables Built

All of the fixing and execution resulted in the following files you can directly import to your BTP thesis report:

1. [final_validation_table.csv](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/final_validation_table.csv) - Combined metrics across all models and datasets.
2. [model_comparison_results.csv](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/InHouseCollectedData/model_comparison_results.csv) / [per_clip_comparison.csv](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/InHouseCollectedData/per_clip_comparison.csv) - In-House breakdowns for Qwen/Llama4.
3. [kemocon_llama_results.csv](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/kemocon_llama_results.csv) - The raw outputs from the new K-EmoCon JSON inference workflow.
