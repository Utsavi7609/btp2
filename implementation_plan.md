# Generating Correct Validation Results

After a full audit of all files in both the outer and inner directories, here is the diagnosis and plan.

---

## What's Wrong / What Needs To Be Done

### 1. In-House Data (`InHouseCollectedData/`)

| Script | Status | Issue |
|--------|--------|-------|
| [evaluate_models.py](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/InHouseCollectedData/evaluate_models.py) | ✅ Correct | Reads [clean_dataset.csv](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/InHouseCollectedData/clean_dataset.csv), evaluates all 3 models. **Just needs to be run.** |
| [compare_models.py](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/InHouseCollectedData/compare_models.py) | ⚠️ Bug | Imports `os` at line 234 but uses it at line 180 → crashes. Easy fix. |
| [check_reported_vs_inferred.py](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/InHouseCollectedData/check_reported_vs_inferred.py) | ✅ Correct | Works on [llm_results_final2.csv](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/InHouseCollectedData/llm_results_final2.csv). |

### 2. K-EmoCon Validation (Outer dir)

| Item | Status | Issue |
|------|--------|-------|
| [kemocon_llm_predictions.csv](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/kemocon_llm_predictions.csv) | ❌ Broken | Old script used a free-text prompt without `response_format=json_object` → 85%+ rows are `PARSE_ERROR`. Need to re-run. |
| [kemocon_features.csv](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/kemocon_features.csv) | ❌ Incomplete | Only 5 of 28 participants recorded. Caused by [kemocon_llm_validation.py](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/kemocon_llm_validation.py) only processing `[:5]`. |
| [kemocon_llm_inference_final.py](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/kemocon_llm_inference_final.py) | ✅ Correct | This is the good script (Health-LLM format, JSON response_format, resume logic). Must be **run from the outer dir** (`InHouseCollectedData-20260118T212357Z-1-001/`). |
| K-EmoCon annotations | ✅ Present | All 32 participant files in `aggregated_external_annotations/`. Format: valence & arousal, 1-5 scale annotated every 5s. |
| E4 HR/IBI data | ✅ Present | All 28 participant directories extracted. |

> [!IMPORTANT]
> The K-EmoCon annotations note: file [P2.external.csv](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/K-EmoCon/extracted/emotion_annotations/aggregated_external_annotations/P2.external.csv) exists but there is no participant folder `2` in `e4_data/`. 
> Participants 2, 3, 6, 7 have annotation files but NO e4_data directory (they were excluded from the dataset). 
> Only participants with both annotation + e4_data will be evaluated (28 participants).

---

## Proposed Changes

### Component 1 — Fix [compare_models.py](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/InHouseCollectedData/compare_models.py) in InHouseCollectedData/

#### [MODIFY] [compare_models.py](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/InHouseCollectedData/compare_models.py)
- Move `import os` from line 234 to the top of the file (line 26).
- Also add `Llama4` model to the comparison (it's in [clean_dataset.csv](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/InHouseCollectedData/clean_dataset.csv) but compare_models.py only handles Llama and Mistral).

### Component 2 — Fix [evaluate_models.py](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/InHouseCollectedData/evaluate_models.py) in InHouseCollectedData/

**No changes needed.** Run as-is from the `InHouseCollectedData/` directory.

### Component 3 — Re-run K-EmoCon inference from outer dir

**No script changes needed.** The [kemocon_llm_inference_final.py](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/kemocon_llm_inference_final.py) in the outer dir is correct.  
But we need to **delete the old broken output files** first so the resume logic starts fresh:
- `kemocon_llama_results.csv` (does not exist yet — good)
- `kemocon_mistral_results.csv` (does not exist yet — good)

Run from: `D:\BTP\btp2\salmamaterials\InHouseCollectedData-20260118T212357Z-1-001\`

### Component 4 — Final merged results CSV

After all runs complete, write a **new script `generate_final_results.py`** in the outer dir that:
1. Loads in-house model metrics from the evaluate_models output
2. Loads K-EmoCon metrics from [kemocon_validation_metrics.csv](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/kemocon_validation_metrics.csv)  
3. Merges into one clean `final_validation_table.csv`
4. Produces the key inferred valence/arousal CSVs for each model

---

## Execution Order

```
Step 1: Fix compare_models.py (move import os to top, add llama4)
Step 2: Run evaluate_models.py  → prints all 3 model metrics on in-house data  
Step 3: Run compare_models.py   → per-clip and per-participant breakdown
Step 4: Run kemocon_llm_inference_final.py (from outer dir) → generates kemocon_llama/mistral_results.csv
Step 5: Create + run generate_final_results.py → final_validation_table.csv
```

---

## Verification Plan

### Step 2 Output Check
[evaluate_models.py](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/InHouseCollectedData/evaluate_models.py) should print MAE, Pearson r, Acc±1 for `llama`, `mistral`, `llama4`.  
Expected: MAE in range 0.5–1.5, Pearson r in 0.0–0.4 (in-house data is small).

### Step 3 Output Check
[compare_models.py](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/InHouseCollectedData/compare_models.py) should produce:
- `model_comparison_results.csv`  
- `per_clip_comparison.csv`

### Step 4 Output Check (K-EmoCon)
[kemocon_llm_inference_final.py](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/kemocon_llm_inference_final.py) should produce:
- [kemocon_features.csv](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/kemocon_features.csv) — 28 rows  
- `kemocon_llama_results.csv` — 28 rows  
- `kemocon_mistral_results.csv` — 28 rows  
- [kemocon_validation_metrics.csv](file:///D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/kemocon_validation_metrics.csv) — 2 rows (one per model)

> [!WARNING]
> Step 4 requires a **Groq API key** embedded in the script. It will make **56 API calls** (28 participants × 2 models) at 1.5s delay each. Estimated time: ~5-10 minutes, may hit rate limits (the script has automatic wait-and-retry logic for 429 errors).

### Step 5 Output Check
`generate_final_results.py` should produce `final_validation_table.csv` with all models compared side by side.
