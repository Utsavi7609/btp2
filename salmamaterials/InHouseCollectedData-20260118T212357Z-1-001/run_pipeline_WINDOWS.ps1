# BTP Emotion Inference Pipeline - Windows Version
# Run this script from: D:\BTP\btp2\salmamaterials\InHouseCollectedData-20260118T212357Z-1-001

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "BTP Emotion Inference Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Extract User Profiles
Write-Host "[1/3] Extracting user profiles..." -ForegroundColor Yellow
python extract_user_profiles_WINDOWS.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Profile extraction failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Profile extraction complete!" -ForegroundColor Green
Write-Host ""

# Step 2: Extract Heart Rate Data
Write-Host "[2/3] Extracting heart rate data..." -ForegroundColor Yellow
python extract_heart_rate_data_WINDOWS.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Heart rate extraction failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Heart rate extraction complete!" -ForegroundColor Green
Write-Host ""

# Step 3: LLM Emotion Inference
Write-Host "[3/3] Running LLM emotion inference..." -ForegroundColor Yellow
Write-Host "This may take 10-30 minutes depending on data size..." -ForegroundColor Cyan
python llm_emotion_inference_WINDOWS.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: LLM inference failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PIPELINE COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Output files created:" -ForegroundColor Yellow
Write-Host "  1. user_profiles.csv" -ForegroundColor White
Write-Host "  2. fitbit_hr_raw.csv" -ForegroundColor White
Write-Host "  3. fitbit_hr_windows.csv" -ForegroundColor White
Write-Host "  4. inferred_emotions_llm.csv" -ForegroundColor White
Write-Host "  5. inferred_valence_summary.csv" -ForegroundColor White
Write-Host "  6. inferred_arousal_summary.csv" -ForegroundColor White
Write-Host ""
