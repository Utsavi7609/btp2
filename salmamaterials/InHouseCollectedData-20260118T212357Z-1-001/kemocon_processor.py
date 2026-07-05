# """
# K-EmoCon Dataset Validation
# Template for validating against public benchmark
# """

# import pandas as pd
# import numpy as np
# from pathlib import Path

# print("="*70)
# print("K-EMOCON DATASET VALIDATION")
# print("="*70)

# # K-EmoCon dataset structure (you need to download it first)
# # https://drive.google.com/drive/folders/1lYAzLsvtm2ahsEZoDaIOXbsKt7JmKiXJ

# kemocon_path = Path("K-EmoCon")  # Update this path

# if not kemocon_path.exists():
#     print("\n⚠️  K-EmoCon dataset not found!")
#     print("\nTo use this validation:")
#     print("1. Download K-EmoCon from:")
#     print("   https://drive.google.com/drive/folders/1lYAzLsvtm2ahsEZoDaIOXbsKt7JmKiXJ")
#     print("2. Extract to a folder named 'K-EmoCon'")
#     print("3. Show me the folder structure")
#     print("\nOnce you have it, I'll update this script to:")
#     print("  - Load K-EmoCon physiological data")
#     print("  - Extract same features (HR, HRV)")
#     print("  - Run LLM inference")
#     print("  - Compare with K-EmoCon ground truth labels")
#     exit()

# print("\n✅ K-EmoCon dataset found")
# print("\nNext: Show me the folder structure so I can complete this script")


"""
K-EmoCon Dataset Processor
Validates LLM predictions against public benchmark
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

print("="*70)
print("K-EMOCON DATASET VALIDATION")
print("="*70)

# Step 1: Check if dataset exists
kemocon_path = Path("K-EmoCon")

if not kemocon_path.exists():
    print("\n⚠️  K-EmoCon dataset not found")
    print("\nDOWNLOAD INSTRUCTIONS:")
    print("="*70)
    print("1. Visit: https://drive.google.com/drive/folders/1lYAzLsvtm2ahsEZoDaIOXbsKt7JmKiXJ")
    print("2. Download the entire folder")
    print("3. Extract to current directory")
    print("4. Rename folder to 'K-EmoCon'")
    print("\nAfter download, run this script again.")
    print("\nI will then:")
    print("  ✓ Extract physiological features from K-EmoCon")
    print("  ✓ Run LLM emotion inference")
    print("  ✓ Compare with K-EmoCon ground truth labels")
    print("  ✓ Generate validation report")
    exit()

# Step 2: Analyze structure
print("\n✅ K-EmoCon dataset found!")
print(f"   Location: {kemocon_path.absolute()}\n")

print("Analyzing dataset structure...")
print("-"*70)

# Find all files
all_files = list(kemocon_path.rglob("*"))
file_types = {}

for f in all_files:
    if f.is_file():
        ext = f.suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1

print("\nFile types found:")
for ext, count in sorted(file_types.items()):
    print(f"  {ext:10s}: {count:4d} files")

print("\n" + "="*70)
print("DATASET STRUCTURE ANALYSIS")
print("="*70)
print("\nPlease share the folder structure by running:")
print("\nWindows:")
print("  tree K-EmoCon /F > kemocon_structure.txt")
print("\nLinux/Mac:")
print("  tree K-EmoCon > kemocon_structure.txt")
print("\nThen paste the content here or share kemocon_structure.txt")

print("\n" + "="*70)
print("ONCE YOU SHARE THE STRUCTURE:")
print("="*70)
print("I will create a complete processing script that:")
print("  1. Loads K-EmoCon physiological signals")
print("  2. Extracts HR and HRV features")
print("  3. Runs LLM emotion inference (same as your data)")
print("  4. Compares with K-EmoCon ground truth emotion labels")
print("  5. Generates comparison metrics and plots")
print("  6. Creates final validation report for Salma ma'am")