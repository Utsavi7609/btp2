"""
Extract user profiles from Fitbit Profile.csv
WINDOWS VERSION with CORRECT folder structure
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

def extract_user_profile(user_folder):
    """
    Extract profile from: InHouseCollectedData/{user}/takeout-*/Takeout/Fitbit/Your Profile/Profile.csv
    """
    user_name = user_folder.name
    
    # CORRECTED PATH: Look for Takeout/Fitbit/Your Profile/Profile.csv
    for profile_file in user_folder.rglob('Takeout/Fitbit/Your Profile/Profile.csv'):
        print(f"📄 Reading: {profile_file}")
        
        try:
            df = pd.read_csv(profile_file)
            
            # Calculate age from date_of_birth
            dob_str = df['date_of_birth'].iloc[0]  # Format: YYYY-MM-DD
            birth_year = int(dob_str.split('-')[0])
            current_year = 2025
            age = current_year - birth_year
            
            profile = {
                'user_id': user_name,
                'age': age,
                'gender': df['gender'].iloc[0],
                'height': df['height'].iloc[0],  # cm
                'weight': df['weight'].iloc[0],  # kg
                'full_name': df['full_name'].iloc[0]
            }
            
            return profile
        except Exception as e:
            print(f"  ⚠️  Error reading {profile_file}: {e}")
            continue
    
    return None

def main():
    # UPDATED: Handle the nested InHouseCollectedData structure
    base_path = Path('InHouseCollectedData')
    
    if not base_path.exists():
        print(f"❌ Error: {base_path} not found!")
        print(f"   Current directory: {Path.cwd()}")
        print(f"   Please run this script from: D:\\BTP\\btp2\\salmamaterials\\InHouseCollectedData-20260118T212357Z-1-001")
        return
    
    profiles = []
    
    for user_folder in base_path.iterdir():
        if not user_folder.is_dir():
            continue
        
        print(f"\n🔍 Processing user: {user_folder.name}")
        profile = extract_user_profile(user_folder)
        
        if profile:
            profiles.append(profile)
            print(f"  ✅ {profile['full_name']} - Age: {profile['age']}, Gender: {profile['gender']}")
        else:
            print(f"  ⚠️  No profile found for {user_folder.name}")
    
    if not profiles:
        print("\n❌ No profiles extracted!")
        print("\n📁 Folder structure check:")
        print(f"   Expected: InHouseCollectedData/{{user}}/takeout-*/Takeout/Fitbit/Your Profile/Profile.csv")
        return
    
    # Save
    result_df = pd.DataFrame(profiles)
    result_df.to_csv('user_profiles.csv', index=False)
    
    print(f"\n✅ Saved {len(result_df)} profiles to: user_profiles.csv")
    print("\nProfile Summary:")
    print(result_df.to_string(index=False))

if __name__ == "__main__":
    main()