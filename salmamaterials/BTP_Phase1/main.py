import pandas as pd
import os

# 1. SETUP FILENAMES
# This must match your file name exactly
excel_file = 'clip_responses_export.xlsx' 

if not os.path.exists(excel_file):
    print(f"ERROR: Could not find '{excel_file}' in this folder.")
    print("Please make sure the script and the Excel file are in the same folder.")
else:
    # 2. LOAD DATA
    # We read the Excel file. If your data is not on the first sheet, 
    # you can add sheet_name='Sheet1' inside the brackets.
    df = pd.read_excel(excel_file)
    print("Successfully loaded Excel data.")

    # 3. DEFINE YOUR 6 USERS
    target_users = ['Nishant', 'debjit2001', 'pranjal123', 'aritra', 'sayantankuila', 'ajaycc17']
    
    # Filter the data to keep only these users
    df_filtered = df[df['participant'].isin(target_users)].copy()

    def create_and_save_csv(mode):
        """
        mode: 'valence' or 'arousal'
        """
        # Pick the columns we need for this specific file
        cols = ['clip_title', 'clip_order', 'participant', 
                f'participant_{mode}', f'clip_{mode}', f'impact_{mode}']
        
        # 'Pivot' the data: Turn user rows into columns
        # This makes it look like your example_file.csv
        pivoted = df_filtered[cols].pivot(index=['clip_title', 'clip_order'], columns='participant')
        
        # Rename columns to the format: User_Stat_Mode (e.g., Nishant_impact_valence)
        pivoted.columns = [f'{user}_{stat}' for stat, user in pivoted.columns]
        pivoted = pivoted.reset_index()
        
        # Organize columns so they appear in a nice order per user
        final_order = ['clip_title', 'clip_order']
        for user in target_users:
            # Check if this user actually has data before adding columns
            user_cols = [f'{user}_participant_{mode}', f'{user}_clip_{mode}', f'{user}_impact_{mode}']
            if user_cols[0] in pivoted.columns:
                final_order.extend(user_cols)
        
        # Create final dataframe and save
        output_df = pivoted[final_order]
        output_filename = f'{mode}_btp2.csv'
        output_df.to_csv(output_filename, index=False)
        print(f"Done! Created: {output_filename}")

    # 4. RUN FOR BOTH VALENCE AND AROUSAL
    create_and_save_csv('valence')
    create_and_save_csv('arousal')
    
    print("\nPhase 1 complete. You now have the two CSV files needed for the next steps.")