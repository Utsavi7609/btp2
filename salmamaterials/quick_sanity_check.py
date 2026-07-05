import pandas as pd

def check_health(dim):
    # Load what they said (Self) and what the LLM said (Body)
    self_df = pd.read_csv(f'induced_{dim}_summary.csv')
    body_df = pd.read_csv(f'inferred_{dim}_summary.csv')
    
    # Simple average check
    self_avg = self_df[f'avg_{dim}'].mean()
    body_avg = body_df[f'avg_inferred_{dim}'].mean()
    
    print(f"--- {dim.upper()} Check ---")
    print(f"Average Self-Reported: {self_avg:.2f}")
    print(f"Average LLM-Inferred: {body_avg:.2f}")
    
    # Check for "Flatlining" (LLM giving the same number for everything)
    variation = body_df[f'avg_inferred_{dim}'].std()
    if variation < 0.1:
        print("ALERT: Your LLM is 'flatlining' (outputting the same score for every clip). Redo generation!")
    else:
        print("SUCCESS: LLM values show variation between clips.")

check_health('valence')
check_health('arousal')