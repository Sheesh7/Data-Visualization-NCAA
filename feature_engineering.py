import os
import re
import pandas as pd
import numpy as np

DATA_DIR = "data"

def engineer_shot_features():
    print("--- 1. Loading Shot Subset ---")
    pbp_path = os.path.join(DATA_DIR, "play_by_play.parquet")
    
    cols = [
        'game_id', 'id', 'sequence_number', 'type_id', 'type_text', 'text', 
        'short_description', 'shooting_play', 'scoring_play', 'score_value', 
        'points_attempted', 'coordinate_x', 'coordinate_y', 'period', 
        'start_game_seconds_remaining', 'team_id'
    ]
    
    pbp_df = pd.read_parquet(pbp_path, columns=cols)
    shots_df = pbp_df[pbp_df['shooting_play'] == True].copy()
    
    print(f"Processing {len(shots_df):,} shots...")

    print("\n--- 2. Defining Target Variable (Shot Made) ---")

    shots_df['shot_made'] = shots_df['scoring_play'].astype(int)
    

    shots_df['points_attempted'] = shots_df['points_attempted'].fillna(shots_df['score_value'])
    

    three_point_mask = shots_df['text'].str.contains('Three Point|3-pointer|3pt', case=False, na=False)
    shots_df.loc[shots_df['points_attempted'].isna() & three_point_mask, 'points_attempted'] = 3.0
    shots_df['points_attempted'] = shots_df['points_attempted'].fillna(2.0) # Default field goal is 2

    print("\n--- 3. Extracting Text/NLP Shot Features ---")
    text_series = shots_df['text'].fillna('')
    
    shots_df['is_three'] = text_series.str.contains('Three Point|3-pointer|3pt', case=False).astype(int)
    shots_df['is_layup'] = text_series.str.contains('layup', case=False).astype(int)
    shots_df['is_jumper'] = text_series.str.contains('jumper|jump shot', case=False).astype(int)
    shots_df['is_dunk'] = text_series.str.contains('dunk', case=False).astype(int)
    shots_df['is_hook'] = text_series.str.contains('hook', case=False).astype(int)
    

    shots_df['has_coords'] = shots_df['coordinate_x'].notna().astype(int)
    shots_df['dist_to_basket'] = np.where(
        shots_df['has_coords'] == 1,
        np.sqrt((shots_df['coordinate_x'].fillna(0) - 50)**2 + (shots_df['coordinate_y'].fillna(0) - 25)**2),
        np.nan
    )
    
    print("\n--- 4. Summary of Engineered Feature Matrix ---")
    print(shots_df[['shot_made', 'points_attempted', 'is_three', 'is_layup', 'is_jumper', 'has_coords']].head(10).to_string())
    print("\nTarget Class Balance (Shot Made):")
    print(shots_df['shot_made'].value_counts(normalize=True))
    

    output_path = os.path.join(DATA_DIR, "processed_shots_sample.parquet")
    shots_df.head(100000).to_parquet(output_path)
    print(f"\nSaved 100,000 processed training rows to {output_path} for fast experimentation.")

if __name__ == "__main__":
    engineer_shot_features()