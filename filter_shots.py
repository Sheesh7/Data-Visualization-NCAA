import os
import pandas as pd

DATA_DIR = "data"

def analyze_shot_subset():
    print("--- Loading Play-by-Play for Shot Filtering ---")
    pbp_path = os.path.join(DATA_DIR, "play_by_play.parquet")
    

    cols_to_load = [
        'game_id', 'id', 'sequence_number', 'type_id', 'type_text', 'text', 
        'short_description', 'shooting_play', 'scoring_play', 'score_value', 
        'points_attempted', 'coordinate_x', 'coordinate_y', 'team_id', 'season'
    ]
    
    pbp_df = pd.read_parquet(pbp_path, columns=cols_to_load)
    
    print(f"Total rows loaded: {len(pbp_df):,}")
    

    shots_df = pbp_df[pbp_df['shooting_play'] == True].copy()
    print(f"Total shooting plays identified: {len(shots_df):,}")
    

    coords_present = shots_df['coordinate_x'].notna().sum()
    coords_missing = shots_df['coordinate_x'].isna().sum()
    pct_coords = (coords_present / len(shots_df)) * 100
    
    print(f"\n--- Spatial Coordinate Coverage ---")
    print(f"Shots with X/Y coordinates: {coords_present:,} ({pct_coords:.2f}%)")
    print(f"Shots missing X/Y coordinates: {coords_missing:,} ({100 - pct_coords:.2f}%)")
    

    print(f"\n--- Target & Value Distribution ---")
    print(shots_df['scoring_play'].value_counts(dropna=False))
    print("\nPoints Attempted Value Counts:")
    print(shots_df['points_attempted'].value_counts(dropna=False))
    

    print(f"\n--- Sample Shot Descriptions ---")
    print(shots_df[['short_description', 'text', 'scoring_play', 'points_attempted']].head(5).to_string())

if __name__ == "__main__":
    analyze_shot_subset()