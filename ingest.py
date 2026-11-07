import os
import pandas as pd

# Define data directory path
DATA_DIR = "data"

def load_and_validate_data():
    print("--- 1. Loading Parquet Datasets ---")
    
    # Load core files using PyArrow engine via pandas
    pbp_path = os.path.join(DATA_DIR, "play_by_play.parquet")
    games_path = os.path.join(DATA_DIR, "games.parquet")
    team_box_path = os.path.join(DATA_DIR, "team_box.parquet")
    
    pbp_df = pd.read_parquet(pbp_path)
    games_df = pd.read_parquet(games_path)
    team_box_df = pd.read_parquet(team_box_path)
    
    print(f"Play-by-Play loaded: {pbp_df.shape[0]:,} rows, {pbp_df.shape[1]} columns")
    print(f"Games loaded: {games_df.shape[0]:,} rows, {games_df.shape[1]} columns")
    print(f"Team Box loaded: {team_box_df.shape[0]:,} rows, {team_box_df.shape[1]} columns")
    
    print("\n--- 2. Schema Inspection (Play-by-Play Columns) ---")
    print(pbp_df.dtypes.to_string())
    
    print("\n--- 3. Checking for Missing Critical Keys ---")
    missing_games = pbp_df['game_id'].isna().sum()
    print(f"Missing game_ids in PBP: {missing_games}")
    
    return pbp_df, games_df, team_box_df

if __name__ == "__main__":
    pbp, games, team_box = load_and_validate_data()