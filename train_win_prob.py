import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA_DIR = "data"

def train_win_probability_model():
    print("--- 1. Loading Master Processed Shots ---")
    df_path = os.path.join(DATA_DIR, "master_processed_shots.parquet")
    df = pd.read_parquet(df_path)
    
    print(f"Loaded {len(df):,} shots for Win Probability modeling.")

    print("\n--- 2. Computing Game Outcomes ---")
    final_scores = df.groupby('game_id')['score_diff'].last().reset_index()
    final_scores['home_win'] = (final_scores['score_diff'] > 0).astype(int)
    
    df = df.merge(final_scores[['game_id', 'home_win']], on='game_id', how='left')
    df = df[df['score_diff'] != 0].copy()

    print("\n--- 3. Training Win Probability Model ---")
    feature_cols = ['score_diff', 'start_game_seconds_remaining']
    
    X = df[feature_cols].fillna(0)
    y = df['home_win']
    
    wp_model = LogisticRegression(max_iter=1000, random_state=42)
    wp_model.fit(X, y)
    
    df['pre_shot_wp'] = wp_model.predict_proba(X)[:, 1]
    print(f"Win Probability Model ROC-AUC: {roc_auc_score(y, df['pre_shot_wp']):.4f}")

    print("\n--- 4. Calculating WPA and Game Leverage ---")

    post_score_diff = np.where(
        df['shot_made'] == 1,
        df['score_diff'] + df['points_attempted'],
        df['score_diff'] - (df['points_attempted'] * 0.5)
    )
    

    X_post = pd.DataFrame({
        'score_diff': post_score_diff,
        'start_game_seconds_remaining': df['start_game_seconds_remaining']
    }).fillna(0)
    
    df['post_shot_wp'] = wp_model.predict_proba(X_post)[:, 1]
    

    df['wpa'] = df['post_shot_wp'] - df['pre_shot_wp']
    

    df['leverage'] = df['wpa'].abs()
    
    print("\nSample WPA and Leverage Output:")
    print(df[['game_id', 'score_diff', 'start_game_seconds_remaining', 'pre_shot_wp', 'wpa', 'leverage']].head(10).to_string())

    output_path = os.path.join(DATA_DIR, "master_shots_with_wpa.parquet")
    df.to_parquet(output_path)
    print(f"\nSaved WPA-enhanced dataset to {output_path}!")

if __name__ == "__main__":
    train_win_probability_model()