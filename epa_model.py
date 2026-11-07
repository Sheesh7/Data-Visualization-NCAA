import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

DATA_DIR = "data"

def train_shot_model():
    print("--- 1. Loading Processed Training Sample ---")
    df_path = os.path.join(DATA_DIR, "processed_shots_sample.parquet")
    df = pd.read_parquet(df_path)
    

    df['points_attempted'] = np.where(df['is_three'] == 1, 3.0, 2.0)
    
    print(f"Loaded {len(df):,} rows for model training.")
    print("Corrected Points Attempted Distribution:")
    print(df['points_attempted'].value_counts())

    print("\n--- 2. Defining Features & Target ---")
    feature_cols = ['is_three', 'is_layup', 'is_jumper', 'is_dunk', 'is_hook', 'has_coords', 'dist_to_basket']
    

    X = df[feature_cols].fillna(0)
    y = df['shot_made']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n--- 3. Training Classification Model (Logistic Regression Baseline) ---")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    

    preds_prob = model.predict_proba(X_test)[:, 1]
    

    auc = roc_auc_score(y_test, preds_prob)
    brier = brier_score_loss(y_test, preds_prob)
    print(f"Model ROC-AUC Score: {auc:.4f}")
    print(f"Model Brier Score (Calibration): {brier:.4f}")
    

    coefficients = pd.DataFrame({'Feature': feature_cols, 'Coefficient': model.coef_[0]})
    print("\nLearned Feature Coefficients:")
    print(coefficients.to_string(index=False))

    print("\n--- 4. Calculating Expected Points & EPA ---")

    test_df = X_test.copy()
    test_df['shot_made'] = y_test
    test_df['points_attempted'] = np.where(test_df['is_three'] == 1, 3.0, 2.0)
    test_df['make_prob'] = preds_prob
    test_df['expected_points'] = test_df['make_prob'] * test_df['points_attempted']
    

    test_df['actual_points'] = test_df['shot_made'] * test_df['points_attempted']
    test_df['epa'] = test_df['actual_points'] - test_df['expected_points']
    
    print("\nSample EPA Output:")
    print(test_df[['is_three', 'make_prob', 'expected_points', 'actual_points', 'epa']].head(10).to_string())

if __name__ == "__main__":
    train_shot_model()