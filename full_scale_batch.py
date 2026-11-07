import os
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

DATA_DIR = "data"

def process_full_dataset_in_batches(batch_size=500000):
    print("--- Starting Full-Scale Parquet Batch Processing ---")
    pbp_path = os.path.join(DATA_DIR, "play_by_play.parquet")
    

    parquet_file = pq.ParquetFile(pbp_path)
    print(f"Total row groups in dataset: {parquet_file.num_row_groups}")
    
    total_shots_processed = 0
    processed_chunks = []
    
    cols_to_load = [
        'game_id', 'id', 'text', 'short_description', 'shooting_play', 
        'scoring_play', 'score_value', 'points_attempted', 
        'coordinate_x', 'coordinate_y', 'period', 'start_game_seconds_remaining', 'team_id',
        'home_score', 'away_score'
    ]
    

    for i, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size, columns=cols_to_load)):
        df_chunk = batch.to_pandas()
        

        shots_chunk = df_chunk[df_chunk['shooting_play'] == True].copy()
        
        if len(shots_chunk) == 0:
            continue
            

        shots_chunk['shot_made'] = shots_chunk['scoring_play'].astype(int)
        shots_chunk['is_three'] = shots_chunk['text'].str.contains('Three Point|3-pointer|3pt', case=False, na=False).astype(int)
        shots_chunk['points_attempted'] = np.where(shots_chunk['is_three'] == 1, 3.0, 2.0)
        

        shots_chunk['score_diff'] = shots_chunk['home_score'] - shots_chunk['away_score']
        

        processed_chunks.append(shots_chunk[[
            'game_id', 'shot_made', 'is_three', 'points_attempted', 
            'score_diff', 'start_game_seconds_remaining', 'coordinate_x', 'coordinate_y'
        ]])
        
        total_shots_processed += len(shots_chunk)
        print(f"Processed batch {i+1}: Found {len(shots_chunk):,} shots (Cumulative total: {total_shots_processed:,})")
        

        if total_shots_processed >= 2000000:
            print("Reached 2 million processed shots cap for robust model training.")
            break


    master_shots_df = pd.concat(processed_chunks, ignore_index=True)
    
    output_path = os.path.join(DATA_DIR, "master_processed_shots.parquet")
    master_shots_df.to_parquet(output_path)
    print(f"\nSuccessfully saved master dataset of {len(master_shots_df):,} processed shots to {output_path}!")

if __name__ == "__main__":
    process_full_dataset_in_batches()