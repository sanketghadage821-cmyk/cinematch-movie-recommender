import os
import pandas as pd

def load_and_merge_tmdb_data():
    movies_path = os.path.join('data', 'tmdb_5000_movies.csv')
    credits_path = os.path.join('data', 'tmdb_5000_credits.csv')

    print("Loading movies dataset...")
    movies_df = pd.read_csv(movies_path)

    print("Loading credits dataset...")
    credits_df = pd.read_csv(credits_path)

    print("Merging datasets on title...")
    merged_df = movies_df.merge(credits_df, on='title')

    return merged_df