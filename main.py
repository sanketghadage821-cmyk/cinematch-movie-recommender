from src.data_loader import load_and_merge_tmdb_data
from src.recommender import recommend, prepare_recommender_data

def main():
    print("Loading TMDB dataset...")
    df = load_and_merge_tmdb_data()
    print("Dataset loaded successfully!\n")

    # Pre-process data once to make searches faster
    processed_df = prepare_recommender_data(df)

    while True:
        user_input = input("\nEnter a movie name (or type 'exit' to quit): ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        recommend(user_input, df, top_n=5)

if __name__ == "__main__":
    main()