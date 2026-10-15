import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def convert_json_column(obj):
    """Extracts 'name' values from JSON string columns like genres and keywords."""
    L = []
    try:
        for i in ast.literal_eval(obj):
            L.append(i['name'])
    except Exception:
        pass
    return L

def get_top_cast(obj):
    """Extracts top 3 cast members from the cast JSON string."""
    L = []
    try:
        for i in ast.literal_eval(obj)[:3]:
            L.append(i['name'])
    except Exception:
        pass
    return L

def get_director(obj):
    """Extracts Director name from the crew JSON string."""
    L = []
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
    except Exception:
        pass
    return L

def prepare_recommender_data(df):
    """Preprocesses text fields and creates a combined feature tags string."""
    print("Extracting genres, keywords, cast, and crew...")
    
    # Filter key columns
    movies = df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']].copy()
    movies.dropna(inplace=True)

    # Convert JSON strings into Python lists
    movies['genres'] = movies['genres'].apply(convert_json_column)
    movies['keywords'] = movies['keywords'].apply(convert_json_column)
    movies['cast'] = movies['cast'].apply(get_top_cast)
    movies['crew'] = movies['crew'].apply(get_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    # Remove spaces from tags to combine words properly (e.g. 'Johnny Depp' -> 'JohnnyDepp')
    for col in ['genres', 'keywords', 'cast', 'crew']:
        movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

    # Combine all metadata into a single list of tags
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    
    # Create final clean DataFrame
    new_df = movies[['movie_id', 'title', 'tags']].copy()
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())
    
    return new_df

def recommend(movie_title, df, top_n=5):
    """Computes similarity scores and returns top_n recommended movie titles."""
    new_df = prepare_recommender_data(df)

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    # Search for movie index
    matching_movies = new_df[new_df['title'].str.lower() == movie_title.lower()]
    
    if matching_movies.empty:
        print(f"\nMovie '{movie_title}' not found in the dataset.")
        return []

    movie_idx = matching_movies.index[0]
    distances = similarity[movie_idx]

    # Rank movies by similarity score
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n+1]

    print(f"\nRecommendations for '{new_df.iloc[movie_idx]['title']}':\n" + "-" * 40)
    recommendations = []
    for i in movies_list:
        rec_title = new_df.iloc[i[0]]['title']
        recommendations.append(rec_title)
        print(f"• {rec_title}")

    return recommendations