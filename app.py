import streamlit as st
from src.data_loader import load_and_merge_tmdb_data
from src.recommender import recommend, prepare_recommender_data

st.set_page_config(
    page_title="CineMatch | Modern Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

@st.cache_data
def load_data():
    df = load_and_merge_tmdb_data()
    processed_df = prepare_recommender_data(df)
    return df, processed_df

df, processed_df = load_data()

st.title("🎬 CineMatch Movie Intelligence")

movie_list = processed_df['title'].values
selected_movie = st.selectbox("Search or Select a Movie:", movie_list)

if st.button("✨ Recommend"):
    recommendations = recommend(selected_movie, df, top_n=5)
    st.subheader(f"Top Recommendations for '{selected_movie}':")
    for idx, movie in enumerate(recommendations, 1):
        st.write(f"**{idx}.** {movie}")