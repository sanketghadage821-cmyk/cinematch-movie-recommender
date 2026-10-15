import streamlit as st
import requests
from src.data_loader import load_and_merge_tmdb_data
from src.recommender import recommend, prepare_recommender_data

# Page Setup
st.set_page_config(
    page_title="CineMatch AI | Ultra UI Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Glassmorphism & High-End UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Background Setup */
    .stApp {
        background: radial-gradient(circle at top right, #1a102f 0%, #080711 60%, #030206 100%);
        color: #f1f5f9;
    }

    /* Glassmorphism Header Banner */
    .glass-header {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }

    .main-glow-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ff4b4b 0%, #ff8a00 50%, #e52e71 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 0 10px 30px rgba(255, 75, 75, 0.3);
    }

    .sub-glow-text {
        color: #94a3b8;
        font-size: 1.2rem;
        font-weight: 400;
    }

    /* Movie Poster Cards */
    .movie-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.4);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .movie-card:hover {
        transform: translateY(-10px) scale(1.02);
        border-color: #ff4b4b;
        box-shadow: 0 20px 40px rgba(255, 75, 75, 0.3);
        background: rgba(255, 255, 255, 0.06);
    }

    .card-badge {
        background: linear-gradient(90deg, #ff4b4b 0%, #ff8a00 100%);
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 800;
        padding: 4px 14px;
        border-radius: 50px;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: inline-block;
        margin-bottom: 0.8rem;
    }

    .movie-title-text {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 0.5rem;
    }

    /* High-Tech Custom Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #ff4b4b 0%, #e52e71 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.8rem 1.5rem;
        border-radius: 14px;
        border: none;
        box-shadow: 0 10px 25px rgba(255, 75, 75, 0.4);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 15px 35px rgba(255, 75, 75, 0.6);
        background: linear-gradient(135deg, #e52e71 0%, #ff4b4b 100%);
    }

    /* Poster Frame Styling */
    .poster-img {
        width: 100%;
        border-radius: 12px;
        object-fit: cover;
        box-shadow: 0 8px 20px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to fetch movie poster
def fetch_poster(movie_title, df):
    try:
        movie_row = df[df['title'] == movie_title]
        if not movie_row.empty:
            movie_id = movie_row.iloc[0]['movie_id']
            # Fetch from TMDB API
            url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265a5b71962334024912a2507d71a4f"
            data = requests.get(url, timeout=3).json()
            poster_path = data.get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        pass
    # Fallback stylish poster placeholder
    return "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=500&auto=format&fit=crop"

# Load Datasets with Caching
@st.cache_data
def load_dataset():
    df = load_and_merge_tmdb_data()
    processed_df = prepare_recommender_data(df)
    return df, processed_df

# Sidebar Setup
st.sidebar.image("https://img.icons8.com/isometric/100/movie-projector.png", width=75)
st.sidebar.title("🎬 CineMatch Studio")
st.sidebar.markdown("---")
st.sidebar.markdown("**Engine:** Cosine Tag Similarity")
st.sidebar.markdown("**Source:** TMDB 5000 Movies")

with st.spinner("Initializing AI Neural Engine..."):
    df, processed_df = load_dataset()

st.sidebar.success(f"Indexed **{len(processed_df):,}** Movies")

# Glassmorphism Header
st.markdown("""
<div class="glass-header">
    <div class="main-glow-title">🎬 CineMatch Intelligence</div>
    <div class="sub-glow-text">Next-generation movie recommendation platform driven by Machine Learning.</div>
</div>
""", unsafe_allow_html=True)

# Selection Controls
col_select, col_btn = st.columns([3, 1])

movie_list = processed_df['title'].values

with col_select:
    selected_movie = st.selectbox(
        "Search or Select a Movie:",
        movie_list,
        help="Type any movie title to discover similar films"
    )

with col_btn:
    st.write(" ")
    st.write(" ")
    get_rec = st.button("✨ Get Recommendations")

# Display Recommendations Grid with Posters
if get_rec:
    recommendations = recommend(selected_movie, df, top_n=5)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### 🍿 Top 5 Recommendations for **'{selected_movie}'**")
    
    cols = st.columns(5)
    
    for idx, (col, movie) in enumerate(zip(cols, recommendations), 1):
        poster_url = fetch_poster(movie, df)
        with col:
            st.markdown(f"""
            <div class="movie-card">
                <div>
                    <span class="card-badge">Match #{idx}</span>
                    <img src="{poster_url}" class="poster-img" />
                    <div class="movie-title-text">{movie}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)