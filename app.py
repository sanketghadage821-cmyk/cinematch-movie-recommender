import streamlit as st
from src.data_loader import load_and_merge_tmdb_data
from src.recommender import recommend, prepare_recommender_data

# Page Configuration
st.set_page_config(
    page_title="CineMatch AI | Premium Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Modern CSS Customization
st.markdown("""
<style>
    /* Dark Gradient App Background */
    .stApp {
        background: linear-gradient(135deg, #0b0d17 0%, #121629 50%, #080a12 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero Section Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        backdrop-filter: blur(12px);
        padding: 2.8rem 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
    }
    
    .hero-title {
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #ff4b4b 0%, #ff8a00 50%, #e52e71 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.6rem;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        font-weight: 400;
        max-width: 650px;
        margin: 0 auto;
    }

    /* Movie Recommendation Cards */
    .card-container {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border-radius: 16px;
        padding: 1.5rem 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .card-container:hover {
        transform: translateY(-8px);
        border-color: #ff4b4b;
        box-shadow: 0 15px 35px rgba(255, 75, 75, 0.25);
    }

    .match-rank {
        background: linear-gradient(90deg, #ff4b4b 0%, #ff8a00 100%);
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 800;
        padding: 4px 12px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: inline-block;
        margin-bottom: 0.8rem;
    }

    .movie-title-text {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.4;
        margin-top: 0.4rem;
    }

    /* Customizing Streamlit Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #ff4b4b 0%, #e52e71 100%);
        color: #ffffff;
        font-size: 1.05rem;
        font-weight: 700;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        border: none;
        box-shadow: 0 8px 20px rgba(255, 75, 75, 0.35);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #e52e71 0%, #ff4b4b 100%);
        box-shadow: 0 12px 28px rgba(255, 75, 75, 0.5);
        transform: translateY(-2px);
    }

    /* Selectbox Input Customization */
    .stSelectbox label {
        color: #cbd5e1 !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# Data Loader with Caching
@st.cache_data
def load_dataset():
    df = load_and_merge_tmdb_data()
    processed_df = prepare_recommender_data(df)
    return df, processed_df

# Sidebar Dashboard Section
st.sidebar.image("https://img.icons8.com/isometric/100/movie-projector.png", width=75)
st.sidebar.title("🎬 CineMatch AI")
st.sidebar.markdown("---")
st.sidebar.write("**Dataset Engine:** TMDB 5000 Movies")
st.sidebar.write("**Model:** Cosine Similarity Vectorizer")

with st.spinner("Initializing Recommendation Engine..."):
    df, processed_df = load_dataset()

st.sidebar.success(f"Indexed **{len(processed_df):,}** Movies")

# Hero Header Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🎬 CineMatch Movie Intelligence</div>
    <div class="hero-subtitle">Discover tailored movie recommendations computed using metadata vector analysis.</div>
</div>
""", unsafe_allow_html=True)

# Main Controls Layout
col_select, col_btn = st.columns([3, 1])

movie_list = processed_df['title'].values

with col_select:
    selected_movie = st.selectbox(
        "Search or Select a Movie:",
        movie_list,
        help="Type or select a title to find similar recommendations"
    )

with col_btn:
    st.write(" ")  # Alignment spacing
    st.write(" ")
    get_rec = st.button("✨ Get Recommendations")

# Recommendation Output Display
if get_rec:
    recommendations = recommend(selected_movie, df, top_n=5)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### 🍿 Top Recommendations for **'{selected_movie}'**")
    
    # 5-Column Grid Layout for Movie Cards
    cols = st.columns(5)
    
    for idx, (col, movie) in enumerate(zip(cols, recommendations), 1):
        with col:
            st.markdown(f"""
            <div class="card-container">
                <div>
                    <span class="match-rank">Match #{idx}</span>
                    <div class="movie-title-text">{movie}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)