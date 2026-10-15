import streamlit as st
from src.data_loader import load_and_merge_tmdb_data
from src.recommender import recommend, prepare_recommender_data

# Page Configuration
st.set_page_config(
    page_title="CineMatch | Modern Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI styling (using unsafe_allow_html=True)
st.markdown("""
<style>
    /* Background and Global Styles */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Header Container Styling */
    .header-container {
        background: linear-gradient(135deg, #1e1e2f 0%, #11111d 100%);
        padding: 2.5rem;
        border-radius: 16px;
        border: 1px solid #2d2d44;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff4b4b 0%, #ff8c42 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        color: #a0a0b8;
        font-size: 1.1rem;
        margin-bottom: 0;
    }

    /* Card Styling for Recommendations */
    .rec-card {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #282846;
        transition: transform 0.2s ease, border-color 0.2s ease;
        text-align: center;
        margin-bottom: 1rem;
        height: 100%;
    }
    
    .rec-card:hover {
        transform: translateY(-5px);
        border-color: #ff4b4b;
    }

    .card-badge {
        background: rgba(255, 75, 75, 0.15);
        color: #ff4b4b;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }

    .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 5px;
    }

    /* Customizing Primary Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #ff4b4b 0%, #ff6b6b 100%);
        color: white;
        font-weight: 700;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        border: none;
        box-shadow: 0 4px 14px 0 rgba(255, 75, 75, 0.39);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #ff6b6b 0%, #ff4b4b 100%);
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading for performance
@st.cache_data
def load_data():
    df = load_and_merge_tmdb_data()
    processed_df = prepare_recommender_data(df)
    return df, processed_df

# Sidebar Dashboard Info
st.sidebar.image("https://img.icons8.com/isometric/100/movie-projector.png", width=80)
st.sidebar.title("🎬 CineMatch AI")
st.sidebar.write("Content-based recommendation engine powered by TMDB metadata.")

# Load Datasets
with st.spinner("Initializing Recommendation Engine..."):
    df, processed_df = load_data()

st.sidebar.success(f"Successfully indexed **{len(processed_df)}** movies!")

# Main Header
st.markdown("""
<div class="header-container">
    <div class="main-title">🎬 CineMatch Movie Intelligence</div>
    <div class="sub-title">Discover your next favorite movie powered by machine learning tag similarity.</div>
</div>
""", unsafe_allow_html=True)

# Control Section
col_select, col_btn = st.columns([3, 1])

movie_list = processed_df['title'].values

with col_select:
    selected_movie = st.selectbox(
        "Search or Select a Movie:",
        movie_list,
        help="Type or choose a movie to discover similar recommendations"
    )

with col_btn:
    st.write(" ") # Spacing alignment
    st.write(" ")
    get_rec = st.button("✨ Recommend")

# Display Results
if get_rec:
    recommendations = recommend(selected_movie, df, top_n=5)
    
    st.markdown("---")
    st.subheader(f"🍿 Top Recommendations for **'{selected_movie}'**")
    
    # Display recommendations side-by-side in styled cards
    cols = st.columns(5)
    
    for idx, (col, movie) in enumerate(zip(cols, recommendations), 1):
        with col:
            st.markdown(f"""
            <div class="rec-card">
                <span class="card-badge">Match #{idx}</span>
                <div class="card-title">{movie}</div>
            </div>
            """, unsafe_allow_html=True)