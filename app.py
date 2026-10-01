"""
=============================================================================
STREAMLIT WEB APPLICATION - MOVIE RECOMMENDATION SYSTEM
Personalized Movie Recommendations using Item-Item Collaborative Filtering
Full-width Centered UI (No Sidebar)
=============================================================================
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from recommender import MovieRecommender

# Set Streamlit Page Configuration with sidebar collapsed
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Dark Cinematic Styling (Netflix-inspired Full-Width Theme)
st.markdown("""
<style>
    /* Dark Theme Colors */
    :root {
        --bg-color: #141414;
        --card-bg: #1f1f1f;
        --card-border: #2e2e2e;
        --accent-red: #E50914;
        --accent-hover: #ff1f2d;
        --text-primary: #FFFFFF;
        --text-secondary: #AAAAAA;
        --gold: #FFD700;
    }
    
    /* Completely hide Streamlit sidebar and toggle button */
    [data-testid="stSidebar"],
    [data-testid="stSidebarNav"],
    [data-testid="stSidebarCollapsedControl"],
    section[data-testid="stSidebar"] {
        display: none !important;
        width: 0 !important;
        margin-left: -9999px !important;
    }
    
    /* Center and widen main content container */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-primary);
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    .block-container {
        max-width: 1240px !important;
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        margin: 0 auto !important;
    }
    
    /* Header & Hero Section */
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 0.3rem;
        text-shadow: 0 2px 12px rgba(229, 9, 20, 0.45);
        text-align: center;
    }
    .hero-title span {
        color: var(--accent-red);
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: var(--text-secondary);
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* Badge Pills Container */
    .badge-container {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 25px;
    }
    .badge-pill {
        display: inline-block;
        background: #222226;
        color: #e5e5e5;
        border: 1px solid #3d3d45;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.88rem;
        font-weight: 600;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
    }
    .badge-pill-accent {
        background: rgba(229, 9, 20, 0.18);
        color: #ff4d58;
        border: 1px solid rgba(229, 9, 20, 0.5);
    }
    
    /* Quick Benchmark Buttons Row */
    .benchmark-container {
        background: #18181d;
        border: 1px solid #2a2a33;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
    }
    .benchmark-label {
        font-size: 0.9rem;
        font-weight: 700;
        color: #e5e5e5;
        margin-right: 6px;
    }
    
    /* Selected Movie Spotlight Card */
    .spotlight-card {
        background: linear-gradient(135deg, #1e1e24 0%, #16161a 100%);
        border: 1px solid #ff3b47;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(229, 9, 20, 0.2);
    }
    .spotlight-title {
        font-size: 1.7rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 4px;
    }
    .spotlight-meta {
        color: #B3B3B3;
        font-size: 0.95rem;
    }
    
    /* Recommendation Card */
    .rec-card {
        background-color: #1a1a1e;
        border: 1px solid #2d2d35;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 18px;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .rec-card:hover {
        transform: translateY(-4px);
        border-color: #E50914;
        box-shadow: 0 6px 20px rgba(229, 9, 20, 0.3);
    }
    
    .rec-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 8px;
    }
    .rec-rank {
        background: #E50914;
        color: #fff;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 6px;
        text-transform: uppercase;
    }
    .rec-score {
        color: #00e676;
        font-weight: 700;
        font-size: 0.92rem;
    }
    .rec-title {
        font-size: 1.18rem;
        font-weight: 700;
        color: #ffffff;
        margin: 6px 0 4px 0;
        line-height: 1.3;
    }
    .rec-year {
        color: #8c8c9e;
        font-size: 0.88rem;
        margin-bottom: 10px;
    }
    .rec-explanation {
        font-size: 0.84rem;
        color: #a0a0b0;
        background: rgba(255, 255, 255, 0.04);
        padding: 9px 12px;
        border-radius: 6px;
        border-left: 3px solid #E50914;
        margin-top: 10px;
        line-height: 1.4;
    }
    
    /* Progress bar */
    .progress-bar-container {
        width: 100%;
        background-color: #2b2b36;
        border-radius: 10px;
        height: 7px;
        overflow: hidden;
        margin: 8px 0;
    }
    .progress-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #E50914, #ff5e62);
        border-radius: 10px;
    }
    
    /* Step Cards for How It Works */
    .step-card {
        background: #18181d;
        border: 1px solid #2a2a33;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 16px;
    }
    .step-number {
        font-size: 1.15rem;
        font-weight: 800;
        color: #E50914;
        margin-bottom: 4px;
    }
    .step-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
    }
    .step-desc {
        font-size: 0.9rem;
        color: #b0b0be;
        line-height: 1.45;
    }
    
    /* Primary Action Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #E50914 0%, #B81D24 100%);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.65rem 2rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 14px rgba(229, 9, 20, 0.4);
        transition: all 0.2s ease;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #ff1f2d 0%, #E50914 100%);
        box-shadow: 0 6px 20px rgba(229, 9, 20, 0.6);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_cached_recommender():
    """Load model artifacts once into cache for instant application loading."""
    rec = MovieRecommender()
    rec.load_artifacts()
    return rec


# Initialize recommender engine
try:
    recommender = get_cached_recommender()
    valid_movies_df = recommender.get_valid_movies()
    meta = recommender.metadata
except Exception as e:
    st.error(f"❌ Failed to load model artifacts: {e}")
    st.info("💡 Please execute `python train_model.py` first to generate model artifacts.")
    st.stop()


# =============================================================================
# MAIN HEADER & BADGES (CENTERED FULL-WIDTH)
# =============================================================================

st.markdown('<div class="hero-title">🎬 Movie <span>Recommendation</span> System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Personalized movie recommendations using Item-Item Collaborative Filtering on the Netflix Dataset</div>',
    unsafe_allow_html=True
)

# Badges Row (All 5 Required Badges Preserved)
st.markdown(
    f"""
    <div class="badge-container">
        <span class="badge-pill badge-pill-accent">⚡ Item-Item Collaborative Filtering</span>
        <span class="badge-pill">📐 Cosine Similarity</span>
        <span class="badge-pill">💾 17.3M+ Netflix Ratings</span>
        <span class="badge-pill">🎯 Precision@10: {meta.get('precision_at_10', 0.3551)*100:.1f}%</span>
        <span class="badge-pill">🔒 8 GB RAM Optimized (Sparse CSR)</span>
    </div>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# MAIN TABS (DISCOVER, USER RECS, EDA, HOW IT WORKS)
# =============================================================================
tab_discover, tab_user, tab_eda, tab_how = st.tabs([
    "🍿 Discover Movies (Item-Item CF)",
    "👤 User Personalized Recs",
    "📊 Dataset Statistics & EDA",
    "🧠 How It Works"
])


# =============================================================================
# TAB 1: DISCOVER MOVIES (ITEM-ITEM COLLABORATIVE FILTERING)
# =============================================================================
with tab_discover:
    st.markdown("##### 🔍 **Find Movies with Similar Rating Patterns**")
    st.write(
        "Select any movie from the verified Netflix model catalog or search by title. "
        "The model calculates pairwise cosine similarity over high-dimensional user-interaction vectors."
    )
    
    # Format options as: "Title (Year)"
    valid_movies_df["Display_Name"] = valid_movies_df.apply(
        lambda r: f"{r['Name']} ({int(r['Year']) if pd.notnull(r['Year']) and r['Year'] > 0 else 'N/A'})",
        axis=1
    )
    
    movie_options = valid_movies_df["Display_Name"].tolist()
    name_to_id = dict(zip(valid_movies_df["Display_Name"], valid_movies_df["Movie_ID"]))
    
    # Quick Benchmark Buttons (Conveniently placed on main page)
    st.markdown("**🔥 Quick Popular Benchmark Titles:**")
    bench_cols = st.columns(6)
    benchmark_titles = [
        ("The Silence of the Lambs", 0),
        ("Shrek 2", 1),
        ("Pirates of the Caribbean", 2),
        ("Lord of the Rings", 3),
        ("American Beauty", 4),
        ("Reservoir Dogs", 5)
    ]
    for b_title, c_idx in benchmark_titles:
        with bench_cols[c_idx]:
            if st.button(f"🎥 {b_title}", key=f"btn_{b_title}", use_container_width=True):
                st.session_state["selected_movie_title"] = b_title
    
    # Check default selected movie from session state
    default_index = 0
    if "selected_movie_title" in st.session_state:
        target_name = st.session_state["selected_movie_title"].lower()
        for idx, option in enumerate(movie_options):
            if target_name in option.lower():
                default_index = idx
                break
                
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    col_input1, col_input2 = st.columns([3, 1])
    with col_input1:
        selected_display = st.selectbox(
            "Select Movie from Trained Model Catalog:",
            options=movie_options,
            index=default_index,
            help="Contains all movies that exist in the trained Collaborative Filtering model."
        )
    with col_input2:
        num_recs = st.slider("Recommendations count:", min_value=5, max_value=20, value=10, step=1)
        
    # Free-text manual search expander for testing out-of-model or custom queries
    with st.expander("🔎 Or search catalog with custom keywords (test out-of-model handling)"):
        custom_query = st.text_input(
            "Enter movie title to search:",
            placeholder="e.g. Lord of the Rings, Titanic, Star Wars..."
        )
        search_clicked = st.button("Search Catalog Keyword")
        if search_clicked and custom_query:
            search_results = recommender.search_movie(custom_query, limit=5)
            if search_results:
                st.write("**Matching titles found:**")
                for sr in search_results:
                    in_model_badge = "✅ In Model" if sr["In_Model"] else "⚠️ No Interactions in Model"
                    st.write(f"- **{sr['Title']}** ({sr['Year']}) - {in_model_badge}")
            else:
                st.warning(f"No movie found matching '{custom_query}'.")

    # Recommendation Trigger Button
    recommend_clicked = st.button("✨ Recommend Movies", use_container_width=True)

    if recommend_clicked or "selected_movie_title" in st.session_state:
        chosen_movie_id = name_to_id.get(selected_display)
        
        with st.spinner("Finding movies with similar user-rating patterns..."):
            result = recommender.recommend_movies(movie_id=chosen_movie_id, n=num_recs)
            
        if result["status"] == "insufficient_data":
            st.error("⚠️ **This movie does not have enough interaction data for a reliable recommendation.**")
        elif result["status"] == "not_found":
            st.warning("⚠️ Movie not found in the catalog.")
        elif result["status"] == "success":
            sel = result["selected_movie"]
            recs = result["recommendations"]
            
            # Selected Movie Spotlight Card
            avg_star = "⭐" * int(round(sel.get("Avg_Rating", 3.5)))
            st.markdown(
                f"""
                <div class="spotlight-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div>
                            <span class="badge-pill badge-pill-accent">Selected Movie</span>
                            <div class="spotlight-title">{sel['Title']}</div>
                            <div class="spotlight-meta">
                                📅 Release Year: <b>{sel['Year']}</b> &nbsp;|&nbsp; 
                                🆔 Movie ID: <b>{sel['Movie_ID']}</b> &nbsp;|&nbsp; 
                                👥 Total Ratings: <b>{sel.get('Rating_Count', 0):,}</b> &nbsp;|&nbsp; 
                                ⭐ Avg Rating: <b>{sel.get('Avg_Rating', 0.0):.2f}/5</b> {avg_star}
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown(f"#### 🏆 **Top {len(recs)} Recommended Movies for You**")
            st.caption("Ranked by Cosine Similarity across 143,458 Netflix user rating profiles:")
            
            # Display recommendations in a responsive 2-column grid
            col_left, col_right = st.columns(2)
            
            for idx, r in enumerate(recs):
                target_col = col_left if idx % 2 == 0 else col_right
                match_pct = int(round(r["Similarity"] * 100))
                
                with target_col:
                    st.markdown(
                        f"""
                        <div class="rec-card">
                            <div>
                                <div class="rec-header">
                                    <span class="rec-rank">Rank #{r['Rank']}</span>
                                    <span class="rec-score">⚡ {r['Similarity']:.4f} ({match_pct}% Match)</span>
                                </div>
                                <div class="rec-title">{r['Title']}</div>
                                <div class="rec-year">📅 Release: {r['Year']} &nbsp;•&nbsp; 👥 {r.get('Rating_Count', 0):,} ratings &nbsp;•&nbsp; ⭐ {r.get('Avg_Rating', 0.0):.1f}/5</div>
                                <div class="progress-bar-container">
                                    <div class="progress-bar-fill" style="width: {match_pct}%;"></div>
                                </div>
                            </div>
                            <div class="rec-explanation">
                                💡 <b>Why this movie?</b><br>
                                {r['Explanation']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# =============================================================================
# TAB 2: USER PERSONALIZED RECOMMENDATIONS (USER PROFILE EXTENSION)
# =============================================================================
with tab_user:
    st.markdown("##### 👤 **Personalized Recommendations for User Profile**")
    st.write(
        "Demonstrating the **User-Based Extension**: this module retrieves an active user's rating history, "
        "identifies their highly-rated movies (rating ≥ 4.0), aggregates the item-item similarity vectors, "
        "and recommends top movies the user has *not yet watched*."
    )
    
    available_users = list(recommender.sample_users.keys())
    if available_users:
        col_u1, col_u2 = st.columns([2, 1])
        with col_u1:
            selected_uid = st.selectbox(
                "Select a sample active Netflix User ID:",
                options=available_users,
                format_func=lambda uid: f"User ID: {uid} ({len(recommender.sample_users[uid])} rated movies in model)"
            )
        with col_u2:
            num_user_recs = st.slider("User recommendations count:", 5, 15, 10, key="slider_user_recs")
            
        user_rec_btn = st.button("🚀 Generate User Recommendations", use_container_width=True)
        
        if user_rec_btn:
            user_result = recommender.recommend_for_user(selected_uid, n=num_user_recs)
            
            if user_result["status"] == "success":
                # Display user profile summary
                st.markdown("---")
                st.markdown(f"#### 🎬 User {selected_uid}'s Favorite Movies (Rated ≥ 4.0 Stars)")
                st.caption("These highly-rated films serve as the anchor for the user's taste profile:")
                
                liked_cols = st.columns(min(4, len(user_result["liked_movies"])))
                for idx, lm in enumerate(user_result["liked_movies"][:4]):
                    with liked_cols[idx % len(liked_cols)]:
                        st.markdown(
                            f"""
                            <div style="background: #1b1b22; border-left: 3px solid #FFD700; padding: 10px 12px; border-radius: 6px; margin-bottom: 10px;">
                                <div style="font-weight: 700; font-size: 0.95rem; color: #fff;">{lm['Title']}</div>
                                <div style="color: #888; font-size: 0.8rem;">📅 {lm['Year']} &nbsp;|&nbsp; Rated: ⭐ <b>{lm['Rating']:.1f}</b></div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                st.markdown(f"#### 🎯 Recommended for User {selected_uid} (Excluding Already Watched)")
                u_col1, u_col2 = st.columns(2)
                for idx, ur in enumerate(user_result["recommendations"]):
                    target_col = u_col1 if idx % 2 == 0 else u_col2
                    match_pct = int(round(ur["Similarity"] * 100))
                    with target_col:
                        st.markdown(
                            f"""
                            <div class="rec-card">
                                <div>
                                    <div class="rec-header">
                                        <span class="rec-rank">Top Pick #{ur['Rank']}</span>
                                        <span class="rec-score">Aggregated Affinity: {ur['Similarity']:.4f}</span>
                                    </div>
                                    <div class="rec-title">{ur['Title']}</div>
                                    <div class="rec-year">📅 Release: {ur['Year']} &nbsp;•&nbsp; 👥 {ur.get('Rating_Count', 0):,} ratings &nbsp;•&nbsp; ⭐ {ur.get('Avg_Rating', 0.0):.1f}/5</div>
                                    <div class="progress-bar-container">
                                        <div class="progress-bar-fill" style="width: {match_pct}%;"></div>
                                    </div>
                                </div>
                                <div class="rec-explanation">
                                    💡 <b>Recommendation Reason:</b><br>
                                    {ur['Explanation']}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
    else:
        st.info("Sample user profiles were not found. Run `python train_model.py` to generate profile artifacts.")


# =============================================================================
# TAB 3: DATASET STATISTICS & EDA
# =============================================================================
with tab_eda:
    st.markdown("##### 📊 **Exploratory Data Analysis & System Metrics**")
    st.write(
        "Overview of the official Netflix Prize dataset and the sparse matrix representation."
    )
    
    # 4 Key metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Ratings", f"{meta.get('total_ratings', 17337458):,}", "100% Valid (1.0 - 5.0)")
    m2.metric("Active Users", f"{meta.get('num_users', 143458):,}", "Evaluated Sample")
    m3.metric("Curated Model Movies", f"{meta.get('num_movies', 1350):,}", "17,770 Total Catalog")
    m4.metric("Precision@10 Score", f"{meta.get('precision_at_10', 0.3551)*100:.2f}%", f"Over {meta.get('eval_users_count', 499)} Test Users")
    
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("###### 📈 **Top 10 Most-Rated Movies in Model**")
        top10_df = valid_movies_df.sort_values(by="Rating_Count", ascending=False).head(10)
        chart_data = top10_df[["Name", "Rating_Count"]].set_index("Name")
        st.bar_chart(chart_data, color="#E50914")
        
    with col_chart2:
        st.markdown("###### 📐 **Sparse Matrix & Efficiency Highlights**")
        st.markdown(
            f"""
            - **User-Movie Interaction Space:** `143,458 users × 1,350 movies`
            - **Potential Dense Matrix Cells:** `193,668,300` (~1.5 GB dense float64)
            - **Non-Zero Interactions:** `17,337,458`
            - **Interaction Sparsity:** `{meta.get('sparsity', 91.05):.2f}%`
            - **Sparse CSR Memory:** `~132.8 MB` *(Over 90% memory savings!)*
            - **Movie Similarity Matrix:** `1,350 × 1,350` (`~6.95 MB` float32)
            - **Inference Latency:** `< 5 ms` per recommendation query
            """
        )
        
    st.markdown("---")
    st.markdown("###### 🔍 **Model Evaluation Details (Precision@K)**")
    st.info(
        f"""
        **Evaluation Methodology:**
        - **Sample Size:** {meta.get('eval_users_count', 499)} active users with ≥ 20 ratings.
        - **Hold-out Strategy:** For each user, 30% of their highly-rated movies (rating ≥ 4.0) were held out as ground truth.
        - **Precision@10 Metric:** Evaluates the proportion of the top 10 recommended items that belong to the user's held-out relevant set.
        - **Observed Precision@10:** **`{meta.get('precision_at_10', 0.3551)*100:.2f}%`**
        - *Academic Note:* In collaborative filtering, precision is calculated against held-out ratings. A non-held-out recommendation may still be highly enjoyable to the user even if unrated in the training slice.
        """
    )


# =============================================================================
# TAB 4: HOW IT WORKS
# =============================================================================
with tab_how:
    st.markdown("##### 🧠 **Algorithm Walkthrough: Item-Item Collaborative Filtering**")
    st.write(
        "Collaborative Filtering relies on the principle that if users agree on the evaluation of certain items, "
        "they are likely to agree on other items as well."
    )
    
    st.markdown(
        """
        <div class="step-card">
            <div class="step-number">STEP 1</div>
            <div class="step-title">📥 User Ratings Data Collection</div>
            <div class="step-desc">
                The system ingests 17.3M user rating interactions from the Netflix dataset. Each record consists of 
                <code>(User_ID, Movie_ID, Rating)</code>, capturing explicit user preference on a 1.0 to 5.0 star scale.
            </div>
        </div>
        
        <div class="step-card">
            <div class="step-number">STEP 2</div>
            <div class="step-title">🧩 Sparse User-Movie Matrix Construction</div>
            <div class="step-desc">
                To prevent memory exhaustion on an 8 GB RAM system, ratings are encoded into a Compressed Sparse Row 
                (<code>scipy.sparse.csr_matrix</code>). Rows index 143,458 users and columns index 1,350 movies. 
                With 91.05% sparsity, the sparse representation compresses 193.6M cells into just ~132.8 MB of RAM.
            </div>
        </div>
        
        <div class="step-card">
            <div class="step-number">STEP 3</div>
            <div class="step-title">📐 Pairwise Cosine Similarity Computation</div>
            <div class="step-desc">
                Each movie is represented by its column vector of user ratings across all 143k users. 
                The cosine similarity between Movie <i>A</i> and Movie <i>B</i> evaluates the cosine of the angle between their user-rating vectors:
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.latex(r"""
    \text{Cosine Similarity}(A, B) = \frac{\mathbf{r}_A \cdot \mathbf{r}_B}{\|\mathbf{r}_A\|_2 \|\mathbf{r}_B\|_2} = \frac{\sum_{u \in U} R_{u, A} \cdot R_{u, B}}{\sqrt{\sum_{u \in U} R_{u, A}^2} \cdot \sqrt{\sum_{u \in U} R_{u, B}^2}}
    """)
    
    st.markdown(
        """
        <div class="step-card">
            <div class="step-number">STEP 4</div>
            <div class="step-title">🎯 Real-time Recommendation & Deduplication</div>
            <div class="step-desc">
                When a user queries Movie <i>M</i>:
                <ol>
                    <li>Resolve Movie_ID and verify its presence in the trained model.</li>
                    <li>Extract row <i>M</i> from the pre-computed similarity matrix.</li>
                    <li>Sort scores in descending order.</li>
                    <li>Filter out the query movie itself and remove duplicate/re-release titles.</li>
                    <li>Return the Top 10 most similar movies along with similarity scores and match percentages.</li>
                </ol>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    st.markdown("##### 🎓 **Viva / Interview Presentation Summary**")
    st.markdown(
        """
        - **Why Item-Item over User-User CF?** 
          Items (1,350 movies) are much more stable and fewer in count than users (143,458 users). 
          Computing and storing a 1,350 × 1,350 similarity matrix takes only ~7 MB, whereas a 143k × 143k user matrix would require over 80 GB of RAM.
        - **Why Sparse Matrix?** 
          Prevents Out-Of-Memory (OOM) errors and accelerates matrix arithmetic by computing only on non-zero rating elements.
        - **Why Cosine Similarity?** 
          Normalizes for vector magnitude, effectively handling differences in overall rating volume between blockbuster and niche films.
        """
    )
