"""
=============================================================================
MOVIE RECOMMENDATION SYSTEM USING ITEM-ITEM COLLABORATIVE FILTERING
Training, Matrix Construction, Cosine Similarity Computation, Evaluation & Artifact Generation
=============================================================================
"""

import os
import sys
import time
import zipfile
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity


def print_banner(text):
    print("\n" + "=" * 75)
    print(f"  {text}")
    print("=" * 75)


def locate_file(filename, search_dirs=None):
    """Locate dataset file in current, data/, or parent directories."""
    if search_dirs is None:
        search_dirs = [
            "data",
            ".",
            "..",
            os.path.expanduser(r"~\OneDrive\Documents\Desktop"),
            os.path.expanduser(r"~\Desktop"),
        ]
    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def ensure_data_extracted():
    """Locate and extract Netflix dataset if required."""
    print_banner("STEP 1: DATASET DISCOVERY & EXTRACTION")
    
    # Check data directory
    os.makedirs("data", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    
    # 1. Locate Movie Dataset
    movie_csv_path = locate_file("Netflix_Dataset_Movie.csv")
    if not movie_csv_path:
        raise FileNotFoundError(
            "Could not locate 'Netflix_Dataset_Movie.csv'. Please place it in data/ or root directory."
        )
    print(f"[+] Found Movie Dataset: {movie_csv_path}")
    
    # Copy to data/ if not already there
    target_movie_path = os.path.abspath("data/Netflix_Dataset_Movie.csv")
    if movie_csv_path != target_movie_path and not os.path.exists(target_movie_path):
        import shutil
        shutil.copyfile(movie_csv_path, target_movie_path)
        movie_csv_path = target_movie_path
    
    # 2. Locate Rating Dataset (CSV or ZIP)
    rating_csv_path = locate_file("Netflix_Dataset_Rating.csv")
    rating_zip_path = locate_file("Netflix_Dataset_Rating.csv.zip")
    
    if rating_csv_path and os.path.exists(rating_csv_path):
        print(f"[+] Found uncompressed Rating CSV: {rating_csv_path}")
        final_rating_path = rating_csv_path
    elif rating_zip_path and os.path.exists(rating_zip_path):
        print(f"[+] Found Rating ZIP archive: {rating_zip_path}")
        print("[*] Extracting 'Netflix_Dataset_Rating.csv' into 'data/' directory...")
        t0 = time.time()
        with zipfile.ZipFile(rating_zip_path, "r") as z:
            z.extract("Netflix_Dataset_Rating.csv", path="data")
        final_rating_path = os.path.abspath("data/Netflix_Dataset_Rating.csv")
        print(f"[+] Extracted successfully in {time.time() - t0:.2f} seconds.")
    else:
        raise FileNotFoundError(
            "Could not locate 'Netflix_Dataset_Rating.csv' or 'Netflix_Dataset_Rating.csv.zip'."
        )
        
    return movie_csv_path, final_rating_path


def load_and_preprocess(movie_path, rating_path):
    """Load, inspect, clean, and validate movie and rating datasets."""
    print_banner("STEP 2: DATA LOADING & PREPROCESSING (MEMORY-OPTIMIZED)")
    
    # --- 1. Load Movie Catalog ---
    t0 = time.time()
    movie_df = pd.read_csv(movie_path, encoding="latin-1")
    # Normalize column names (strip whitespace)
    movie_df.columns = [c.strip() for c in movie_df.columns]
    
    # Ensure standard column names
    col_map = {}
    for col in movie_df.columns:
        low = col.lower()
        if "id" in low:
            col_map[col] = "Movie_ID"
        elif "year" in low:
            col_map[col] = "Year"
        elif "name" in low or "title" in low:
            col_map[col] = "Name"
    movie_df = movie_df.rename(columns=col_map)
    
    # Handle missing values in movies
    movie_nulls = movie_df.isnull().sum().to_dict()
    movie_df["Year"] = pd.to_numeric(movie_df["Year"], errors="coerce").fillna(0).astype(int)
    movie_df["Name"] = movie_df["Name"].fillna("Unknown Title").astype(str).str.strip()
    movie_df["Movie_ID"] = movie_df["Movie_ID"].astype(np.int32)
    
    print(f"[+] Movie Catalog Loaded: {len(movie_df):,} movies in {time.time() - t0:.2f}s")
    print(f"    Columns: {list(movie_df.columns)}")
    print(f"    Null values detected: {movie_nulls}")
    
    # --- 2. Load Ratings with Optimized Dtypes for 8 GB RAM ---
    print("\n[*] Loading Rating Dataset (17M+ rows) with memory-safe dtypes...")
    t1 = time.time()
    # int32 for IDs (up to 2 billion), float32 for Ratings (4 bytes)
    dtypes = {"User_ID": np.int32, "Rating": np.float32, "Movie_ID": np.int32}
    
    rating_df = pd.read_csv(
        rating_path,
        dtype=dtypes,
        usecols=["User_ID", "Rating", "Movie_ID"]
    )
    # Normalize columns
    rating_df.columns = [c.strip() for c in rating_df.columns]
    
    load_time = time.time() - t1
    print(f"[+] Ratings Loaded: {len(rating_df):,} rows in {load_time:.2f}s")
    print(f"    Memory footprint: {rating_df.memory_usage().sum() / (1024**2):.2f} MB")
    
    # --- 3. Cleaning & Validation ---
    print("\n[*] Preprocessing & Integrity Checks:")
    initial_count = len(rating_df)
    
    # Remove nulls if any
    rating_df = rating_df.dropna(subset=["User_ID", "Movie_ID", "Rating"])
    null_dropped = initial_count - len(rating_df)
    print(f"    - Null ratings dropped: {null_dropped}")
    
    # Remove duplicate (User_ID, Movie_ID) pairs if any
    pre_dup_count = len(rating_df)
    rating_df = rating_df.drop_duplicates(subset=["User_ID", "Movie_ID"])
    dup_dropped = pre_dup_count - len(rating_df)
    print(f"    - Duplicate ratings dropped: {dup_dropped}")
    
    # Filter valid rating range (1.0 to 5.0)
    rating_df = rating_df[(rating_df["Rating"] >= 1.0) & (rating_df["Rating"] <= 5.0)]
    
    # Rating statistics
    stats = rating_df["Rating"].describe()
    print(f"    - Valid Rating Summary: Min={stats['min']:.1f}, Max={stats['max']:.1f}, Mean={stats['mean']:.2f}, Std={stats['std']:.2f}")
    
    num_users = rating_df["User_ID"].nunique()
    num_movies = rating_df["Movie_ID"].nunique()
    print(f"    - Unique Users: {num_users:,}")
    print(f"    - Unique Movies with Ratings: {num_movies:,}")
    
    # Verify ID match with Movie catalog
    rating_mids = set(rating_df["Movie_ID"].unique())
    catalog_mids = set(movie_df["Movie_ID"].unique())
    matched_mids = rating_mids.intersection(catalog_mids)
    print(f"    - Movie Catalog Overlap: {len(matched_mids):,} / {len(rating_mids):,} ({len(matched_mids)/len(rating_mids)*100:.1f}%)")
    
    return movie_df, rating_df


def build_sparse_interaction_matrix(rating_df):
    """
    Construct memory-efficient scipy.sparse.csr_matrix.
    Rows: Users, Columns: Movies.
    """
    print_banner("STEP 3: SPARSE USER-MOVIE INTERACTION MATRIX")
    t0 = time.time()
    
    # Map raw IDs to contiguous zero-based matrix indices
    unique_users = np.sort(rating_df["User_ID"].unique())
    unique_movies = np.sort(rating_df["Movie_ID"].unique())
    
    user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    movie_to_idx = {mid: i for i, mid in enumerate(unique_movies)}
    idx_to_movie = {i: mid for mid, i in movie_to_idx.items()}
    
    print(f"[*] Mapping {len(unique_users):,} users and {len(unique_movies):,} movies to sparse indices...")
    row_indices = rating_df["User_ID"].map(user_to_idx).values
    col_indices = rating_df["Movie_ID"].map(movie_to_idx).values
    ratings_data = rating_df["Rating"].values
    
    num_users = len(unique_users)
    num_movies = len(unique_movies)
    num_ratings = len(ratings_data)
    
    print(f"[*] Assembling scipy.sparse.csr_matrix of shape ({num_users:,}, {num_movies:,})...")
    user_item_matrix = sp.csr_matrix(
        (ratings_data, (row_indices, col_indices)),
        shape=(num_users, num_movies),
        dtype=np.float32
    )
    
    total_possible_entries = num_users * num_movies
    sparsity = (1.0 - (num_ratings / total_possible_entries)) * 100.0
    csr_mem_mb = (user_item_matrix.data.nbytes + user_item_matrix.indices.nbytes + user_item_matrix.indptr.nbytes) / (1024**2)
    
    print(f"[+] Sparse Matrix Built in {time.time() - t0:.2f}s:")
    print(f"    - Dimensions (Users x Movies): {num_users:,} x {num_movies:,}")
    print(f"    - Non-Zero Interactions (Ratings): {num_ratings:,}")
    print(f"    - Matrix Density: {100.0 - sparsity:.2f}%")
    print(f"    - Matrix Sparsity: {sparsity:.2f}%")
    print(f"    - CSR Memory Footprint: {csr_mem_mb:.2f} MB")
    
    return user_item_matrix, unique_users, unique_movies, user_to_idx, movie_to_idx, idx_to_movie, sparsity


def compute_item_similarity(user_item_matrix):
    """
    Calculate Item-Item Cosine Similarity.
    Item vectors are the columns of user_item_matrix (rows of user_item_matrix.T).
    """
    print_banner("STEP 4: ITEM-ITEM COSINE SIMILARITY COMPUTATION")
    t0 = time.time()
    
    # Transpose to get movie representations across all users
    # Shape: (num_movies, num_users)
    movie_user_matrix = user_item_matrix.T.tocsr()
    print(f"[*] Movie-User representation matrix shape: {movie_user_matrix.shape}")
    print(f"[*] Calculating pairwise Cosine Similarity between all {movie_user_matrix.shape[0]:,} movies...")
    
    # Cosine similarity on sparse movie-user matrix
    similarity_matrix = cosine_similarity(movie_user_matrix, dense_output=True)
    
    # Ensure float32 for memory efficiency
    similarity_matrix = similarity_matrix.astype(np.float32)
    sim_mem_mb = similarity_matrix.nbytes / (1024**2)
    
    print(f"[+] Movie-Movie Similarity Matrix computed in {time.time() - t0:.2f}s:")
    print(f"    - Similarity Matrix Shape: {similarity_matrix.shape}")
    print(f"    - Memory Size: {sim_mem_mb:.2f} MB")
    print(f"    - Min Similarity: {similarity_matrix.min():.4f}")
    print(f"    - Max Similarity: {similarity_matrix.max():.4f}")
    print(f"    - Mean Similarity: {similarity_matrix.mean():.4f}")
    
    return similarity_matrix


def evaluate_precision_at_k(user_item_matrix, similarity_matrix, k=10, sample_size=500):
    """
    Precision@K Evaluation on a representative sample of active users.
    Hold out high-rated movies (rating >= 4.0) and evaluate top-K recommendations.
    """
    print_banner(f"STEP 5: MODEL EVALUATION (PRECISION@{k})")
    print(f"[*] Selecting sample of active users with >= 20 ratings (sample size = {sample_size})...")
    
    # Number of ratings per user
    user_counts = np.diff(user_item_matrix.indptr)
    active_users = np.where(user_counts >= 20)[0]
    
    if len(active_users) == 0:
        active_users = np.where(user_counts >= 10)[0]
    
    np.random.seed(42)
    eval_user_indices = np.random.choice(
        active_users,
        size=min(sample_size, len(active_users)),
        replace=False
    )
    
    precisions = []
    evaluated_users_count = 0
    
    for u_idx in eval_user_indices:
        start_idx = user_item_matrix.indptr[u_idx]
        end_idx = user_item_matrix.indptr[u_idx + 1]
        
        u_movies = user_item_matrix.indices[start_idx:end_idx]
        u_ratings = user_item_matrix.data[start_idx:end_idx]
        
        # Movies user liked (rating >= 4.0)
        high_mask = u_ratings >= 4.0
        relevant_movies = set(u_movies[high_mask])
        
        # Need at least 5 relevant movies for meaningful hold-out
        if len(relevant_movies) < 5:
            continue
            
        relevant_list = list(relevant_movies)
        # Hold out 30% of relevant movies as test ground truth
        holdout_count = max(1, int(len(relevant_list) * 0.3))
        test_target = set(relevant_list[:holdout_count])
        
        # Input training profile for this user
        train_input = [m for m in u_movies if m not in test_target]
        train_ratings = [r for m, r in zip(u_movies, u_ratings) if m not in test_target]
        
        liked_input = [m for m, r in zip(train_input, train_ratings) if r >= 4.0]
        if not liked_input:
            continue
            
        # Aggregate similarity scores of liked input movies
        agg_scores = similarity_matrix[liked_input].mean(axis=0)
        
        # Mask out movies the user already rated in train input
        agg_scores[train_input] = -1.0
        
        # Top-K recommended movie indices
        top_k_indices = set(np.argsort(-agg_scores)[:k])
        
        # Hits: intersection of recommended items with held-out relevant items
        hits = len(top_k_indices.intersection(test_target))
        precisions.append(hits / float(k))
        evaluated_users_count += 1
        
    avg_precision = float(np.mean(precisions)) if precisions else 0.0
    print(f"[+] Evaluation Completed Successfully:")
    print(f"    - Number of Evaluation Users: {evaluated_users_count}")
    print(f"    - Metric: Precision@{k} (Hold-out ratio = 30%, Relevancy threshold >= 4.0)")
    print(f"    - Precision@{k} Score: {avg_precision:.4f} ({avg_precision * 100:.2f}%)")
    print(f"    - Note: In Collaborative Filtering, Precision@{k} reflects the proportion")
    print(f"      of top-{k} recommendations that match known ground-truth user preferences.")
    
    return avg_precision, evaluated_users_count


def save_artifacts(similarity_matrix, unique_movies, movie_to_idx, movie_df, rating_df,
                   user_item_matrix, unique_users, avg_precision, eval_count, sparsity):
    """Save all model artifacts required for serving the Streamlit application."""
    print_banner("STEP 6: SAVING MODEL ARTIFACTS")
    
    # 1. model/movie_similarity.npz
    # Save as scipy.sparse CSR format for strict compatibility & compactness
    sim_sparse = sp.csr_matrix(similarity_matrix)
    sp.save_npz("model/movie_similarity.npz", sim_sparse)
    print(f"[+] Saved 'model/movie_similarity.npz' ({os.path.getsize('model/movie_similarity.npz') / (1024**2):.2f} MB)")
    
    # 2. model/movie_ids.pkl
    with open("model/movie_ids.pkl", "wb") as f:
        pickle.dump(unique_movies.tolist(), f)
    print(f"[+] Saved 'model/movie_ids.pkl' ({len(unique_movies):,} movie IDs)")
    
    # 3. model/movie_index.pkl
    with open("model/movie_index.pkl", "wb") as f:
        pickle.dump(movie_to_idx, f)
    print(f"[+] Saved 'model/movie_index.pkl' (Movie_ID -> index mapping)")
    
    # 4. model/movies.pkl
    # Filter movie_df to strictly contain movies present in the model
    model_movie_set = set(unique_movies)
    model_movies_df = movie_df[movie_df["Movie_ID"].isin(model_movie_set)].copy()
    
    # Add rating count and mean rating statistics for UI enrichment
    rating_counts = rating_df["Movie_ID"].value_counts().to_dict()
    mean_ratings = rating_df.groupby("Movie_ID")["Rating"].mean().to_dict()
    
    model_movies_df["Rating_Count"] = model_movies_df["Movie_ID"].map(rating_counts).fillna(0).astype(int)
    model_movies_df["Avg_Rating"] = model_movies_df["Movie_ID"].map(mean_ratings).fillna(0.0).round(2)
    model_movies_df = model_movies_df.sort_values(by="Name").reset_index(drop=True)
    
    with open("model/movies.pkl", "wb") as f:
        pickle.dump(model_movies_df, f)
    print(f"[+] Saved 'model/movies.pkl' ({len(model_movies_df):,} verified model movies)")
    
    # 5. model/full_catalog.pkl (for handling search for non-model movies gracefully)
    full_catalog = movie_df.set_index("Movie_ID")[["Name", "Year"]].to_dict(orient="index")
    with open("model/full_catalog.pkl", "wb") as f:
        pickle.dump(full_catalog, f)
    print(f"[+] Saved 'model/full_catalog.pkl' ({len(full_catalog):,} total catalog movies)")
    
    # 6. model/sample_user_profiles.pkl (for User-Based extension in Streamlit)
    user_counts = np.diff(user_item_matrix.indptr)
    active_user_indices = np.where(user_counts >= 30)[0][:50]
    sample_users = {}
    for u_idx in active_user_indices:
        uid = int(unique_users[u_idx])
        start = user_item_matrix.indptr[u_idx]
        end = user_item_matrix.indptr[u_idx + 1]
        m_indices = user_item_matrix.indices[start:end]
        r_vals = user_item_matrix.data[start:end]
        sample_users[uid] = [
            {"Movie_ID": int(unique_movies[midx]), "Rating": float(r)}
            for midx, r in zip(m_indices, r_vals)
        ]
        
    with open("model/sample_user_profiles.pkl", "wb") as f:
        pickle.dump(sample_users, f)
    print(f"[+] Saved 'model/sample_user_profiles.pkl' ({len(sample_users)} active user profiles)")
    
    # 7. model/metadata.pkl
    metadata = {
        "num_users": len(unique_users),
        "num_movies": len(unique_movies),
        "total_catalog_movies": len(movie_df),
        "total_ratings": len(rating_df),
        "sparsity": sparsity,
        "precision_at_10": avg_precision,
        "eval_users_count": eval_count,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": "Item-Item Collaborative Filtering",
        "similarity_metric": "Cosine Similarity"
    }
    with open("model/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"[+] Saved 'model/metadata.pkl' (Model metadata and stats)")


def verify_recommendation_flow(similarity_matrix, movie_to_idx, unique_movies, movies_df):
    """Verify recommendation pipeline with concrete test queries."""
    print_banner("STEP 7: VALIDATION OF RECOMMENDATION FLOW")
    
    idx_to_movie = {i: mid for mid, i in movie_to_idx.items()}
    movie_dict = movies_df.set_index("Movie_ID")[["Name", "Year"]].to_dict(orient="index")
    
    test_queries = [
        "Lord of the Rings",
        "Silence of the Lambs",
        "Shrek"
    ]
    
    for q in test_queries:
        print(f"\n[*] Query: '{q}'")
        # Search matching movies in model
        matches = movies_df[movies_df["Name"].str.lower().str.contains(q.lower(), na=False)]
        if matches.empty:
            print(f"    [-] No match found in model.")
            continue
            
        selected_movie = matches.iloc[0]
        selected_id = int(selected_movie["Movie_ID"])
        selected_title = selected_movie["Name"]
        selected_year = selected_movie["Year"]
        
        print(f"    Selected: '{selected_title}' ({selected_year}) [Movie_ID={selected_id}]")
        
        if selected_id not in movie_to_idx:
            print("    [-] Movie not in model index.")
            continue
            
        midx = movie_to_idx[selected_id]
        scores = similarity_matrix[midx]
        
        # Sort descending
        top_indices = np.argsort(-scores)
        
        recs = []
        for idx in top_indices:
            mid = idx_to_movie[idx]
            if mid == selected_id:
                continue
            meta = movie_dict.get(mid, {"Name": "Unknown", "Year": 0})
            recs.append({
                "Title": meta["Name"],
                "Year": meta["Year"],
                "Similarity": float(scores[idx])
            })
            if len(recs) == 5:
                break
                
        print("    Top 5 Recommendations:")
        for rank, r in enumerate(recs, 1):
            print(f"      {rank}. {r['Title']} ({r['Year']}) - Similarity: {r['Similarity']:.4f}")


def main():
    start_total = time.time()
    print_banner("MOVIE RECOMMENDATION SYSTEM - COLLABORATIVE FILTERING TRAINING PIPELINE")
    
    # 1. Dataset Discovery
    movie_path, rating_path = ensure_data_extracted()
    
    # 2. Preprocess & Clean
    movie_df, rating_df = load_and_preprocess(movie_path, rating_path)
    
    # 3. Sparse Interaction Matrix
    (user_item_matrix, unique_users, unique_movies,
     user_to_idx, movie_to_idx, idx_to_movie, sparsity) = build_sparse_interaction_matrix(rating_df)
    
    # 4. Item-Item Cosine Similarity
    similarity_matrix = compute_item_similarity(user_item_matrix)
    
    # 5. Precision@10 Evaluation
    avg_precision, eval_count = evaluate_precision_at_k(
        user_item_matrix, similarity_matrix, k=10, sample_size=500
    )
    
    # 6. Save Artifacts
    save_artifacts(
        similarity_matrix, unique_movies, movie_to_idx, movie_df, rating_df,
        user_item_matrix, unique_users, avg_precision, eval_count, sparsity
    )
    
    # 7. Verification
    verify_recommendation_flow(similarity_matrix, movie_to_idx, unique_movies, movie_df)
    
    total_duration = time.time() - start_total
    print_banner(f"TRAINING COMPLETE IN {total_duration:.2f} SECONDS! ALL ARTIFACTS READY.")


if __name__ == "__main__":
    main()
