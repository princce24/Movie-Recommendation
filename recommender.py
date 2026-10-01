"""
=============================================================================
RECOMMENDER MODULE - COLLABORATIVE FILTERING ENGINE
Core logic for Item-Item CF, User-based CF extension, search, and inference.
=============================================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp


class MovieRecommender:
    """Production-ready inference engine for Collaborative Filtering."""

    def __init__(self, model_dir="model"):
        self.model_dir = model_dir
        self.movies_df = None
        self.movie_to_idx = None
        self.movie_ids = None
        self.idx_to_movie = None
        self.full_catalog = None
        self.similarity_matrix = None
        self.sample_users = {}
        self.metadata = {}
        self.is_loaded = False

    def load_artifacts(self):
        """Load pre-trained artifacts from the model directory."""
        if self.is_loaded:
            return self

        # 1. Load movies DataFrame
        movies_path = os.path.join(self.model_dir, "movies.pkl")
        if not os.path.exists(movies_path):
            raise FileNotFoundError(f"Model file '{movies_path}' not found. Please run train_model.py first.")
        with open(movies_path, "rb") as f:
            self.movies_df = pickle.load(f)

        # 2. Load movie index mapping
        index_path = os.path.join(self.model_dir, "movie_index.pkl")
        with open(index_path, "rb") as f:
            self.movie_to_idx = pickle.load(f)
        self.idx_to_movie = {idx: mid for mid, idx in self.movie_to_idx.items()}

        # 3. Load movie IDs list
        ids_path = os.path.join(self.model_dir, "movie_ids.pkl")
        with open(ids_path, "rb") as f:
            self.movie_ids = pickle.load(f)

        # 4. Load full catalog for out-of-model handling
        cat_path = os.path.join(self.model_dir, "full_catalog.pkl")
        if os.path.exists(cat_path):
            with open(cat_path, "rb") as f:
                self.full_catalog = pickle.load(f)
        else:
            self.full_catalog = self.movies_df.set_index("Movie_ID")[["Name", "Year"]].to_dict(orient="index")

        # 5. Load similarity matrix
        sim_path = os.path.join(self.model_dir, "movie_similarity.npz")
        try:
            # First attempt scipy.sparse loader
            csr_sim = sp.load_npz(sim_path)
            self.similarity_matrix = csr_sim.toarray().astype(np.float32)
        except Exception:
            # Fallback to numpy archive
            npz_data = np.load(sim_path)
            first_key = npz_data.files[0]
            self.similarity_matrix = npz_data[first_key].astype(np.float32)

        # 6. Load sample user profiles (if available)
        users_path = os.path.join(self.model_dir, "sample_user_profiles.pkl")
        if os.path.exists(users_path):
            with open(users_path, "rb") as f:
                self.sample_users = pickle.load(f)

        # 7. Load metadata
        meta_path = os.path.join(self.model_dir, "metadata.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)

        self.is_loaded = True
        return self

    def get_valid_movies(self):
        """Return the list of movies strictly available in the trained model."""
        if not self.is_loaded:
            self.load_artifacts()
        return self.movies_df

    def search_movie(self, keyword, limit=10):
        """
        Partial search for movie titles.
        Returns matching records from the model first, then catalog.
        """
        if not self.is_loaded:
            self.load_artifacts()

        if not keyword or not str(keyword).strip():
            return []

        kw = str(keyword).strip().lower()

        # Search in model movies
        model_matches = self.movies_df[
            self.movies_df["Name"].str.lower().str.contains(kw, na=False)
        ].copy()

        results = []
        for _, row in model_matches.head(limit).iterrows():
            results.append({
                "Movie_ID": int(row["Movie_ID"]),
                "Title": str(row["Name"]),
                "Year": int(row["Year"]) if pd.notnull(row["Year"]) else "N/A",
                "Rating_Count": int(row.get("Rating_Count", 0)),
                "Avg_Rating": float(row.get("Avg_Rating", 0.0)),
                "In_Model": True
            })

        # If few results, also check full catalog to let user know they exist
        if len(results) < limit and self.full_catalog:
            existing_ids = {r["Movie_ID"] for r in results}
            for mid, meta in self.full_catalog.items():
                if kw in meta["Name"].lower() and mid not in existing_ids:
                    results.append({
                        "Movie_ID": int(mid),
                        "Title": str(meta["Name"]),
                        "Year": int(meta["Year"]) if meta["Year"] else "N/A",
                        "Rating_Count": 0,
                        "Avg_Rating": 0.0,
                        "In_Model": False
                    })
                    if len(results) >= limit:
                        break

        return results

    def recommend_movies(self, movie_name=None, movie_id=None, n=10):
        """
        Item-Item Collaborative Filtering Recommendation.
        
        Parameters:
        - movie_name: string title or partial title
        - movie_id: integer Movie_ID
        - n: number of recommendations (default 10)
        
        Returns dict with status, selected movie metadata, and top-N recommendations.
        """
        if not self.is_loaded:
            self.load_artifacts()

        target_mid = None
        selected_meta = None

        # 1. Resolve Movie_ID
        if movie_id is not None:
            target_mid = int(movie_id)
        elif movie_name is not None:
            clean_name = str(movie_name).strip().lower()

            # Exact match in model movies
            exact = self.movies_df[self.movies_df["Name"].str.lower() == clean_name]
            if not exact.empty:
                target_mid = int(exact.iloc[0]["Movie_ID"])
            else:
                # Substring match in model movies
                sub = self.movies_df[self.movies_df["Name"].str.lower().str.contains(clean_name, na=False)]
                if not sub.empty:
                    target_mid = int(sub.iloc[0]["Movie_ID"])
                else:
                    # Check if present in full catalog
                    catalog_match = [
                        (mid, meta) for mid, meta in self.full_catalog.items()
                        if clean_name in meta["Name"].lower()
                    ]
                    if catalog_match:
                        # Movie exists in Netflix catalog but was filtered out (lacks interaction data)
                        return {
                            "status": "insufficient_data",
                            "message": "This movie does not have enough interaction data for a reliable recommendation."
                        }
                    return {
                        "status": "not_found",
                        "message": f"Movie '{movie_name}' was not found in the catalog."
                    }

        if target_mid is None:
            return {
                "status": "error",
                "message": "Please specify a valid movie title or Movie_ID."
            }

        # 2. Verify target_mid exists in the trained model
        if target_mid not in self.movie_to_idx:
            return {
                "status": "insufficient_data",
                "message": "This movie does not have enough interaction data for a reliable recommendation."
            }

        # Retrieve selected movie metadata
        movie_row = self.movies_df[self.movies_df["Movie_ID"] == target_mid]
        if not movie_row.empty:
            r = movie_row.iloc[0]
            selected_meta = {
                "Movie_ID": target_mid,
                "Title": str(r["Name"]),
                "Year": int(r["Year"]) if pd.notnull(r["Year"]) else "N/A",
                "Rating_Count": int(r.get("Rating_Count", 0)),
                "Avg_Rating": float(r.get("Avg_Rating", 0.0))
            }
        else:
            cat_meta = self.full_catalog.get(target_mid, {"Name": "Unknown", "Year": "N/A"})
            selected_meta = {
                "Movie_ID": target_mid,
                "Title": str(cat_meta["Name"]),
                "Year": int(cat_meta["Year"]) if cat_meta["Year"] else "N/A",
                "Rating_Count": 0,
                "Avg_Rating": 0.0
            }

        # 3. Retrieve item similarity vector and rank
        model_idx = self.movie_to_idx[target_mid]
        sim_scores = self.similarity_matrix[model_idx]

        # Sort descending
        ranked_indices = np.argsort(-sim_scores)

        # Lookup dictionary for fast O(1) metadata access
        model_lookup = self.movies_df.set_index("Movie_ID").to_dict(orient="index")

        recommendations = []
        seen_titles = {selected_meta["Title"].strip().lower()}

        for idx in ranked_indices:
            mid = self.idx_to_movie[idx]
            # Exclude the selected movie itself
            if mid == target_mid:
                continue

            movie_info = model_lookup.get(mid)
            if not movie_info:
                continue

            raw_title = str(movie_info["Name"]).strip()
            norm_title = raw_title.lower()

            # Prevent duplicate titles (e.g. slight re-releases)
            if norm_title in seen_titles:
                continue
            seen_titles.add(norm_title)

            score = float(sim_scores[idx])
            pct = max(0, min(100, int(round(score * 100))))

            recommendations.append({
                "Rank": len(recommendations) + 1,
                "Movie_ID": mid,
                "Title": raw_title,
                "Year": int(movie_info["Year"]) if pd.notnull(movie_info["Year"]) else "N/A",
                "Rating_Count": int(movie_info.get("Rating_Count", 0)),
                "Avg_Rating": float(movie_info.get("Avg_Rating", 0.0)),
                "Similarity": round(score, 4),
                "Match_Score": f"{pct}%",
                "Explanation": "Recommended because users with similar rating patterns also rated it similarly."
            })

            if len(recommendations) == n:
                break

        return {
            "status": "success",
            "selected_movie": selected_meta,
            "recommendations": recommendations
        }

    def recommend_for_user(self, user_id, n=10):
        """
        User-Based Collaborative Filtering Extension.
        Aggregates item-item similarity vectors for movies the user rated highly (rating >= 4.0).
        Excludes movies the user has already rated.
        """
        if not self.is_loaded:
            self.load_artifacts()

        user_id = int(user_id)
        if user_id not in self.sample_users:
            return {
                "status": "not_found",
                "message": f"User ID {user_id} not found in sample profiles. Available IDs: {list(self.sample_users.keys())[:5]}"
            }

        user_ratings = self.sample_users[user_id]
        
        # Movies user liked (rating >= 4.0)
        liked_movies = [r for r in user_ratings if r["Rating"] >= 4.0 and r["Movie_ID"] in self.movie_to_idx]
        all_rated_mids = {r["Movie_ID"] for r in user_ratings}

        if not liked_movies:
            # Fallback to top-rated movies by user if none >= 4.0
            sorted_ratings = sorted(user_ratings, key=lambda x: -x["Rating"])
            liked_movies = [r for r in sorted_ratings[:5] if r["Movie_ID"] in self.movie_to_idx]

        if not liked_movies:
            return {
                "status": "insufficient_data",
                "message": f"User ID {user_id} has not rated any movies present in the active collaborative filtering model."
            }

        # Collect similarity vectors of liked movies
        liked_indices = [self.movie_to_idx[r["Movie_ID"]] for r in liked_movies]
        liked_weights = np.array([r["Rating"] for r in liked_movies], dtype=np.float32)

        # Weighted average of similarity scores
        sim_slice = self.similarity_matrix[liked_indices]
        weighted_scores = np.average(sim_slice, axis=0, weights=liked_weights)

        # Mask out movies the user has already rated
        for rated_mid in all_rated_mids:
            if rated_mid in self.movie_to_idx:
                rated_idx = self.movie_to_idx[rated_mid]
                weighted_scores[rated_idx] = -1.0

        # Sort descending
        ranked_indices = np.argsort(-weighted_scores)

        model_lookup = self.movies_df.set_index("Movie_ID").to_dict(orient="index")
        recommendations = []
        seen_titles = set()

        for idx in ranked_indices:
            mid = self.idx_to_movie[idx]
            movie_info = model_lookup.get(mid)
            if not movie_info:
                continue

            raw_title = str(movie_info["Name"]).strip()
            norm_title = raw_title.lower()

            if norm_title in seen_titles:
                continue
            seen_titles.add(norm_title)

            score = float(weighted_scores[idx])
            pct = max(0, min(100, int(round(score * 100))))

            recommendations.append({
                "Rank": len(recommendations) + 1,
                "Movie_ID": mid,
                "Title": raw_title,
                "Year": int(movie_info["Year"]) if pd.notnull(movie_info["Year"]) else "N/A",
                "Rating_Count": int(movie_info.get("Rating_Count", 0)),
                "Avg_Rating": float(movie_info.get("Avg_Rating", 0.0)),
                "Similarity": round(score, 4),
                "Match_Score": f"{pct}%",
                "Explanation": "Recommended based on your high ratings for related movies in your profile."
            })

            if len(recommendations) == n:
                break

        # Format liked movies for display
        liked_display = []
        for r in liked_movies[:8]:
            mid = r["Movie_ID"]
            info = model_lookup.get(mid, {"Name": "Unknown", "Year": "N/A"})
            liked_display.append({
                "Movie_ID": mid,
                "Title": info["Name"],
                "Year": info["Year"],
                "Rating": r["Rating"]
            })

        return {
            "status": "success",
            "user_id": user_id,
            "total_ratings": len(user_ratings),
            "liked_movies": liked_display,
            "recommendations": recommendations
        }


# Global singleton instance for clean import
recommender = MovieRecommender()


def search_movie(keyword, limit=10):
    return recommender.search_movie(keyword, limit=limit)


def recommend_movies(movie_name=None, movie_id=None, n=10):
    return recommender.recommend_movies(movie_name=movie_name, movie_id=movie_id, n=n)


def recommend_for_user(user_id, n=10):
    return recommender.recommend_for_user(user_id=user_id, n=n)
