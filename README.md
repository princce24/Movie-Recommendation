# 🎬 HCL Movie Recommendation System Using Collaborative Filtering

A complete, beginner-friendly Machine Learning project developed using **Item-Item Collaborative Filtering** and **Cosine Similarity** on the official **Netflix Prize Dataset** (17.3M+ ratings).

---

## 🚀 Quick Start Guide (One-Click Launchers)

### 🟢 Normal Use (Start Application):
Simply **double-click** the file:
```
run_app.bat
```
This automatically verifies dependencies and launches the Movie Recommendation System at **`http://localhost:8501`**.

---

### ⚙️ Model Training (Retrain Model):
Only use this if you want to retrain the model from scratch:
```
train_model.bat
```
*(Takes ~28 seconds. Computes the sparse CSR matrix and item-item cosine similarity, evaluates Precision@10, and saves all artifacts into the `model/` folder).*

---

## 📌 1. Project Objective

The goal of this system is to recommend top movies that share similar user-rating patterns with a selected movie. When a user selects or searches for a film, the engine discovers latent behavioral correlations across 143,458 users and returns the **Top 10 recommended movies** with similarity scores and match percentages, without relying on manual genre tags.

---

## 📂 2. Project Structure

```
HCL Movie Recommendation System/
│
├── app.py                      # Interactive full-width Streamlit web app
├── recommender.py              # Recommendation inference engine (Item-Item & User-profile CF)
├── train_model.py              # Offline model training & evaluation script
├── requirements.txt            # Python dependencies (Streamlit, Scikit-learn, etc.)
├── README.md                   # This beginner-friendly documentation & viva guide
├── .gitignore                  # Git ignore rules
│
├── run_app.bat                 # One-click launcher for the web application
├── train_model.bat             # One-click batch file to retrain the ML model
│
├── assets/                     # Project UI assets
│
├── data/                       # Datasets
│   ├── Netflix_Dataset_Movie.csv        # 17,770 catalog movies
│   ├── Netflix_Dataset_Rating.csv.zip   # 17.3M raw ratings (compressed)
│   └── Netflix_Dataset_Rating.csv       # Extracted rating dataset
│
└── model/                      # Pre-trained ML model artifacts
    ├── movie_similarity.npz    # Compressed item-item cosine similarity matrix
    ├── movies.pkl              # Clean in-model movies metadata DataFrame
    ├── movie_ids.pkl           # Array of model movie IDs
    ├── movie_index.pkl         # Movie_ID to matrix index mapping
    ├── full_catalog.pkl        # Full 17,770 movie lookup dictionary
    ├── sample_user_profiles.pkl# Active user profiles for user-based rec demo
    └── metadata.pkl            # Model metrics, sparsity %, and Precision@10 score
```

---

## 📊 3. Datasets Used

This project strictly utilizes the two provided Netflix datasets:

1. **`Netflix_Dataset_Movie.csv`**
   - Contains **17,770 movies** with `Movie_ID`, `Year`, and `Name`.
2. **`Netflix_Dataset_Rating.csv.zip`** (`Netflix_Dataset_Rating.csv`)
   - Contains **17,337,458 explicit ratings** (1.0 to 5.0 stars) across **143,458 users** and **1,350 unique movies**.
   - Average ratings per movie: **~12,842 ratings**.

---

## 🧮 4. Core Algorithm: Item-Item Collaborative Filtering

### Why Item-Item over User-User CF?
- **Stability & Scale:** The number of movies (1,350) is much smaller than the number of users (143k). 
- **Memory Footprint:** An item similarity matrix ($1,350 \times 1,350$) takes only **~6.95 MB** of RAM, whereas a user similarity matrix ($143,000 \times 143,000$) would exceed **80 GB** of RAM and cause system crashes on an 8 GB computer.

### Sparse Interaction Matrix (`scipy.sparse.csr_matrix`):
- Potential matrix entries: $143,458 \times 1,350 = 193,668,300$ cells.
- Non-zero ratings: $17,337,458$.
- **Sparsity:** **91.05%** (91% of cells are unrated).
- Storing this as a sparse CSR matrix takes only **132.8 MB**, preventing memory overload.

### Cosine Similarity Formula:
Cosine similarity measures the angle between two movie rating vectors $\mathbf{r}_A$ and $\mathbf{r}_B$:

$$\text{Cosine Similarity}(A, B) = \frac{\mathbf{r}_A \cdot \mathbf{r}_B}{\|\mathbf{r}_A\|_2 \|\mathbf{r}_B\|_2} = \frac{\sum_{u} R_{u, A} \cdot R_{u, B}}{\sqrt{\sum_{u} R_{u, A}^2} \cdot \sqrt{\sum_{u} R_{u, B}^2}}$$

This normalizes for differences in rating volume so that blockbusters and niche movies can be compared fairly based on correlation.

---

## 🛠️ 5. Step-by-Step Manual Instructions

### Step 1: Install Dependencies
Open Command Prompt in this folder and run:
```bash
pip install -r requirements.txt
```

### Step 2: Train Model (Offline)
```bash
python train_model.py
```
*(Generates all `.pkl` and `.npz` files in `model/`).*

### Step 3: Launch Web Application
```bash
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

### Step 4: Stop Application
Press **`Ctrl + C`** in the command prompt or terminal window running Streamlit.

---

## 🖥️ 6. Web Application Features

1. **🍿 Discover Movies (Item-Item CF):**
   - Searchable movie dropdown of 1,350 verified model movies.
   - 6 Quick Benchmark movie buttons for instant 1-click recommendations.
   - Selected movie spotlight card with release year, total ratings, and star rating.
   - Top 10 recommendation cards with match percentages, progress bars, and explanations.
   - Out-of-model handling: displays *"This movie does not have enough interaction data for a reliable recommendation."* for catalog movies without rating data.
2. **👤 User Personalized Recs:**
   - Demonstrates the User-Based extension. Select a sample active user to view their favorite movies ($\ge 4.0$ stars) and generate recommendations excluding already-watched titles.
3. **📊 Dataset Statistics & EDA:**
   - Total ratings, active users, sparsity (91.05%), and Precision@10 score (35.51%).
   - Interactive bar chart of the Top 10 most-rated Netflix movies.
4. **🧠 How It Works & Architecture:**
   - Educational walkthrough explaining the 4-step pipeline and mathematical formulas.

---

## 🎓 7. Short Viva / Presentation Q&A

> **Q: How does the system make recommendations?**
> **A:** *"It builds a sparse user-movie matrix using `scipy.sparse.csr_matrix` from 17.3M Netflix ratings. It transposes this to represent each movie as a vector of user ratings and calculates pairwise Cosine Similarity. When a movie is selected, it looks up its pre-computed similarity vector, sorts descending, and returns the top 10 most similar films."*

> **Q: Why separate `train_model.py` and `app.py`?**
> **A:** *"Training takes ~28 seconds to compute on 17.3M rows. By computing the similarity matrix offline and saving it to `model/movie_similarity.npz`, the Streamlit app loads in less than 1 second and generates recommendations in under 5 milliseconds."*

> **Q: What is the Precision@10 score?**
> **A:** *"We evaluated the model on 499 active users with a 30% hold-out split on liked movies ($\ge 4.0$ stars). The model achieved a **35.51% Precision@10**, confirming strong alignment with real user preferences."*

---

## 🔮 8. Limitations & Future Enhancements

- **Cold-Start Problem:** Movies with zero or few ratings cannot be accurately recommended using collaborative filtering alone. Future improvement: combine with content-based embeddings (e.g. plot synopsis or actors) into a hybrid model.
- **Implicit Feedback:** Add watch time, skips, and clicks to complement explicit star ratings.
