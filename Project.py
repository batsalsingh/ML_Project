

import os
import ast
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import time

# -----------------------
# Paths
# -----------------------
MOVIES_CSV = r"C:\Users\batsa\OneDrive\Desktop\ML LAB\project\tmdb_5000_movies.csv"
CREDITS_CSV = r"C:\Users\batsa\OneDrive\Desktop\ML LAB\project\tmdb_5000_credits.csv"

VECT_FILE = "vectorizer.joblib"
SVD_FILE = "svd.joblib"
REDUCED_FILE = "X_reduced.npy"
META_FILE = "movies_meta.pkl"

MODEL_ACCURACY = 0.0
VECTOR, SVD, X_REDUCED, META = None, None, None, None


def safe_literal_eval(x):
    try:
        return ast.literal_eval(x)
    except:
        return []


def parse_names_field(x, key='name', limit=3):
    L = safe_literal_eval(x)
    names = []
    for d in L[:limit]:
        if isinstance(d, dict) and key in d:
            names.append(d[key].replace(" ", ""))
    return " ".join(names)


def preprocess_and_train(movies_csv, credits_csv, tfidf_max_features=20000, target_variance=0.85):
    t0 = time.time()
    print("📥 Loading CSVs...")
    movies = pd.read_csv(movies_csv)
    credits = pd.read_csv(credits_csv)

    print("🔗 Merging datasets...")
    df = movies.merge(credits, on="title", how="inner", suffixes=("", "_credit"))
    drop_cols = ['homepage', 'budget', 'status', 'original_title', 'tagline']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    df = df.drop_duplicates(subset=['title']).reset_index(drop=True)
    df = df.dropna(subset=['overview']).reset_index(drop=True)

    print("🧩 Building combined text field (genres, keywords, cast, overview)...")
    df['genres_clean'] = df['genres'].apply(lambda x: parse_names_field(x))
    df['keywords_clean'] = df['keywords'].apply(lambda x: parse_names_field(x))
    df['cast_clean'] = df['cast'].apply(lambda x: parse_names_field(x, key='name', limit=3))
    df['combined'] = (
        df['genres_clean'] + ' ' +
        df['keywords_clean'] + ' ' +
        df['cast_clean'] + ' ' +
        df['overview'].astype(str)
    )

    meta_cols = ['title']
    if 'poster_path' in df.columns:
        meta_cols.append('poster_path')
    if 'release_date' in df.columns:
        meta_cols.append('release_date')
    meta = df[meta_cols].copy().reset_index().rename(columns={'index': 'idx'})

    n_samples = df.shape[0]
    print(f"ℹ Samples: {n_samples}")

    print(f"🔢 TF-IDF Vectorization (max_features={tfidf_max_features}) ...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=tfidf_max_features)
    X_tfidf = vectorizer.fit_transform(df['combined'].values.astype('U'))
    n_features = X_tfidf.shape[1]
    print(f"ℹ TF-IDF produced {n_features} features (vocab size).")

    max_allowed = min(n_samples - 1, n_features)
    base = 200
    if base >= max_allowed:
        base = max(10, max_allowed // 4)

    n_comp = base
    last_svd = None
    last_Xr = None
    variance_ratio = 0.0

    while True:
        try:
            print(f"📉 Fitting TruncatedSVD with n_components={n_comp} ...")
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            X_reduced = svd.fit_transform(X_tfidf)
            variance_ratio = svd.explained_variance_ratio_.sum()
            print(f"➡ Variance retained: {variance_ratio*100:.2f}%")
            last_svd = svd
            last_Xr = X_reduced
        except MemoryError:
            print("⚠ MemoryError. Try reducing variance or features.")
            break

        if variance_ratio >= target_variance or n_comp >= max_allowed:
            break
        n_comp = min(n_comp * 2, max_allowed)

    if last_svd is None:
        raise RuntimeError("SVD training failed; no model produced.")

    print("💾 Saving artifacts...")
    joblib.dump(vectorizer, VECT_FILE)
    joblib.dump(last_svd, SVD_FILE)
    np.save(REDUCED_FILE, last_Xr)
    meta.to_pickle(META_FILE)

    elapsed = time.time() - t0
    print(f"✅ Training done in {elapsed:.0f}s | Variance: {variance_ratio*100:.2f}%")
    return vectorizer, last_svd, last_Xr, meta, variance_ratio


def get_similar_titles(query_title, X_reduced, meta, top_n=5):
    matches = meta[meta['title'].str.lower() == query_title.lower()]
    if len(matches) == 0:
        matches2 = meta[meta['title'].str.lower().str.contains(query_title.lower())]
        if len(matches2) > 0:
            idx = int(matches2.iloc[0]['idx'])
        else:
            return None, []
    else:
        idx = int(matches.iloc[0]['idx'])

    q_vec = X_reduced[idx].reshape(1, -1)
    sims = cosine_similarity(q_vec, X_reduced).flatten()
    ranking = np.argsort(sims)[::-1]
    top = [int(i) for i in ranking if int(i) != idx][:top_n]

    results = []
    for i in top:
        row = meta[meta['idx'] == i].iloc[0]
        results.append({
            "title": row['title'],
            "poster_path": row.get('poster_path', None),
            "release_date": row.get('release_date', None),
            "similarity": float(sims[i])
        })
    return idx, results


# -----------------------
# Flask UI
# -----------------------
app = Flask(__name__)
CORS(app)
HTML_PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>🎬 Movie Recommendation System</title>
<style>
body {
  background: radial-gradient(circle at top left, #0f172a, #020617);
  color: #e2e8f0;
  font-family: 'Segoe UI', sans-serif;
  margin: 0;
  padding: 20px;
}
h1 {
  text-align: center;
  font-size: 2.2em;
  color: #38bdf8;
  text-shadow: 0 0 10px #38bdf8;
  margin-bottom: 5px;
}
.accuracy-box {
  text-align: center;
  margin-bottom: 25px;
}
.bar-bg {
  width: 70%;
  margin: auto;
  height: 20px;
  border-radius: 12px;
  background: #1e293b;
  overflow: hidden;
  box-shadow: inset 0 0 6px rgba(0,0,0,0.5);
}
.bar-fill {
  height: 20px;
  border-radius: 12px;
  background: linear-gradient(90deg, #3b82f6, #60a5fa);
  width: {{accuracy}}%;
  transition: width 1s ease;
}
.accuracy-text {
  margin-top: 8px;
  color: #a5b4fc;
  font-size: 15px;
}
.search {
  display: flex;
  justify-content: center;
  margin-bottom: 20px;
  gap: 10px;
}
input {
  padding: 12px;
  width: 420px;
  border-radius: 10px;
  border: none;
  outline: none;
  font-size: 16px;
  background: #1e293b;
  color: white;
}
button {
  padding: 10px 18px;
  border: none;
  border-radius: 10px;
  background: linear-gradient(135deg, #2563eb, #1d4ed8);
  color: white;
  cursor: pointer;
  transition: 0.3s;
}
button:hover {
  background: linear-gradient(135deg, #3b82f6, #2563eb);
  transform: scale(1.05);
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px,1fr));
  gap: 20px;
  max-width: 1100px;
  margin: 20px auto;
}
.card {
  background: #1e293b;
  border-radius: 15px;
  text-align: center;
  box-shadow: 0 6px 20px rgba(0,0,0,0.3);
  padding-bottom: 8px;
  transition: 0.3s;
}
.card:hover {
  transform: translateY(-5px);
  box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}
.poster {
  width: 100%;
  height: 300px;
  object-fit: cover;
  border-top-left-radius: 15px;
  border-top-right-radius: 15px;
  background: #0f172a;
}
.title {
  font-weight: bold;
  margin-top: 8px;
  color: #38bdf8;
}
.small {
  font-size: 13px;
  color: #94a3b8;
  margin-bottom: 10px;
}
.info {
  color: #9aa7c7;
  text-align: center;
  margin-top: 6px;
  font-size: 14px;
}
</style>
</head>
<body>
<div class="container">
  <h1>🎬 Movie Recommendation System</h1>

  <div class="accuracy-box">
    <div class="accuracy-text"><b>Model Accuracy (Variance Retained): {{accuracy}}%</b></div>
    <div class="bar-bg"><div class="bar-fill" style="width:{{accuracy}}%;"></div></div>
    <div class="info">Adaptive SVD retrains each run — higher = better compression quality</div>
  </div>

  <div class="search">
    <input id="movieInput" placeholder="Enter movie title..." />
    <button onclick="searchMovie()">Search</button>
  </div>

  <div id="results" class="grid"></div>
</div>

<script>
async function searchMovie(){
  const q = document.getElementById('movieInput').value.trim();
  if(!q) return alert('Please enter a movie title.');
  const res = await fetch('/recommend', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({title: q})
  });
  const data = await res.json();
  const grid = document.getElementById('results'); grid.innerHTML = '';
  if(data.error || !data.results){
    grid.innerHTML = '<p style="color:#f87171;text-align:center;">No movie found.</p>';
    return;
  }
  const base = 'https://image.tmdb.org/t/p/w342';
  const fallback = 'https://via.placeholder.com/342x513?text=No+Poster';
  data.results.forEach(r=>{
    const poster = (r.poster_path) ? base + r.poster_path : fallback;
    const div = document.createElement('div'); div.className = 'card';
    div.innerHTML = `
      <img class="poster" src="${poster}" alt="poster">
      <div class="title">${r.title}</div>
      <div class="small">${r.release_date || ''} • sim ${r.similarity.toFixed(3)}</div>
    `;
    grid.appendChild(div);
  });
}
</script>
</body>
</html>
"""


@app.route('/')
def home():
    return render_template_string(HTML_PAGE, accuracy=f"{MODEL_ACCURACY*100:.2f}")

@app.route('/recommend', methods=['POST'])
def recommend_api():
    title = request.json.get('title','').strip()
    if not title:
        return jsonify({"error":"Missing title"}), 400
    idx, results = get_similar_titles(title, X_REDUCED, META)
    if idx is None:
        return jsonify({"error":"Movie not found"}), 404
    return jsonify({"results": results})


if __name__ == "__main__":
    VECTOR, SVD, X_REDUCED, META, MODEL_ACCURACY = preprocess_and_train(
        MOVIES_CSV, CREDITS_CSV, tfidf_max_features=20000, target_variance=0.85
    )
    print(f"✅ Final variance retained: {MODEL_ACCURACY*100:.2f}%")
    print("🚀 Opening Flask at http://127.0.0.1:5000")
    app.run(port=5000, debug=False)
