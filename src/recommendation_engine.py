import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- LOAD DATA ----------------
df = pd.read_csv("outputs/flipkart_clean_products.csv")

df["price"] = pd.to_numeric(df["price"], errors="coerce")
df.dropna(subset=["price"], inplace=True)
df.reset_index(drop=True, inplace=True)

# ---------------- POPULARITY SCORE ----------------
popularity = df.groupby("product_name").size()
df["popularity_score"] = df["product_name"].map(popularity)
df["popularity_score"] = df["popularity_score"] / df["popularity_score"].max()

# ---------------- TF-IDF VECTORIZATION ----------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=8000
)
tfidf_matrix = vectorizer.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df["product_name"]).drop_duplicates()

# ---------------- COLD START HANDLER ----------------
def cold_start_recommendation(top_n=5):
    popular = df.sort_values("popularity_score", ascending=False).head(top_n)

    results = []
    for _, row in popular.iterrows():
        results.append({
            "product": row["product_name"],
            "category": row["category"],
            "price": round(row["price"], 2),
            "similarity_score": 0.0,
            "popularity_score": round(row["popularity_score"], 3),
            "final_score": round(row["popularity_score"], 3),
            "confidence": "High",
            "explanation": "Recommended due to high popularity (cold-start handling)."
        })
    return results

# ---------------- MAIN RECOMMENDER ----------------
def recommend_products(product_name, top_n=5, min_price=0, max_price=100000):

    if product_name not in indices:
        return cold_start_recommendation(top_n)

    idx = indices[product_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    base_category = df.iloc[idx]["category"]

    recommendations = []

    for i, sim in sim_scores:
        if i == idx:
            continue

        price = df.iloc[i]["price"]
        if not (min_price <= price <= max_price):
            continue

        if df.iloc[i]["category"] != base_category:
            continue

        hybrid_score = (0.7 * sim) + (0.3 * df.iloc[i]["popularity_score"])

        # Confidence badge
        if hybrid_score > 0.7:
            confidence = "High"
        elif hybrid_score > 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"

        recommendations.append({
            "product": df.iloc[i]["product_name"],
            "category": df.iloc[i]["category"],
            "price": round(price, 2),
            "similarity_score": round(sim, 3),
            "popularity_score": round(df.iloc[i]["popularity_score"], 3),
            "final_score": round(hybrid_score, 3),
            "confidence": confidence,
            "explanation": (
                "Recommended because it belongs to the same category, "
                "has similar product descriptions, and is popular among users."
            )
        })

    recommendations = sorted(
        recommendations,
        key=lambda x: x["final_score"],
        reverse=True
    )[:top_n]

    return recommendations
