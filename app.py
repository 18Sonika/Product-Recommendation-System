import streamlit as st
import pandas as pd
from src.recommendation_engine import recommend_products

st.set_page_config(
    page_title="Flipkart Product Recommendation System",
    layout="centered"
)

@st.cache_data
def load_data():
    return pd.read_csv("outputs/flipkart_clean_products.csv")

df = load_data()

# ---------------- TITLE ----------------
st.title("🛒 Flipkart Product Recommendation System")

st.write(
    "Hybrid recommendation engine using TF-IDF similarity, popularity ranking, "
    "confidence scoring, explainable AI, and cold-start handling."
)

# ---------------- MODEL SUMMARY ----------------
with st.expander("🔍 How this recommendation system works"):
    st.markdown("""
    **Model Overview**
    - TF-IDF converts product text into numerical vectors  
    - Cosine similarity finds similar products  
    - Popularity improves ranking reliability  
    - Hybrid Score = 70% similarity + 30% popularity  
    - Cold-start handled using popularity-based fallback  

    **Why this approach**
    - Improves accuracy and trust  
    - Matches real e-commerce systems  
    - Transparent and explainable
    """)

# ---------------- USER INPUTS ----------------
product_name = st.selectbox(
    "Select a product",
    sorted(df["product_name"].unique())
)

min_price, max_price = st.slider(
    "Select price range (₹)",
    int(df["price"].min()),
    int(df["price"].max()),
    (int(df["price"].min()), int(df["price"].max()))
)

top_n = st.slider("Number of recommendations", 1, 5, 3)

sort_by = st.selectbox(
    "Sort recommendations by",
    ["Hybrid Score (Recommended)", "Price (Low to High)", "Popularity"]
)

results = []

# ---------------- RUN MODEL ----------------
if st.button("Get Recommendations"):
    with st.spinner("Generating recommendations..."):
        results = recommend_products(
            product_name,
            top_n=top_n,
            min_price=min_price,
            max_price=max_price
        )

    if results:
        result_df = pd.DataFrame(results)

        # Sorting
        if sort_by == "Price (Low to High)":
            result_df = result_df.sort_values("price")
        elif sort_by == "Popularity":
            result_df = result_df.sort_values("popularity_score", ascending=False)
        else:
            result_df = result_df.sort_values("final_score", ascending=False)

        # ---------------- TABLE ----------------
        st.subheader("📋 Recommended Products")
        st.dataframe(
            result_df[
                [
                    "product",
                    "category",
                    "price",
                    "similarity_score",
                    "popularity_score",
                    "final_score",
                    "confidence"
                ]
            ],
            use_container_width=True
        )

        # ---------------- CSV EXPORT ----------------
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download Recommendations as CSV",
            csv,
            "flipkart_recommendations.csv",
            "text/csv"
        )

        # ---------------- VISUAL CHARTS ----------------
        st.subheader("📊 Final Score Comparison")
        st.bar_chart(result_df.set_index("product")["final_score"])

        st.subheader("📈 Similarity vs Popularity")
        st.bar_chart(
            result_df.set_index("product")[["similarity_score", "popularity_score"]]
        )

        st.subheader("💰 Price Comparison")
        st.bar_chart(result_df.set_index("product")["price"])

        # ---------------- EXPLAINABLE AI ----------------
        st.subheader("🧠 Why these products were recommended")
        for r in results:
            st.markdown(
                f"""
                **{r['product']}**  
                _{r['explanation']}_  
                **Confidence:** {r['confidence']}  
                ---
                """
            )
    else:
        st.warning("No recommendations found.")
