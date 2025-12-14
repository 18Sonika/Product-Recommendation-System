🛒 Flipkart Product Recommendation System

A real-world, hybrid product recommendation system built using machine learning (TF-IDF + cosine similarity) and popularity-based ranking, deployed as an interactive Streamlit web application.
The system is enhanced with explainable AI, visual analytics, confidence scoring, sorting controls, cold-start handling, and CSV export.

📌 Project Highlights

🔹 Real Flipkart product dataset

🔹 Hybrid recommendation (Similarity + Popularity)

🔹 Confidence score for recommendations

🔹 Explainable AI (why a product is recommended)

🔹 Visual analytics (charts & comparisons)

🔹 Cold-start handling

🔹 User-controlled sorting

🔹 CSV export support

🔹 Streamlit-based interactive UI

🎯 Objective

To design and implement an industry-grade e-commerce product recommendation system that suggests relevant products based on textual similarity, popularity trends, and business constraints, similar to real Flipkart/Amazon recommendation engines.

🧠 Recommendation Approach
🔹 1. Content-Based Filtering

Product descriptions are converted into vectors using TF-IDF

Cosine similarity is used to measure similarity between products

🔹 2. Popularity-Based Ranking

Product frequency is used as a popularity proxy

Popularity scores are normalized for fair ranking

🔹 3. Hybrid Recommendation

Final score is calculated as:

Hybrid Score = (0.7 × Similarity Score) + (0.3 × Popularity Score)

🔹 4. Cold-Start Handling

If a product is not found:

The system recommends top popular products

Ensures stable behavior for new or unknown inputs

📊 Additional Features

Confidence Badge (High / Medium / Low)

Explainable AI (clear recommendation reasons)

Sorting Options

Hybrid score (default)

Price (low → high)

Popularity

Visual Charts

Hybrid score comparison

Similarity vs popularity
🛠 Technologies Used

Python 3.11

pandas

scikit-learn

Streamlit

TF-IDF Vectorizer

Cosine Similarity
🧪 Sample Output

Product recommendations with:

Category

Price

Similarity score

Popularity score

Final hybrid score

Confidence level

Visual charts for comparison

Downloadable CSV report

Price comparison

CSV Export of recommendations
