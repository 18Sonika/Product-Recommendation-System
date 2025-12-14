import pandas as pd
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)       # remove html tags
    text = re.sub(r"[^a-zA-Z ]", " ", text)  # remove special characters
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_flipkart_data(input_path, output_path):
    # Load dataset
    df = pd.read_csv(input_path, encoding="utf-8")

    # Select only required columns
    df = df[
        [
            "product_name",
            "product_category_tree",
            "retail_price",
            "brand",
            "description"
        ]
    ]

    # Rename columns to standard names
    df.rename(columns={
        "retail_price": "price",
        "product_category_tree": "category"
    }, inplace=True)

    # Drop rows with missing product name or price
    df.dropna(subset=["product_name", "price"], inplace=True)

    # Convert price to numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"] > 0]

    # Simplify category tree (keep top-level category)
    df["category"] = df["category"].apply(
        lambda x: x.split(">>")[0].strip("[]' ") if isinstance(x, str) else "unknown"
    )

    # Clean text fields
    df["product_name"] = df["product_name"].apply(clean_text)
    df["brand"] = df["brand"].fillna("").apply(clean_text)
    df["description"] = df["description"].fillna("").apply(clean_text)

    # Create combined text feature (used later by recommender)
    df["combined_features"] = (
        df["product_name"] + " " +
        df["brand"] + " " +
        df["description"]
    )

    # Remove duplicate products
    df.drop_duplicates(subset=["product_name"], inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    # Save cleaned dataset
    df.to_csv(output_path, index=False)

    print("✅ Flipkart data preprocessing completed successfully")
    print(f"➡ Clean dataset saved at: {output_path}")
    print(f"➡ Total unique products: {len(df)}")


if __name__ == "__main__":
    preprocess_flipkart_data(
        input_path="flipkart_products.csv",
        output_path="outputs/flipkart_clean_products.csv"
    )
