import pandas as pd
from pathlib import Path

DATA_DIR = Path("../app/data")

all_daily_data = []

for csv_file in DATA_DIR.glob("*_Forum_data.csv"):

    company = csv_file.stem.replace("_Forum_data", "")

    print(f"Processing {company}")

    df = pd.read_csv(csv_file)

    df["Date"] = pd.to_datetime(df["Date"])

    # Daily aggregation
    daily = df.groupby("Date").agg(
        avg_sentiment=("sentiment_score", "mean"),
        message_count=("message", "count"),
        avg_likes=("msg_like_count", "mean"),
        avg_replies=("msg_reply_count", "mean"),
        avg_reposts=("msg_repost_count", "mean"),
        price=("NSE_current_price", "mean")
    ).reset_index()

    # Sentiment distribution
    sentiment_dist = (
        df.groupby(["Date", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    daily = daily.merge(sentiment_dist, on="Date", how="left")

    # Ensure columns exist
    for col in ["positive", "negative", "neutral"]:
        if col not in daily.columns:
            daily[col] = 0

    total_posts = (
        daily["positive"]
        + daily["negative"]
        + daily["neutral"]
    )

    daily["positive_ratio"] = daily["positive"] / total_posts
    daily["negative_ratio"] = daily["negative"] / total_posts
    daily["neutral_ratio"] = daily["neutral"] / total_posts

    # --------------------------------------------------
    # Momentum Features
    # --------------------------------------------------

    daily = daily.sort_values("Date")

    # Price momentum
    daily["price_return_1d"] = daily["price"].pct_change()

    daily["price_return_3d"] = (
        daily["price"] / daily["price"].shift(3)
    ) - 1

    daily["price_return_7d"] = (
        daily["price"] / daily["price"].shift(7)
    ) - 1

    # Sentiment momentum
    daily["sentiment_3d_avg"] = (
        daily["avg_sentiment"]
        .rolling(window=3)
        .mean()
    )

    daily["sentiment_7d_avg"] = (
        daily["avg_sentiment"]
        .rolling(window=7)
        .mean()
    )

    # Message volume momentum
    daily["volume_change"] = (
        daily["message_count"]
        .pct_change()
    )

    # --------------------------------------------------
    # Company label
    # --------------------------------------------------

    daily["company"] = company

    # --------------------------------------------------
    # Target Variable
    # --------------------------------------------------

    # Next day's price
    daily["next_price"] = daily["price"].shift(-1)

    # Percentage return
    daily["return_pct"] = (
        (daily["next_price"] - daily["price"])
        / daily["price"]
    )

    # Target:
    # 1 if stock rises by more than 0.5%
    # 0 otherwise
    daily["target"] = (
        daily["return_pct"] > 0.005
    ).astype(int)

    # Remove rows created by rolling windows
    daily = daily.dropna()

    all_daily_data.append(daily)

# Combine all companies
final_df = pd.concat(
    all_daily_data,
    ignore_index=True
)

# Save dataset
final_df.to_csv(
    "combined_stock_dataset.csv",
    index=False
)

print("\nDataset Saved Successfully")
print("Shape:", final_df.shape)
print("\nColumns:")
print(final_df.columns.tolist())