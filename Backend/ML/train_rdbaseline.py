import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

df = pd.read_csv("combined_stock_dataset.csv")

df["Date"] = pd.to_datetime(df["Date"])

# One-hot encode companies
df = pd.get_dummies(
    df,
    columns=["company"],
    drop_first=True
)

# Time-based split
train = df[df["Date"] < "2025-01-01"]
test = df[df["Date"] >= "2025-01-01"]

feature_columns = [
    "avg_sentiment",
    "message_count",
    "avg_likes",
    "avg_replies",
    "avg_reposts",
    "positive_ratio",
    "negative_ratio",
    "neutral_ratio",
    "price_return_1d",
    "price_return_3d",
    "price_return_7d",
    "sentiment_3d_avg",
    "sentiment_7d_avg",
    "volume_change"
]

# Add company dummy columns
company_cols = [
    col for col in df.columns
    if col.startswith("company_")
]

feature_columns.extend(company_cols)

X_train = train[feature_columns]
X_test = test[feature_columns]

y_train = train["target"]
y_test = test["target"]


model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

import joblib

model.fit(X_train, y_train)

preds = model.predict(X_test)

print("\nMetrics")
print("Accuracy :", accuracy_score(y_test, preds))
print("Precision:", precision_score(y_test, preds))
print("Recall   :", recall_score(y_test, preds))
print("F1 Score :", f1_score(y_test, preds))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, preds))

joblib.dump(
    model,
    "../models/stock_predictor.pkl"
)

print("Model saved successfully!")

print("\nTop Features")

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

print(
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .head(15)
)