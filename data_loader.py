"""Loads either a generic CSV (text,label) or the Sentiment140 CSV."""
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    # Sentiment140 has no header: target, ids, date, flag, user, text (latin-1)
    try:
        df = pd.read_csv(path)
        if {"text", "label"}.issubset(df.columns):
            return df[["text", "label"]].dropna()
    except UnicodeDecodeError:
        pass
    df = pd.read_csv(path, encoding="latin-1", header=None,
                     names=["target", "ids", "date", "flag", "user", "text"])
    df["label"] = df["target"].map({0: "negative", 2: "neutral", 4: "positive"})
    return df[["text", "label"]].dropna()
