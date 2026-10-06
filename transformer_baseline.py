"""Optional: evaluate a pretrained Twitter RoBERTa model on the same test split.
Needs `pip install transformers torch` and internet access to download the model.

Usage: python transformer_baseline.py --data data/sample_posts.csv --n 500
"""
import argparse
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import pipeline
from data_loader import load_data

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/sample_posts.csv")
ap.add_argument("--n", type=int, default=500)
args = ap.parse_args()

df = load_data(args.data)
_, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
test = test.head(args.n)

clf = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest",
               truncation=True, max_length=128)
preds = [r["label"].lower() for r in clf(test["text"].tolist(), batch_size=32)]
print(classification_report(test["label"], preds, digits=3))
