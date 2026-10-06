"""Train and compare sentiment models, then save the best one.

Usage:
    python make_sample_data.py
    python train.py --data data/sample_posts.csv
    python train.py --data data/training.1600000.processed.noemoticon.csv --sample 100000
"""
import argparse
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data_loader import load_data
from preprocess import clean_text


def vader_predict(texts):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    out = []
    for t in texts:
        c = sia.polarity_scores(t)["compound"]
        out.append("positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral")
    return out


def make_pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", clf),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sample_posts.csv")
    ap.add_argument("--sample", type=int, default=None, help="subsample N rows for speed")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df = load_data(args.data)
    if args.sample and len(df) > args.sample:
        df = df.sample(args.sample, random_state=42)
    df["clean"] = df["text"].map(clean_text)
    print(f"Loaded {len(df)} posts. Class balance:\n{df['label'].value_counts()}\n")

    X_tr, X_te, y_tr, y_te, raw_tr, raw_te = train_test_split(
        df["clean"], df["label"], df["text"],
        test_size=args.test_size, random_state=42, stratify=df["label"])

    results, models = {}, {}

    # Baseline 1: VADER (rule-based, no training)
    try:
        preds = vader_predict(raw_te)
        results["VADER (rule-based)"] = (y_te, preds)
    except ImportError:
        print("vaderSentiment not installed, skipping baseline.")

    # Baseline 2/3: TF-IDF + classical ML
    for name, clf in [
        ("TF-IDF + Logistic Regression", LogisticRegression(max_iter=1000, C=2.0)),
        ("TF-IDF + Linear SVM", LinearSVC(C=1.0)),
    ]:
        pipe = make_pipeline(clf).fit(X_tr, y_tr)
        results[name] = (y_te, pipe.predict(X_te))
        models[name] = pipe

    # Evaluate
    summary = {}
    for name, (yt, yp) in results.items():
        acc = accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, average="macro")
        summary[name] = {"accuracy": round(acc, 4), "macro_f1": round(f1, 4)}
        print(f"=== {name} ===\nAccuracy: {acc:.4f}  Macro-F1: {f1:.4f}")
        print(classification_report(yt, yp, digits=3))

    with open("results/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Pick best trained model by macro-F1
    best_name = max(models, key=lambda n: summary[n]["macro_f1"])
    best = models[best_name]
    joblib.dump(best, "models/best_model.joblib")
    print(f"Saved best model: {best_name}")

    # Confusion matrix
    labels = sorted(df["label"].unique())
    cm = confusion_matrix(y_te, best.predict(X_te), labels=labels)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion matrix\n{best_name}")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150); plt.close()

    # Model comparison chart
    plt.figure(figsize=(7, 3.5))
    names = list(summary)
    plt.barh(names, [summary[n]["macro_f1"] for n in names], color="#4C78A8")
    plt.xlabel("Macro F1"); plt.xlim(0, 1); plt.title("Model comparison")
    plt.tight_layout(); plt.savefig("results/model_comparison.png", dpi=150); plt.close()

    # Top features per class (works for linear models)
    clf = best.named_steps["clf"]
    vocab = np.array(best.named_steps["tfidf"].get_feature_names_out())
    coefs = clf.coef_
    fig, axes = plt.subplots(1, len(clf.classes_), figsize=(4.5 * len(clf.classes_), 4))
    for ax, cls, row in zip(np.atleast_1d(axes), clf.classes_, coefs):
        idx = np.argsort(row)[-10:]
        ax.barh(vocab[idx], row[idx], color="#54A24B")
        ax.set_title(f"Top terms: {cls}")
    plt.tight_layout(); plt.savefig("results/top_terms.png", dpi=150); plt.close()

    # Error analysis: misclassified examples
    import pandas as pd
    errs = pd.DataFrame({"text": raw_te.values, "true": y_te.values,
                         "predicted": best.predict(X_te)})
    errs = errs[errs["true"] != errs["predicted"]]
    errs.head(50).to_csv("results/misclassified_examples.csv", index=False)
    print("Artifacts written to results/ and models/")


if __name__ == "__main__":
    main()
