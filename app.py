"""Streamlit demo. Run: streamlit run app.py"""
import joblib
import pandas as pd
import streamlit as st
from preprocess import clean_text

st.set_page_config(page_title="Social Media Sentiment Analyzer", page_icon="💬")
st.title("💬 Social Media Sentiment Analyzer")


@st.cache_resource
def load_model():
    return joblib.load("models/best_model.joblib")


try:
    model = load_model()
except FileNotFoundError:
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

tab1, tab2 = st.tabs(["Single post", "Batch (CSV)"])

with tab1:
    text = st.text_area("Paste a post or tweet", "I absolutely love the new update!")
    if st.button("Analyze"):
        pred = model.predict([clean_text(text)])[0]
        emoji = {"positive": "😊", "negative": "😠", "neutral": "😐"}.get(pred, "")
        st.subheader(f"{pred.capitalize()} {emoji}")

with tab2:
    up = st.file_uploader("Upload a CSV with a 'text' column", type="csv")
    if up:
        df = pd.read_csv(up)
        if "text" not in df.columns:
            st.error("CSV needs a 'text' column.")
        else:
            df["sentiment"] = model.predict(df["text"].map(clean_text))
            st.bar_chart(df["sentiment"].value_counts())
            st.dataframe(df)
            st.download_button("Download results", df.to_csv(index=False), "predictions.csv")
