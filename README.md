# Sentiment Analysis of Social Media

A complete course project: preprocessing, baseline and ML models, evaluation,
error analysis and a Streamlit demo app.

## Quick start
```bash
pip install -r requirements.txt
python make_sample_data.py          # tiny synthetic dataset so everything runs
python train.py                     # trains + evaluates + saves best model
streamlit run app.py                # interactive demo
```

## Use a real dataset (do this for your final results)
1. Download **Sentiment140** from Kaggle (`training.1600000.processed.noemoticon.csv`)
   and put it in `data/`. Or use any CSV with `text` and `label` columns
   (`positive` / `negative` / `neutral`).
2. Train on a subsample for speed:
   ```bash
   python train.py --data data/training.1600000.processed.noemoticon.csv --sample 100000
   ```
3. Optional transformer comparison:
   ```bash
   python transformer_baseline.py --data data/<your file>.csv --n 1000
   ```

> The synthetic sample data gives ~100% accuracy because it is built from
> templates. Real results on Sentiment140 are typically ~76-80% (binary) for
> TF-IDF models and higher for transformers. Report those, not the sample numbers.

## Project structure
| File | Purpose |
|---|---|
| `preprocess.py` | Cleaning: URLs, mentions, hashtags, emoticons, elongated words |
| `data_loader.py` | Loads generic CSV or Sentiment140 |
| `train.py` | VADER baseline, TF-IDF + LR / SVM, metrics, plots, saves best model |
| `transformer_baseline.py` | Optional pretrained RoBERTa comparison |
| `app.py` | Streamlit demo (single post + CSV batch) |
| `results/` | Metrics JSON, confusion matrix, model comparison, top terms, error cases |

## Suggested report outline
1. **Introduction**: why social media sentiment matters (brands, politics, public health).
2. **Dataset**: source, size, class balance, collection ethics and privacy.
3. **Preprocessing**: what you clean and why (show before/after examples).
4. **Methods**: VADER, TF-IDF + LR/SVM, (optional) RoBERTa. Explain each briefly.
5. **Results**: table from `results/metrics.json`, confusion matrix, model comparison chart.
6. **Error analysis**: use `results/misclassified_examples.csv` (sarcasm, negation, slang, emojis).
7. **Limitations & ethics**: sarcasm, dataset bias, language/dialect, representativeness.
8. **Future work**: multilingual models, aspect-based sentiment, time-series trends.
