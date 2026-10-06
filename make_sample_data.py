"""Generates a small synthetic labelled dataset so the project runs out of the box.
Replace with a real dataset (e.g. Sentiment140) for meaningful results."""
import random
import pandas as pd

random.seed(42)
subjects = ["the new phone", "this movie", "the service", "the update", "that restaurant",
            "the airline", "my delivery", "the concert", "the new app", "customer support"]
pos = ["is amazing", "is absolutely fantastic", "exceeded my expectations", "made my day",
       "is so good, highly recommend", "works perfectly", "is the best ever", "is really impressive"]
neg = ["is terrible", "is a total waste of money", "was so disappointing", "ruined my day",
       "is awful, do not buy", "keeps crashing", "is the worst ever", "is really frustrating"]
neu = ["is okay I guess", "is available from Monday", "was announced today", "is what it is",
       "arrived on Tuesday", "has a new version out", "is open until 9pm", "is on the schedule"]
tails = ["", " #review", " :)", " !!!", " http://t.co/abc", " @friend", ""]
rows = []
for label, bank in [("positive", pos), ("negative", neg), ("neutral", neu)]:
    for _ in range(150):
        t = f"{random.choice(subjects)} {random.choice(bank)}{random.choice(tails)}"
        rows.append((t.capitalize(), label))
random.shuffle(rows)
pd.DataFrame(rows, columns=["text", "label"]).to_csv("data/sample_posts.csv", index=False)
print("wrote data/sample_posts.csv", len(rows))
