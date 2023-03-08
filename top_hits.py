import pandas as pd

df = pd.read_csv("banana_scores.csv")
df = df.query("score > 0.0")
df = df.sort_values(by="score", ascending=False)
df.to_csv("all_hits.csv", index=False)
df.head(500000).to_csv("top_hits.csv", index=False)