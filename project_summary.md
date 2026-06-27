# Project notes

Demo pipeline combining text features (TF-IDF + abuse/positive lexicon density) with reply-graph stats (out-degree, pair frequency, neighbor count). Logistic regression in NumPy only.

**Why two cyberbullying folders:** Kaggle tweet data has no author→target edges, so graph features don't apply there. This repo uses a tiny synthetic CSV where reply pairs exist, mainly to show the feature engineering pattern.

**What I'd say in an interview:** walked through preprocessing, leakage-safe train/test split, interpretable weights, and why I'd want real graph data before claiming "multimodal" in production.