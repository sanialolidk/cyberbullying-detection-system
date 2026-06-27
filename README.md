# Cyberbullying Detection

Sania Thankan — small teaching/demo pipeline

TF-IDF + lexicon features on the text, plus simple interaction stats from who replied to whom. Trained with plain NumPy logistic regression — no sklearn, no deep learning, runs in seconds.

## Which repo is this?

| This repo | `cyberbullying_project` (local, not on GitHub) |
|-----------|--------------------------------------------------|
| 24-row synthetic CSV with `user_id` / `target_user_id` | ~47k real Kaggle tweets |
| Text **and** reply-graph features | Text only (dataset has no real interaction graph) |
| Single script, NumPy LR | sklearn LR, CV, threshold tuning, Streamlit demo |

This one exists to show the interaction-feature idea on data that actually has reply pairs. The bigger project is what I'd put on a resume for real evaluation.

## Files

- `sample_cyberbullying_data.csv` — synthetic labeled posts
- `pipeline.py` — train + evaluate
- `requirements.txt`
- `project_summary.md` — short notes for portfolio copy

## Run

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python pipeline.py --data sample_cyberbullying_data.csv
```

Prints train/test metrics, top feature weights, and held-out predictions. Features are fit on train only (no leakage).

## Notes

Perfect scores on the sample set are expected — 24 rows, clean synthetic labels. Don't read F1=1.0 as production-ready.