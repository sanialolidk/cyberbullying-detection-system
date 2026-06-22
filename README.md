# Cyberbullying Detection

Small demo pipeline — TF-IDF + lexicon features on the text, plus simple interaction stats from who replied to whom. Trained with plain NumPy logistic regression (no sklearn).

## Files

- `sample_cyberbullying_data.csv` — synthetic labeled posts with author/target ids
- `multimodal_cyberbullying_pipeline.py` — train + evaluate script
- `requirements.txt`
- `project_summary.md`

## Run

```bash
python3 multimodal_cyberbullying_pipeline.py --data sample_cyberbullying_data.csv
```

Prints metrics, top feature weights, and predictions on a held-out slice of the sample data.

## Notes

Dataset is tiny and made for demo purposes. The bigger cyberbullying project in my other repo uses real Kaggle tweets and drops the interaction stuff when the data doesn't support it — this one keeps both because the sample csv actually has reply pairs.