# Credit Card Fraud Detector

A fraud detection system built on the real, well-known ULB (Université Libre de
Bruxelles) credit card fraud dataset — 284,807 European cardholder transactions
from September 2013, with 492 confirmed frauds (0.1727% of all transactions).

## The problem

Fraud detection on this dataset is a severe class imbalance problem: fraud makes
up less than 0.2% of transactions. A model that always predicts "not fraud"
scores **99.83% accuracy** while catching zero fraud — which is exactly why
accuracy alone is a meaningless metric here. This project is built around
evaluating and comparing approaches honestly, not chasing an inflated headline
number.

## Results

Two approaches were compared on an identical stratified train/test split
(80/20, fraud rate preserved in both sets):

| Approach | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| **`class_weight='balanced'`** | 0.96 | 0.73 | **0.83** | **0.843** | 0.948 |
| SMOTE oversampling | 0.52 | 0.85 | 0.64 | 0.809 | 0.980 |
| Naive baseline (predict "not fraud") | — | 0.00 | — | — | 0.500 |

**Key finding:** SMOTE catches more fraud (85% recall vs. 73%) but at a much
higher false-positive cost — precision drops from 0.96 to 0.52, meaning roughly
half of SMOTE's fraud flags are false alarms. Notably, **ROC-AUC alone would
misleadingly favor SMOTE** (0.980 vs. 0.948) — on data this imbalanced, PR-AUC
and the precision/recall tradeoff tell the more honest, decision-relevant story.
The `class_weight='balanced'` approach was selected as the better model by PR-AUC.

Validated with 3-fold stratified cross-validation: PR-AUC = 0.828 ± 0.023,
confirming the result isn't a fluke of one particular split.

*Note: the SMOTE model used fewer trees and capped depth (50 estimators,
max_depth=12) compared to the balanced approach (100 estimators, unlimited
depth), due to training-time constraints on the much larger SMOTE-resampled
training set (454,902 rows vs. 227,845). This isn't a perfectly matched-capacity
comparison — a fully fair comparison would give both approaches equal capacity.*

### Top predictive features
`V14`, `V10`, `V4`, `V17`, `V11`, `V12`, `V3`, `V16`, `V7`, `V2` — all PCA-
anonymized features from the original dataset (the dataset's real feature names
are withheld for confidentiality by the original data provider).

## Methodology

- **Stratified train/test split** — critical on a 0.17%-positive dataset; a
  random split could leave very few frauds in the test set, making evaluation
  unreliable
- **Feature scaling** — `Time` and `Amount` are raw and on different scales
  than the PCA-transformed `V1`-`V28` features; scaled with `StandardScaler`
- **Baseline comparison** — a `DummyClassifier` predicting the majority class
  every time, to make clear why accuracy alone is misleading here
- **Two approaches compared honestly** rather than assuming SMOTE is
  automatically better than class reweighting
- **PR-AUC reported alongside ROC-AUC** — PR-AUC is the more informative
  metric for severely imbalanced classification
- **Stratified k-fold cross-validation** for a more robust estimate than a
  single train/test split

## Dataset

Sourced from the original ULB Machine Learning Group's Credit Card Fraud
Detection dataset (Worldline and ULB, collaboration on big data mining and
fraud detection). Not included in this repo (102MB, excluded via
`.gitignore`) — download it yourself:

```bash
curl -o creditcard.csv https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv
```

## Running this yourself

```bash
pip install -r requirements.txt
python train_model.py
```

This reproduces the exact results above: loads the real dataset, splits it,
trains both approaches, evaluates honestly, cross-validates, and saves
`fraud_model.pkl`, `time_scaler.pkl`, `amount_scaler.pkl`, and `metrics.json`.

`fix_scalers.py` is a one-time historical fix (see below) — `train_model.py`
already produces correct scalers directly; you don't need to run it separately.

## Limitations

- The SMOTE vs. balanced comparison isn't perfectly capacity-matched (see note
  above) — a rigorous ablation would equalize tree count and depth
- No hyperparameter search was performed (e.g., grid search over
  `max_depth`, `n_estimators`) — current settings are reasonable defaults,
  not tuned
- The dataset is from September 2013; fraud patterns evolve over time, and a
  model trained on this data would need retraining on recent transactions
  before any real-world use
- The `/predict` endpoint requires all 28 anonymized `V1`-`V28` features as
  input, which in a real production system would come from an upstream
  feature pipeline, not be supplied directly by a caller

## A real bug found and fixed during development

The original scaling code used a single shared `StandardScaler` object,
calling `fit_transform()` on it twice — once for `Time`, once for `Amount`.
The second call silently overwrote the first's fitted statistics, so the
persisted scaler only retained `Amount`'s mean/std. Using it to scale a new
`Time` value at inference would have silently applied the wrong statistics.
The model itself was unaffected (it trained on correctly-computed values at
the time each line ran) — only the *persisted* scaler artifact was broken.
Fixed by using two separate scaler objects (`time_scaler.pkl`,
`amount_scaler.pkl`); see `fix_scalers.py` for the fix and
`tests/test_all.py` for a regression test that would catch this specific
bug if it recurred.

## API

A FastAPI inference service (`api/main.py`) exposes the trained model:

```bash
uvicorn api.main:app --reload
```

- `GET /health` — model load status
- `POST /predict` — takes transaction features, returns fraud probability
  and classification (see `/docs` for interactive API documentation and
  a full example payload)
- `?threshold=0.3` query param — adjust the fraud-classification threshold
  (default 0.5); lower thresholds catch more fraud at the cost of more
  false positives, matching the precision/recall tradeoff documented above

## Testing

```bash
python tests/test_all.py
```

9 tests covering: model/scaler loading, a regression test specifically for
the shared-scaler bug (so it can't silently reappear), API health/predict
endpoints against known real legit and fraud cases from the dataset, input
validation, and metrics sanity checks. Runs automatically via GitHub Actions
on every push.

## Tech stack

Python, pandas, scikit-learn, imbalanced-learn (SMOTE), FastAPI, uvicorn

## License

See [LICENSE](LICENSE).
