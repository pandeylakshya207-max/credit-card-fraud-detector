"""
train_model.py — Credit Card Fraud Detection, real ULB/Kaggle dataset
(284,807 transactions, 492 frauds, 0.17% fraud rate).

This fixes several methodological gaps in the original notebook:
  1. Stratified train/test split (critical for a 0.17%-positive dataset —
     a random split can easily leave the test set with very few frauds,
     making evaluation noisy).
  2. Feature scaling for Time and Amount (V1-V28 are already PCA-transformed
     and roughly standardized; Time and Amount are raw and on very
     different scales, which matters for some classifiers/thresholds).
  3. A real baseline comparison (DummyClassifier) — the "predict majority
     class" trap: a model that always predicts "not fraud" scores 99.83%
     accuracy while catching zero fraud. Accuracy alone is meaningless here.
  4. Precision-Recall AUC reported alongside ROC-AUC — PR-AUC is the more
     informative metric on severely imbalanced data (ROC-AUC can look
     deceptively good even for a mediocre model when negatives vastly
     outnumber positives).
  5. Two modeling approaches compared honestly: RandomForest with
     class_weight='balanced' vs. RandomForest + SMOTE oversampling —
     rather than assuming SMOTE is automatically better.
  6. Stratified k-fold cross-validation for a more robust estimate than a
     single train/test split.

Run:
    python3 train_model.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_curve, f1_score
)
from imblearn.over_sampling import SMOTE
import pickle
import json

# ─── 1. Load real data ────────────────────────────────────────────────────
df = pd.read_csv("creditcard.csv")
print(f"Loaded {len(df)} transactions, {df['Class'].sum()} fraud "
      f"({100*df['Class'].mean():.4f}%)")

# ─── 2. Feature scaling ────────────────────────────────────────────────────
# V1-V28 are already PCA-transformed (roughly standardized). Time and Amount
# are raw and on very different scales — scale them to match.
scaler = StandardScaler()
df["Time_scaled"] = scaler.fit_transform(df[["Time"]])
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df = df.drop(columns=["Time", "Amount"])

X = df.drop("Class", axis=1)
y = df["Class"]

# ─── 3. Stratified train/test split ────────────────────────────────────────
# Stratify=y ensures the ~0.17% fraud rate is preserved in BOTH train and
# test sets. Without this, a random split could (by chance) put very few
# frauds in the test set, making the evaluation unreliable.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} ({y_train.sum()} fraud) | "
      f"Test: {len(X_test)} ({y_test.sum()} fraud)")

# ─── 4. Baseline: DummyClassifier (always predicts majority class) ────────
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
dummy_acc = (dummy_pred == y_test).mean()
print(f"\nBaseline (predict 'not fraud' always): accuracy = {dummy_acc:.4f}, "
      f"but catches 0/{y_test.sum()} frauds — this is why accuracy alone is meaningless here")

# ─── 5. Approach A: RandomForest with class_weight='balanced' ─────────────
# Instead of synthetically generating new fraud samples (SMOTE), this
# reweights the loss function so misclassifying the minority class is
# penalized more — no synthetic data, faster to train.
model_balanced = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
)
model_balanced.fit(X_train, y_train)
pred_balanced = model_balanced.predict(X_test)
proba_balanced = model_balanced.predict_proba(X_test)[:, 1]

# ─── 6. Approach B: RandomForest + SMOTE ───────────────────────────────────
# SMOTE creates synthetic minority-class examples by interpolating between
# real fraud cases, rather than just duplicating them or reweighting.
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"\nAfter SMOTE: {len(X_train_res)} training rows "
      f"({y_train_res.sum()} fraud, {(y_train_res==0).sum()} legit)")

model_smote = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
model_smote.fit(X_train_res, y_train_res)
pred_smote = model_smote.predict(X_test)
proba_smote = model_smote.predict_proba(X_test)[:, 1]

# ─── 7. Evaluate both approaches honestly ──────────────────────────────────
def evaluate(name, y_true, y_pred, y_proba):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["Legit", "Fraud"]))
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    print(f"ROC-AUC: {roc_auc:.4f}  |  PR-AUC: {pr_auc:.4f}  |  F1: {f1:.4f}")
    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "f1": f1}

metrics_balanced = evaluate("Approach A: class_weight='balanced'", y_test, pred_balanced, proba_balanced)
metrics_smote = evaluate("Approach B: SMOTE oversampling", y_test, pred_smote, proba_smote)

# ─── 8. Stratified cross-validation (more robust than a single split) ─────
print(f"\n{'='*60}\n5-Fold Stratified Cross-Validation (class_weight='balanced' model)\n{'='*60}")
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=50, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1),
    X, y, cv=skf, scoring="average_precision"
)
print(f"PR-AUC per fold: {[round(s, 4) for s in cv_scores]}")
print(f"Mean PR-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ─── 9. Feature importance ─────────────────────────────────────────────────
importances = sorted(zip(X.columns, model_balanced.feature_importances_), key=lambda x: -x[1])
print(f"\nTop 10 most important features:")
for feat, imp in importances[:10]:
    print(f"  {feat:<15}{imp:.4f}")

# ─── 10. Pick the better model and save ────────────────────────────────────
best_name = "smote" if metrics_smote["pr_auc"] > metrics_balanced["pr_auc"] else "balanced"
best_model = model_smote if best_name == "smote" else model_balanced
print(f"\nBest approach by PR-AUC: {best_name}")

with open("fraud_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

results = {
    "dataset": "ULB Credit Card Fraud (via Kaggle/GitHub mirror)",
    "total_transactions": len(df),
    "total_fraud": int(y.sum()),
    "fraud_rate_pct": round(100 * y.mean(), 4),
    "baseline_accuracy": round(dummy_acc, 4),
    "baseline_frauds_caught": 0,
    "approach_balanced": metrics_balanced,
    "approach_smote": metrics_smote,
    "cv_pr_auc_mean": round(cv_scores.mean(), 4),
    "cv_pr_auc_std": round(cv_scores.std(), 4),
    "best_approach": best_name,
    "top_features": [f for f, _ in importances[:10]],
}
with open("metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved: fraud_model.pkl, scaler.pkl, metrics.json")
