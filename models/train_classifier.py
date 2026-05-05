"""Train and serialize the pathogen Random Forest classifier."""
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

OUT_PATH = os.path.join(os.path.dirname(__file__), "rf_pathogen.pkl")

def generate_synthetic_dataset(n=5000, seed=7):
    rng = np.random.default_rng(seed)
    mean_rmsd = rng.uniform(0.1, 5.0, n)
    max_rmsd  = mean_rmsd + rng.uniform(0.5, 4.0, n)
    num_hotspots = rng.integers(0, 25, n)
    num_active   = rng.integers(0, 10, n)
    hydro_change = rng.uniform(0, 30, n)
    charge_change = rng.uniform(0, 8, n)

    # ground-truth rule (with noise) — pathogenic if disruptive
    score = (
        (num_hotspots > 5).astype(int) * 1.2 +
        (mean_rmsd > 1.5).astype(int) * 1.0 +
        (num_active > 3).astype(int) * 0.8 +
        (charge_change > 3).astype(int) * 0.5 +
        rng.normal(0, 0.4, n)
    )
    y = (score > 1.5).astype(int)

    X = np.column_stack([mean_rmsd, max_rmsd, num_hotspots,
                         num_active, hydro_change, charge_change])
    return X, y

def train_and_save():
    X, y = generate_synthetic_dataset()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, n_jobs=-1, random_state=42
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)[:, 1]
    print(f"✅ Accuracy: {accuracy_score(yte, pred):.3f}")
    print(f"✅ ROC-AUC : {roc_auc_score(yte, proba):.3f}")
    joblib.dump(clf, OUT_PATH)
    print(f"💾 Saved → {OUT_PATH}")
    return clf

if __name__ == "__main__":
    train_and_save()