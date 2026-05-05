"""Pathogenicity prediction using a pre-trained Random Forest."""
import os
import numpy as np
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "rf_pathogen.pkl")

FEATURE_NAMES = [
    "mean_rmsd", "max_rmsd", "num_hotspots",
    "num_mutations_in_active_site", "hydrophobicity_change", "charge_change"
]

# Hydrophobicity (Kyte-Doolittle) and charge approximations
HYDRO = {
    'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'E':-3.5,'Q':-3.5,'G':-0.4,
    'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
    'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2,'X':0.0
}
CHARGE = {'D':-1,'E':-1,'K':1,'R':1,'H':0.5}

def model_exists():
    return os.path.exists(MODEL_PATH)

def load_model():
    return joblib.load(MODEL_PATH)

def extract_features(mut_df, active_site_range=None):
    """Compute the 6-D feature vector from mutation DataFrame."""
    if active_site_range is None:
        # default: middle 30% of sequence assumed to be active site
        n = len(mut_df)
        lo, hi = int(n*0.35), int(n*0.65)
        active_site_range = (lo, hi)

    devs = mut_df["Deviation (Å)"].values
    mean_rmsd = float(np.mean(devs)) if len(devs) else 0.0
    max_rmsd = float(np.max(devs)) if len(devs) else 0.0
    num_hotspots = int(mut_df["Is Hotspot"].sum())

    in_site = mut_df.iloc[active_site_range[0]:active_site_range[1]]
    num_mut_active = int(in_site["Is Mutation"].sum())

    # Hydrophobicity / charge change summed over mutations only
    hydro_change = 0.0
    charge_change = 0.0
    for _, row in mut_df[mut_df["Is Mutation"]].iterrows():
        hydro_change += abs(HYDRO.get(row["Var AA"], 0) - HYDRO.get(row["Ref AA"], 0))
        charge_change += abs(CHARGE.get(row["Var AA"], 0) - CHARGE.get(row["Ref AA"], 0))

    return np.array([[mean_rmsd, max_rmsd, num_hotspots,
                      num_mut_active, hydro_change, charge_change]])

def predict(mut_df):
    """Returns dict with prediction + probability + feature importances."""
    model = load_model()
    X = extract_features(mut_df)
    proba = model.predict_proba(X)[0]
    pred = int(model.predict(X)[0])
    return {
        "label": "POTENTIALLY PATHOGENIC" if pred == 1 else "BENIGN",
        "is_pathogenic": bool(pred),
        "confidence": float(proba[pred]),
        "prob_pathogenic": float(proba[1]) if len(proba) > 1 else 0.0,
        "features": dict(zip(FEATURE_NAMES, X[0].tolist())),
        "importances": dict(zip(FEATURE_NAMES, model.feature_importances_.tolist()))
    }