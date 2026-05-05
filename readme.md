# 🧬 ProteinScope: Mutation & Pathogen Detection Dashboard
# ProteinScope Command Center 🧬
Advanced Molecular Alignment & Bio-Threat Assessment System

### How to Run Locally:
1. Open terminal and install requirements: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`

> Real-time 3D protein structural comparison and pathogenicity prediction in your browser.

## 🎯 What It Does
Drop two PDB files (a reference protein and a variant) → ProteinScope:
1. **Parses** the atomic coordinates
2. **Aligns** them via the Kabsch algorithm (SVD-based optimal rotation)
3. **Computes** global + per-residue RMSD
4. **Renders** an interactive, rotatable 3D overlay
5. **Flags** mutation hotspots above your threshold
6. **Predicts** pathogenicity with a Random Forest classifier

## 🛠️ Stack
Streamlit · Biopython · NumPy · SciPy · py3Dmol + stmol · Plotly · scikit-learn

## ⚡ Install & Run
```bash
git clone <your-repo>
cd proteinscope
pip install -r requirements.txt
python models/train_classifier.py     # one-time: builds rf_pathogen.pkl
streamlit run app.py
```

## 📂 Project Structure
```
proteinscope/
├── app.py
├── core/
│   ├── pdb_parser.py
│   ├── aligner.py
│   ├── mutation_detector.py
│   ├── pathogen_classifier.py
│   └── synthetic_data.py
├── models/
│   ├── train_classifier.py
│   └── rf_pathogen.pkl
├── requirements.txt
└── README.md
```

## 🎤 Hackathon Pitch
- **Hook:** "Every year, millions of genetic variants are discovered — but 99% are 'variants of unknown significance.' Doctors can't act on them."
- **Problem:** Manually evaluating 3D structural impact takes weeks per variant.
- **Solution:** ProteinScope does it in **2 seconds**, in the browser, with a draggable 3D overlay any clinician can read.
- **Impact:** Faster diagnostics for cancer, rare diseases, and pandemic preparedness.

## 📸 Screenshots
*(Add: 3D overlay tab · RMSD chart · Pathogen risk panel)*