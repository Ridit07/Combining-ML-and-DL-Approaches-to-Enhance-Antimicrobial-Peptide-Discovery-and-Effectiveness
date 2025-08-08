# Combining ML & DL to Enhance Antimicrobial Peptide (AMP) Discovery

A research repo for our paper on fusing machine learning (ML) and deep learning (DL) to predict, design, and validate antimicrobial peptides. We combine sequence-level models (SVM, Random Forest, XGBoost, Logistic Regression, RNN) with image/structure models (custom CNN, VGG16, ResNet50, InceptionV3), and use XAI (SHAP/LIME) to interpret what actually drives predictions. The work targets the AMR (antimicrobial resistance) problem and closes the gap between in-silico ranking and lab validation concepts like SAXS/killing assays.

---

## Short description

Antimicrobial peptides are promising antibiotic alternatives—but finding effective ones is hard. We integrate sequence features and structural signals, evaluate a spectrum of ML/DL models, and interpret them with SHAP/LIME to surface the physicochemical patterns that matter (e.g., molecular weight, length, hydrophobicity). The goal: a practical, reproducible path to identify and refine high-value AMP candidates.

---

## Highlights

- **Problem**: Existing AMP discovery often treats sequence and structure separately, missing cross-modal dependencies that drive real efficacy.  
- **Approach**: Dual track—tabular/sequence ML + structural/image DL—with careful preprocessing, model selection, and hyperparameter tuning.  
- **Interpretability**: SHAP (global) + LIME (local) to explain feature influence and single-prediction rationale; used to sanity-check “what the model learned”.  
- **Outcome**: Consistent strong results on sequence ML (RF/XGB/SVM/LogReg) and best-in-class accuracy from ResNet50 on image tasks; model trade-offs documented via PR/ROC curves.

---

## Data sources (as used/reviewed)

- AMP sequence databases: **CAMP**, **DBAASP**, **DRAMP**, **YADAMP**; negatives from **UniProt**.  
- Structures/imagery: **RCSB PDB** (for structural views/derivatives).

> The paper also surveys three cornerstone studies: SVM + Pareto frontier for membrane activity, dual-LSTM generation/classification for novel AMPs, and **AMPlify** (BiLSTM + attention) for WHO-priority pathogens. We align our pipeline with those directions.

---

## Methods (what we actually ran)

### Sequence / tabular models
- **Random Forest**, **XGBoost**, **Logistic Regression**, **SVM**, **RNN**.  
- Preprocessing: cleaning, type coercion, feature engineering (MW, length, charge, hydrophobicity indices, aromaticity, isoelectric point), class balance checks.  
- Tuning: `GridSearchCV` (n_estimators, depth, min_samples_* for RF; C/γ/kernel for SVM, etc.).  
- **Explainability**: SHAP summary & dependence plots; LIME per-sample attributions.

### Image / structure models
- **Custom CNN** baseline.  
- **Transfer learning**: VGG16, ResNet50, InceptionV3 (frozen → unfrozen heads, small data regimes, regularization).  
- Training diagnostics: train/val curves, precision–recall analysis for threshold choice.

---

## Results (as reported in the paper)

### Sequence models (structural bioinformatics classification)
- **Random Forest**: **Accuracy 98.39%**, **ROC AUC 99.79%**  
- **XGBoost**: **Accuracy 98.36%**, **ROC AUC 99.80%**  
- **Logistic Regression**: Accuracy 97.47%, ROC AUC 99.27%  
- **SVM**: Accuracy 98.10%, ROC AUC 99.67%  
- Additional study variant: XGBoost **ROC AUC ≈ 0.92** under a different sequential setup (dataset/task change noted in paper).

### Image models
- **Custom CNN**: ~**87%** accuracy  
- **VGG16**: ~**90%** accuracy  
- **InceptionV3**: ~**91%** accuracy  
- **ResNet50**: **~92%** accuracy (best overall for image classification in our setting)  
- PR curves show ResNet50’s precision strength across recall ranges; RNN favored when high recall is critical on sequence tasks.

### What mattered (XAI)
- SHAP ranked **Molecular Weight** and **Length** as top influences; hydrophobicity (GRAVY), aromaticity, and isoelectric point showed nuanced, sometimes nonlinear effects.  
- LIME clarified per-sample decisions (pro/con feature bars with predicted probabilities). These checks reduced the risk of spurious correlations.
