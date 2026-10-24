# Adult Stroke Risk Prediction Using Lifestyle and Clinical Factors

A machine learning mini project that predicts whether a person has had a stroke from lifestyle and clinical factors, and compares two ways of handling severe class imbalance (class weighting vs. SMOTENC oversampling).

> **Disclaimer:** This is an educational demonstration. It is **not** a diagnostic tool and must **not** be used for clinical, diagnostic or public-health decisions.

**Course:** Machine Learning (ITPCC507), TY IT-B
**Institute:** AISSMS Institute of Information Technology, Pune (Savitribai Phule Pune University)
**Academic year:** 2026-27

## Authors

- Sunny Rajendra Wankhede
- Govinda Narayan Shinde

**Guide:** Dr. Mrunal K. Pathak

## Overview

- **Task:** Binary classification, stroke (1) vs. no stroke (0)
- **Dataset:** `adult_stroke_lifestyle_clinical_dataset.csv`, 5,110 records and 10 columns (249 stroke cases, 4.87%)
- **Models:** Logistic Regression, Random Forest, LightGBM
- **Imbalance strategies:** Class weights (`balanced`) and SMOTENC (applied only inside training folds)
- **Configurations:** 3 models x 2 strategies = 6
- **Tuning:** `RandomizedSearchCV`, 5-fold stratified CV, scored on PR-AUC (average precision)
- **Threshold tuning:** F1-optimal cut-off chosen from out-of-fold training predictions, then applied once to the test set
- **Split:** Stratified 80% / 20% (4,087 train / 1,022 test), `random_state=42`

## Features

| Column | Type | Description |
|---|---|---|
| gender | Categorical | Female, Male (the single "Other" record is removed) |
| age | Numeric | Age in years |
| hypertension | Binary | 1 if hypertensive |
| heart_disease | Binary | 1 if heart disease present |
| work_type | Categorical | Private, Self-employed, children, Govt_job, Never_worked |
| avg_glucose_level | Numeric | Average glucose level |
| bmi | Numeric | Body-mass index |
| bmi_missing | Binary | 1 if BMI was missing in the source and imputed |
| smoking_status | Categorical | never smoked, formerly smoked, smokes, Unknown |
| stroke | Binary (target) | 1 = stroke, 0 = no stroke |

## Workflow

1. Load data and check shape, missing values and duplicates
2. Exploratory data analysis (class balance, distributions, stroke rate by factor, correlation heatmap)
3. Leakage-free preprocessing in an `imblearn` pipeline (ordinal encoding, one-hot encoding, scaling for Logistic Regression)
4. Stratified train/test split
5. Train and tune all six model/strategy combinations
6. Tune the decision threshold on out-of-fold predictions
7. Evaluate on the held-out test set (precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices)
8. Interpret the models (LightGBM feature importance, scaled Logistic Regression coefficients)
9. Predict risk for example patients

## Results (test set, tuned threshold)

| Model | Strategy | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| **LightGBM** | **Class Weight** | 0.730 | 0.2596 | 0.540 | **0.3506** | 0.8397 | 0.2291 |
| Logistic Regression | Class Weight | 0.762 | 0.2417 | 0.580 | 0.3412 | 0.8385 | 0.2278 |
| Logistic Regression | SMOTE | 0.732 | 0.2214 | 0.580 | 0.3204 | 0.8341 | 0.2676 |
| LightGBM | SMOTE | 0.732 | 0.2185 | 0.520 | 0.3077 | 0.8078 | 0.2252 |
| Random Forest | SMOTE | 0.641 | 0.1806 | 0.560 | 0.2732 | 0.7996 | 0.2026 |
| Random Forest | Class Weight | 0.713 | 0.2245 | 0.220 | 0.2222 | 0.8178 | 0.2166 |

**Key findings**

- LightGBM with class weights was the best model (F1 = 0.351, ROC-AUC = 0.840).
- Accuracy is misleading on this data; an "always no stroke" model scores about 95%.
- Threshold tuning raised the best model's F1 from 0.229 to 0.351, trading recall for fewer false alarms.
- Class weighting was at least as good as SMOTE and simpler.
- Age was the most important factor in both LightGBM and Logistic Regression, followed by glucose level, BMI and hypertension.

## Limitations

- Only 249 stroke cases (50 in the test set), so all metrics are noisy and small differences between models are not statistically reliable.
- Results come from a single train/test split with no external validation.
- Precision is low (0.26): about three of four flagged people did not have a stroke.
- The dataset includes minors (916 records aged 18 or younger) despite the title.
- `bmi_missing` is a data-collection artefact, not a clinical factor.
- "Unknown" smoking status and "children" work type partly act as age proxies.
- No timing information, so the models describe association, not future-stroke prediction.

## Getting Started

### Requirements

- Python 3
- 64-bit CPU, 4 GB RAM minimum (8 GB recommended)

| Package | Version |
|---|---|
| pandas | >= 2.0 |
| numpy | >= 1.24 |
| scikit-learn | >= 1.3 |
| imbalanced-learn | >= 0.11 |
| lightgbm | >= 4.0 |
| matplotlib | >= 3.7 |
| seaborn | >= 0.12 |

### Install and run

```bash
git clone <your-repo-url>
cd <your-repo-folder>

pip install pandas numpy scikit-learn imbalanced-learn lightgbm matplotlib seaborn jupyter
jupyter notebook stroke_risk_prediction.ipynb
```

Place `adult_stroke_lifestyle_clinical_dataset.csv` in the same folder as the notebook, or change `DATA_PATH` in the notebook. The notebook can be re-run on other stroke datasets with the same columns by replacing the CSV.

### Predict for a new patient

The last notebook cell takes a Python dictionary and returns a probability and risk class:

```python
new_patient = pd.DataFrame([{
    "gender": "Male", "work_type": "Private", "smoking_status": "formerly smoked",
    "hypertension": 1, "heart_disease": 0, "bmi_missing": 0,
    "age": 68, "avg_glucose_level": 210.0, "bmi": 31.5
}])[feature_order]

risk = best_pipe.predict_proba(new_patient)[0, 1]
print("HIGH RISK" if risk >= best_thr else "LOW RISK")
```

## Repository Structure

```
.
├── stroke_risk_prediction.ipynb                  # Full commented notebook
├── adult_stroke_lifestyle_clinical_dataset.csv   # Dataset
├── Stroke_Risk_Prediction_Mini_Project_Report.pdf # Project report
└── README.md
```

## Future Work

- Restrict the data to adults
- Repeated cross-validation and probability calibration
- Local explanations with SHAP or LIME
- Other imbalance techniques and models
- Validation on an independent dataset
- A simple web app for demonstration

## References

1. Feigin, V. L., et al. (2022). World Stroke Organization (WSO): Global Stroke Fact Sheet 2022. *International Journal of Stroke*, 17(1), 18-29.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830.
3. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
4. Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 30*.
5. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321-357.
6. Lemaitre, G., Nogueira, F., Aridas, C. K. (2017). Imbalanced-learn. *JMLR*, 18(17), 1-5.
7. Saito, T., Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot. *PLoS ONE*, 10(3), e0118432.
8. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *SciPy 2010*, 56-61.
9. Hosmer, D. W., Lemeshow, S., Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.
10. Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering*, 9(3), 90-95.

## License

Add a license of your choice (for example MIT) before publishing. This project is for educational use only.
