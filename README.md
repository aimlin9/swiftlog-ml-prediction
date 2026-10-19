# SwiftLog Logistics: Intelligent Logistics Prediction & Supply Chain Optimization

Week 5 | Machine Learning Track (Mission ML05) | AnalystLab Africa Experience Lab
Author: Gyimah Ramsey Opoku

SwiftLog Logistics Limited is a fictional client and all data is synthetic.

## Headline result

Four supervised models were built on four logistics problems, with five algorithms each, cross-validation, tuning, feature
selection and class-imbalance handling, and every one was compared with a naive baseline. **None beats its baseline.**

| Problem | Type | Selected model | Test result | Baseline |
|---|---|---|---|---|
| Delivery delay | Classification | Logistic Regression (tuned) | ROC-AUC 0.520 (95% CI 0.476 to 0.563) | 0.50 |
| Product return | Classification | Logistic Regression (tuned) | ROC-AUC 0.491 (95% CI 0.420 to 0.565) | 0.50 |
| Demand forecasting | Regression | Ridge Regression (tuned) | RMSE 14.75, R2 -0.058 | Mean: RMSE 14.79 |
| Customer value | Regression | Ridge Regression (tuned) | RMSE 6320, R2 -0.003 | Mean: RMSE 6313 |

Correlations, 22 hypothesis tests (1 significant, about 1 expected by chance), mutual information and label-shuffled models
all agree that the fields provided carry no usable signal. The deliverable is therefore a tested, re-runnable workflow, an
evidence-based statement of what the current data can and cannot support, and a data-collection roadmap. See the report.

Supplier Reliability Prediction was **not built**: no table links suppliers to any shipment or product.

## Folder structure

```
Week5_MachineLearning_SwiftLog_Ramsey/
  Data/            Original_Datasets/, Processed_Datasets/, Feature_Engineered_Dataset.csv
  Notebook/        Week5_MachineLearning.ipynb   (executed, outputs included)
  Models/          Best_Model.pkl, Model_Comparison.xlsx, Feature_Importance.png, metrics.json, All_Models/
  App/             app.py, requirements.txt, assets/
  Reports/         Technical_Report.pdf, Executive_Summary.pdf, Figures/
  Presentation/    Executive_Presentation.pptx, Executive_Presentation_Video.mp4, Recording/
  References/      references.md
  README.md
```

## How to run

```
python -m venv venv
venv\Scripts\activate                    # Windows
pip install -r requirements-notebook.txt
jupyter notebook Notebook/Week5_MachineLearning.ipynb      # Run All (about 8 minutes)

pip install -r App/requirements.txt
cd App
streamlit run app.py
```

The app loads `Models/Best_Model.pkl` (or `App/Best_Model.pkl`). Run the notebook first if you want to regenerate it.

## Key engineering decisions

* **Leakage:** `Delivery_Performance` duplicates `Shipments.DelayDays`, and its Status equals "DelayDays >= 3". It defines the target only and is never a feature.
* **Pipelines:** imputation, scaling and encoding are inside scikit-learn Pipelines, so nothing is learned from validation or test data.
* **Splits:** 70/15/15 stratified for classification; time-based for demand forecasting; random customer split for customer value.
* **Selection:** the one-standard-error rule picks the simplest model among statistically tied candidates.
* **History features** use earlier orders only.

## Known limitations

* Eight months of data; September 2026 is partial and excluded from monthly views.
* No timestamps within a day, carrier, route, supplier link or stock movements.
* About 9% of cancelled orders carry returns, and 236 orders have no line items (kept and flagged).
* The app is a workflow demonstration. Its predictions should not drive operational decisions, and it says so on screen.
