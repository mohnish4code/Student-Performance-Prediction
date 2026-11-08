# Student Performance Prediction

My first end-to-end machine learning project. It predicts a student's **maths score** from their background
and their reading and writing scores, and serves the model through a small Flask web form.

## How it works

```
notebook/data/stud.csv → data ingestion → train/test split → preprocessing → model selection → saved model → Flask form
```

- **Data ingestion** (`src/components/data_ingestion.py`) reads the dataset and writes train and test splits to `artifacts/`.
- **Transformation** (`src/components/data_transformation.py`) imputes, one-hot encodes the categorical columns and
  scales the numeric ones, then saves the fitted preprocessor as `artifacts/preprocessor.pkl`.
- **Model selection** (`src/components/model_trainer.py`) tunes and compares seven regressors (Linear Regression,
  Decision Tree, Random Forest, Gradient Boosting, AdaBoost, XGBoost and CatBoost) and keeps the one with the best
  test R², rejecting all of them if none reaches 0.6. The winner is saved as `artifacts/model.pkl`.
- **Prediction** (`src/pipeline/predict_pipeline.py` + `app.py`) loads the preprocessor and model and predicts from
  the web form.

**Inputs:** gender, race/ethnicity, parental level of education, lunch type, test preparation course, reading score
and writing score. **Target:** maths score.

EDA and the first round of model training are in `notebook/`.

## Run it

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000/predictdata.

To retrain from scratch, run the ingestion component, which runs transformation and training after it:

```bash
python -m src.components.data_ingestion
```

## Tech stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, CatBoost, Flask.
