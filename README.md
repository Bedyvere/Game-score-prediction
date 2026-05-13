# Game Score Prediction and Critics Analysis

A Streamlit application for exploring how video game characteristics relate to professional review scores. Users can upload data, choose models, tune regularization, compare metrics, and inspect feature coefficients. The app supports an end-to-end ML workflow: EDA → preprocessing → training → testing → interpretation.

## Project Goal

Critic review scores shape a game's commercial success and public perception. Our central question was whether review scores are driven purely by gameplay quality, or whether they may also be associated with representation-related features such as female character presence and sexualization. This motivated us to explore whether representation-related variables contribute predictive information alongside traditional features such as genre and platform. Prior work relies on descriptive statistics rather than ML methodology.

## Streamlit App Overview

The Streamlit application is organized as a four-page workflow:

1. **I. EDA** — Inspect the dataset, clean columns, handle missing values, transform features, and create charts connected to the project question.
2. **II. Train Model** — Select a numeric target, choose input features, define a train/validation split, and train candidate regression models.
3. **III. Test Model** — Compare trained models using validation metrics, review learning curves, select the best regression model, and deploy it for prediction.
4. **IV. Critical Analysis** — Summarize deployment choices, describe the UI and model-selection logic, and reflect on the broader social impact of the project.

## What the App Can Do

- Upload or reuse the processed game dataset inside Streamlit.
- Explore the data with tables, summary metrics, and interactive visualizations.
- Apply lightweight preprocessing such as feature removal, encoding, scaling, feature creation, and outlier handling.
- Train three regression models, all implemented **from scratch using NumPy and Pandas only** (no sklearn model classes):
  - Multiple Linear Regression
  - Ridge Regression (L2 regularization)
  - Lasso Regression (L1 regularization)
- Fine-tune hyperparameters such as learning rate, number of gradient descent iterations, and regularization strength (λ).
- Evaluate candidate models with held-out validation metrics: RMSE, MAE, and R².
- Deploy the selected regression model inside the Streamlit session.
- Enter feature values in a prediction form and generate a predicted review score.

## Data and Features

The repository includes merged game and character data in the `datasets/` folder. The modeling workflow uses non-leaking features only — `Metacritic`, `IGN`, `GameSpot`, and `Destructoid` are **excluded** as inputs because `Avg_Reviews` is defined as their average. Using them as features would cause target leakage.

The default modeling features are:

- `Percentage_non_male_num` — share of non-male characters (0–1)
- `Sexualization` — average sexualization score
- `Protagonist_Non_Male` — number of non-male protagonists
- `Team_percentage_num` — share of female team members (0–1)
- `Release_Year`
- `PEGI` — age rating
- One-hot encoded: `Genre`, `Platform`, `Country` (where present)

## How Model Deployment Works

Candidate models are trained on the **Train Model** page and stored in Streamlit session state. On the **Test Model** page, trained regression models are compared on the held-out validation set. The deployed model is chosen based on validation performance, with validation RMSE as the primary criterion and MAE/R² as supporting checks. After deployment, the same model object powers the prediction form shown in the app.

## Run the App

**Option 1 — Deployed website (recommended):**

[https://game-score-prediction-paml.streamlit.app/](https://game-score-prediction-paml.streamlit.app/)

**Option 2 — Run locally:**

1. Open a terminal in the project root.
2. Install required packages:

```
pip install -r requirements.txt
```

3. Start Streamlit:

```
python -m streamlit run main.py
```

## Jupyter Notebook

The notebook at `notebooks/Game_Score_Prediction_Summary.ipynb` reproduces the full analysis outside the app. It covers EDA, from-scratch model training, validation metrics, coefficient analysis, and learning curves. All three regression models in the notebook are also implemented from scratch using NumPy — no sklearn model classes are used for training.

To run it:

```
cd notebooks
jupyter notebook Game_Score_Prediction_Summary.ipynb
```

## Repository Structure

```
main.py
notebooks/
  Game_Score_Prediction_Summary.ipynb
pages/
  I. EDA.py
  II. Train Model.py
  III. Test Model.py
  IV. Critical Analysis.py
datasets/
  merged_grivg_data.csv
images/
README.md
```

## Authors

- Samyuktha Lokanandi (`sl3539@cornell.edu`)
- Maggie Liang (`ml2927@cornell.edu`)
- Xueer Zhang (`xz946@cornell.edu`)
- Yujun Che (`yc2989@cornell.edu`)
- Vivian Xie (`xx374@cornell.edu`)
- Rui Chen (`rc985@cornell.edu`)
