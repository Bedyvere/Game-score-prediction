# Game Score Prediction

Game Score Prediction and Critics Analysis is a Streamlit application for exploring how video game characteristics relate to professional review scores. The project combines exploratory data analysis, machine learning model training, validation-based model selection, and an interactive front end for score prediction.

## Project Goal

Media review scores affect a game's visibility, reputation, and commercial success. This project studies whether review outcomes are associated only with general game attributes or also with representation-related features such as female character presence and sexualization. The app is designed to support both prediction and critical analysis.

## Streamlit App Overview

The Streamlit program is the main way to use this project. It is organized as a four-page workflow:

1. `I. EDA`
   Inspect the dataset, clean columns, handle missing values, transform features, and create charts connected to the project question.
2. `II. Train Model`
   Select a numeric target, choose input features, define a train/validation split, and train candidate machine learning models.
3. `III. Test Model`
   Compare trained models using validation metrics, review learning curves, select the best regression model, and deploy it for prediction.
4. `IV. Critical Analysis`
   Summarize deployment choices, describe the UI and model-selection logic, and reflect on the broader social impact of the project.

## What the App Can Do

- Upload or reuse the processed game dataset inside Streamlit.
- Explore the data with tables, summary metrics, and interactive visualizations.
- Apply lightweight preprocessing such as feature removal, encoding, scaling, feature creation, and outlier handling.
- Train multiple models in one session, including:
  - Multiple Linear Regression
  - Polynomial Regression
  - Ridge Regression
  - Lasso Regression
  - Naive Bayes
- Evaluate candidate models with held-out validation metrics such as RMSE, MAE, and R2.
- Deploy the selected regression model inside the Streamlit session.
- Enter feature values in a prediction form and generate a predicted review score.

## Why Streamlit Fits This Project

Streamlit makes the project easier to understand and demonstrate because the front end and back end are connected in one lightweight application. A user can move from data exploration to model training and then to deployment without leaving the browser or writing additional code. This is especially useful for classroom presentation, model comparison, and explaining how prediction results are generated.

## Data and Features

The repository includes merged game and character data in the `datasets/` folder. The modeling workflow focuses on numeric review variables and engineered representation-related features, including measures such as average review score, critic-specific scores, non-male character percentage, and sexualization-related indicators.

## How Model Deployment Works

Candidate models are trained on the `Train Model` page and stored in Streamlit session state. On the `Test Model` page, the trained regression models are compared on the held-out validation set. The deployed model is chosen based on validation performance, with validation RMSE treated as the primary deployment metric for continuous score prediction and MAE/R2 used as supporting checks. After deployment, the same model powers the prediction form shown in the app.

## Run the App Locally

1. Open a terminal in the project root.
2. Install the required Python packages if needed.
3. Start Streamlit:

```bash
py -m streamlit run main.py
```

4. Open the local URL shown in the terminal.
5. Start on the `EDA` page and move through the pages in order.

## Repository Structure

```text
main.py
pages/
  I. EDA.py
  II. Train Model.py
  III. Test Model.py
  IV. Critical Analysis.py
datasets/
images/
README.md
```

## Target Users

This application is intended for:

- Students building an end-to-end machine learning project
- Instructors evaluating Streamlit-based ML workflows
- Researchers studying media review patterns
- Game developers and critics interested in how representation-related features may relate to scoring outcomes

## Limitations

- The current deployment is session-based inside Streamlit rather than persisted as a standalone production service.
- The dataset is relatively small, so model results should be interpreted carefully.
- The app is best suited for project demonstration, experimentation, and analysis rather than large-scale production inference.

## Authors

- Samyuktha Lokanandi (`sl3539`)
- Maggie Liang (`ml2927`)
- Xueer Zhang (`xz946`)
- Yujun Che (`yc2989`)
- Vivian Xie (`xx374`)
- Rui Chen (`rc985`)
