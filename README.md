# Game Score Prediction

Game Score Prediction and Critics Analysis is a Streamlit application for exploring how video game characteristics relate to professional review scores. 
Users can upload data, choose models, tune regularization, compare metrics, and inspect feature coefficients.
The app supports an end-to-end ML workflow: EDA → preprocessing → training → testing → interpretation.

## Project Goal

Critic review scores shape a game's commercial success and public perception. Our central question was whether review scores are driven purely by gameplay quality, or whether they may also be associated with representation-related features such as female character presence and sexualization. This motivate us to explored whether representation-related variables contribute predictive information alongside traditional features such as genre and platform. Prior work relies on descriptive stats, while not delve into ML methology. 

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
  3 regression model, 
  - Multiple Linear Regression
  - Ridge Regression
  - Lasso Regression
  and a calcification model, 
  - Naive Bayes
and finetune with adjusting learning rate and more! 
- Evaluate candidate models with held-out validation metrics such as RMSE, MAE, and R2.
- Deploy the selected regression model inside the Streamlit session.
- Enter feature values in a prediction form and generate a predicted review score.


## Data and Features

The repository includes merged game and character data in the `datasets/` folder. The modeling workflow focuses on numeric review variables and engineered representation-related features, including measures such as average review score, critic-specific scores, non-male character percentage, and sexualization-related indicators.

## How Model Deployment Works

Candidate models are trained on the `Train Model` page and stored in Streamlit session state. On the `Test Model` page, the trained regression models are compared on the held-out validation set. The deployed model is chosen based on validation performance, with validation RMSE treated as the primary deployment metric for continuous score prediction and MAE/R2 used as supporting checks. After deployment, the same model powers the prediction form shown in the app.

## Run the App 

1. Open a terminal in the project root.
2. Install the required Python packages if needed.
3. Start Streamlit:

```bash
py -m streamlit run main.py
```

or, run it through deployed website (recommended):

https://game-score-prediction-paml.streamlit.app/




## Repository Structure

```text
main.py
notebooks/
  Game_Score_Prediction_Summary.ipynb
pages/
  I. EDA.py
  II. Train Model.py
  III. Test Model.py
  IV. Critical Analysis.py
datasets/
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
