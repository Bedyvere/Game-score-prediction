import pandas as pd
import streamlit as st


TRAIN_STATE_KEY = "train_state"
TRAINED_MODELS_KEY = "trained_models"
DEPLOYMENT_KEY = "deployment_summary"
COMPARISON_KEY = "latest_model_comparison"


st.markdown("# Practical Applications of Machine Learning (PAML)")
st.markdown("### Game Score Prediction and Critics Analysis")
st.title("Critical Analysis")
st.caption(
    "This page is written to support report section 4.1.5 on model deployment and the Streamlit front-end application."
)


deployment = st.session_state.get(DEPLOYMENT_KEY)
comparison_df = st.session_state.get(COMPARISON_KEY)
train_state = st.session_state.get(TRAIN_STATE_KEY, {})
trained_models = st.session_state.get(TRAINED_MODELS_KEY, {})


def deployed_model_paragraph():
    if deployment is None:
        return (
            "The application is designed to deploy the regression model that performs best on the held-out "
            "validation set. Because the user-facing task is continuous score prediction, validation RMSE is "
            "treated as the primary criterion, while validation MAE and validation R2 are used as secondary "
            "checks for interpretability and explanatory power. The Naive Bayes classifier is retained as a "
            "supplementary analytical model rather than the deployed model because it predicts only a high/low label."
        )

    metrics = deployment["validation_metrics"]
    return (
        f"In the current Streamlit session, the deployed model is **{deployment['model_name']}**. "
        f"It was selected after comparing candidate regression models on the held-out validation set, "
        f"where it achieved validation RMSE = {metrics['rmse']:.3f}, validation MAE = {metrics['mae']:.3f}, "
        f"and validation R2 = {metrics['r2']:.3f}. This choice is justified because the application predicts "
        f"a continuous review score, so minimizing validation RMSE is the most important requirement. "
        f"Validation MAE was used to confirm that the average absolute prediction error remained small and easy to interpret, "
        f"while validation R2 was used to confirm that the model still explained a meaningful share of score variation."
    )


st.markdown("## 4.1.5.1 Web-Based Application Description")
st.write(
    "We developed a multi-page web application in Streamlit to support the full machine learning workflow for "
    "video game review-score prediction and critics analysis. The app allows a user to load the cleaned dataset, "
    "perform exploratory analysis, select modeling features, train multiple candidate algorithms, evaluate them on "
    "a validation split, deploy the best-performing regression model, and interact with that deployed model through "
    "a prediction form."
)

st.markdown("## 4.1.5.2 Model Selection and Deployment Justification")
st.write(deployed_model_paragraph())

if isinstance(comparison_df, pd.DataFrame) and not comparison_df.empty:
    st.dataframe(
        comparison_df[
            [
                "Model",
                "Validation RMSE",
                "Validation MAE",
                "Validation R2",
                "Overall Rank",
            ]
        ],
        use_container_width=True,
    )
else:
    st.info(
        "Run the evaluation step on the Test Model page to populate the live comparison table that supports the deployment argument."
    )

st.markdown("## 4.1.5.3 Target Population and Benefits")
st.write(
    "The target population includes students analyzing media datasets, instructors reviewing an end-to-end machine "
    "learning project, and gaming researchers or critics who want a compact interface for studying how content and "
    "representation variables relate to review outcomes. The application is beneficial because it lowers the technical "
    "barrier for these users: they can inspect data, compare model behavior, and generate score predictions without "
    "writing additional code or manually reconnecting preprocessing and modeling steps."
)

st.markdown("## 4.1.5.4 UI Layout, Inputs, and Outputs")
st.write(
    "The user interface is organized as a sequential workflow. The EDA page contains dataset import controls, feature-cleaning "
    "tools, visualization menus, and summary tables. The Train Model page contains select boxes and multiselect widgets for the "
    "target and input features, sliders for the validation split, and model-specific hyperparameter controls with training buttons. "
    "The Test Model page contains evaluation controls, comparison tables, learning-curve plots, a deployment button, and a prediction "
    "form made of numeric inputs. The main outputs are charts, descriptive tables, validation metrics, model rankings, and the final "
    "predicted review score returned by the deployed model."
)

st.markdown("## 4.1.5.5 Website Layout Figure")
st.graphviz_chart(
    """
    digraph app_layout {
        rankdir=LR;
        node [shape=box, style="rounded,filled", color="#2f4858", fillcolor="#e8f1f5"];
        eda [label="I. EDA\\nUpload data\\nClean features\\nExplore charts"];
        train [label="II. Train Model\\nChoose target/features\\nTune hyperparameters\\nTrain candidates"];
        test [label="III. Test Model\\nCompare validation metrics\\nDeploy best model\\nPredict score"];
        analysis [label="IV. Critical Analysis\\nExplain deployment\\nDocument UI and workflow"];
        eda -> train -> test -> analysis;
    }
    """
)
st.caption(
    "Figure 1. High-level website layout. The app is arranged as a four-page workflow that moves from data preparation to deployment explanation."
)

st.markdown("## 4.1.5.6 Front-End and Back-End Connection Figure")
st.graphviz_chart(
    """
    digraph deployment_flow {
        rankdir=LR;
        node [shape=box, style="rounded,filled", color="#5c3d2e", fillcolor="#f8ede3"];
        user [label="User inputs\\nmenus, sliders, buttons"];
        state [label="Streamlit session state\\ndataset, split, trained models"];
        model [label="Back-end model object\\npredict() on selected features"];
        output [label="Front-end outputs\\nmetrics, tables, prediction result"];
        user -> state [label="submit selections"];
        state -> model [label="load deployed model"];
        model -> output [label="return prediction"];
        output -> user [label="display result"];
    }
    """
)
st.caption(
    "Figure 2. Front-end/back-end connection. Pressing the deploy button stores the selected trained model in session state, and pressing the prediction button sends widget values into that model's `predict()` method."
)

st.markdown("## Critical Reflection")
st.write(
    "The most important contribution of this project is its social impact rather than prediction alone. "
    "Media review scores affect which games gain attention, credibility, and commercial success, so it matters "
    "whether those scores reflect gameplay quality only or are also shaped by representation-related factors such "
    "as female visibility and sexualization. By combining machine learning with character-level and game-level "
    "features, this application creates a practical framework for examining whether seemingly objective critic scores "
    "may encode broader cultural preferences or bias. That makes the project useful not only for technical analysis, "
    "but also for encouraging fairness, transparency, and more reflective review practices in the game industry."
)
st.write(
    "At the same time, model performance still matters because social conclusions should not be overstated if the "
    "predictive signal is weak. The validation metrics therefore need to be interpreted alongside the broader social "
    "question: strong performance would suggest that these features carry meaningful information about review outcomes, "
    "while weaker performance would indicate that the observed relationships are limited, noisy, or insufficient on "
    "their own. A further limitation is that the current dataset is relatively small and the deployment is session-based, "
    "so future work should expand the sample, test more robust models, and persist the deployed pipeline beyond a single "
    "Streamlit session."
)

if deployment is not None:
    st.markdown("## Current Deployment Snapshot")
    st.json(
        {
            "model_name": deployment["model_name"],
            "target": deployment["target"],
            "features": deployment["features"],
            "validation_metrics": deployment["validation_metrics"],
            "selection_rule": deployment["selection_rule"],
        }
    )
elif trained_models:
    st.info(
        "Models have been trained, but a deployed model has not been chosen yet. Use the Test Model page to compare candidates and press the deployment button."
    )
