import copy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split


SESSION_DATA_KEY = "game_df"
LEGACY_SESSION_DATA_KEY = "house_df"
TRAIN_STATE_KEY = "train_state"
TRAINED_MODELS_KEY = "trained_models"
DEPLOYMENT_KEY = "deployment_summary"
COMPARISON_KEY = "latest_model_comparison"

REGRESSION_MODEL_NAMES = [
    "Multiple Linear Regression",
    "Polynomial Regression",
    "Ridge Regression",
    "Lasso Regression",
]


st.markdown("# Practical Applications of Machine Learning (PAML)")
st.markdown("### Game Score Prediction and Critics Analysis")
st.title("Test Model")
st.caption(
    "This page evaluates trained candidates on the held-out validation set, "
    "selects the best deployment model, and exposes a simple prediction interface."
)


def split_dataset(X, y, number, random_state=45):
    return train_test_split(
        X,
        y,
        test_size=number / 100,
        random_state=random_state,
    )


def flatten(values):
    return np.asarray(values, dtype=float).reshape(-1)


def rmse(y_true, y_pred):
    y_true = flatten(y_true)
    y_pred = flatten(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true = flatten(y_true)
    y_pred = flatten(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred):
    y_true = flatten(y_true)
    y_pred = flatten(y_pred)
    total_sum_squares = np.sum((y_true - np.mean(y_true)) ** 2)
    if total_sum_squares == 0:
        return 0.0
    residual_sum_squares = np.sum((y_true - y_pred) ** 2)
    return float(1 - (residual_sum_squares / total_sum_squares))


METRICS_MAP = {
    "mean_absolute_error": mae,
    "root_mean_squared_error": rmse,
    "r2_score": r2,
}


def load_dataset(source):
    df = pd.read_csv(source)
    st.session_state[SESSION_DATA_KEY] = df
    st.session_state[LEGACY_SESSION_DATA_KEY] = df
    return df


def build_train_state_from_dataset(df):
    numeric_columns = list(df.select_dtypes(include="number").columns)
    if len(numeric_columns) < 2:
        return None

    default_target = "Avg_Reviews" if "Avg_Reviews" in numeric_columns else numeric_columns[0]
    target = st.selectbox(
        "Select the numeric target to predict",
        options=numeric_columns,
        index=numeric_columns.index(default_target),
        key="fallback_target",
    )
    features = st.multiselect(
        "Select numeric features",
        options=[column for column in numeric_columns if column != target],
        default=[column for column in numeric_columns if column != target][:4],
        key="fallback_features",
    )
    split_pct = st.slider(
        "Validation split (%)",
        min_value=10,
        max_value=40,
        value=30,
        step=5,
        key="fallback_split",
    )

    if not features:
        return None

    modeling_df = df[features + [target]].dropna().copy()
    if len(modeling_df) < 8:
        return None

    X = modeling_df[features].to_numpy(dtype=float)
    y = modeling_df[target].to_numpy(dtype=float).reshape(-1, 1)
    X_train, X_val, y_train, y_val = split_dataset(X, y, split_pct)

    train_state = {
        "target": target,
        "features": features,
        "split_pct": split_pct,
        "random_state": 45,
        "rows_used": int(len(modeling_df)),
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "feature_defaults": modeling_df[features].median(numeric_only=True).to_dict(),
    }
    st.session_state[TRAIN_STATE_KEY] = train_state
    return train_state


def get_dataset_and_state():
    df = None
    if SESSION_DATA_KEY in st.session_state:
        df = st.session_state[SESSION_DATA_KEY]
    elif LEGACY_SESSION_DATA_KEY in st.session_state:
        df = st.session_state[LEGACY_SESSION_DATA_KEY]
    else:
        uploaded = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
        if uploaded is not None:
            df = load_dataset(uploaded)

    train_state = st.session_state.get(TRAIN_STATE_KEY)
    if df is not None and train_state is None:
        st.info("No stored training split was found, so this page can rebuild one from the current dataset.")
        train_state = build_train_state_from_dataset(df)
    return df, train_state


def compute_eval_metrics(X, y_true, model, metrics):
    y_pred = model.predict(X)
    metric_dict = {}
    for metric in metrics:
        metric_dict[metric] = METRICS_MAP[metric](y_true, y_pred)
    return metric_dict


def plot_learning_curve(X_train, X_val, y_train, y_val, trained_model, metrics, model_name):
    if len(X_train) < 10:
        st.info(f"{model_name} needs a slightly larger training split before plotting a learning curve.")
        return None

    step_count = min(6, len(X_train))
    sample_sizes = np.unique(np.linspace(max(4, len(X_train) // 4), len(X_train), step_count).astype(int))
    fig = go.Figure()

    for metric in metrics:
        train_scores = []
        val_scores = []
        metric_fn = METRICS_MAP[metric]
        for sample_size in sample_sizes:
            model_copy = copy.deepcopy(trained_model)
            model_copy.fit(X_train[:sample_size], y_train[:sample_size])
            train_scores.append(metric_fn(y_train[:sample_size], model_copy.predict(X_train[:sample_size])))
            val_scores.append(metric_fn(y_val, model_copy.predict(X_val)))

        fig.add_trace(
            go.Scatter(
                x=sample_sizes,
                y=train_scores,
                mode="lines+markers",
                name=f"{metric} train",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sample_sizes,
                y=val_scores,
                mode="lines+markers",
                name=f"{metric} validation",
                line=dict(dash="dash"),
            )
        )

    fig.update_layout(
        title=f"{model_name} Learning Curve",
        xaxis_title="Training Set Size",
        yaxis_title="Metric Value",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_comparison_table(selected_models, trained_models, train_state):
    rows = []
    X_train = train_state["X_train"]
    X_val = train_state["X_val"]
    y_train = train_state["y_train"]
    y_val = train_state["y_val"]

    ranking_metrics = [
        "root_mean_squared_error",
        "mean_absolute_error",
        "r2_score",
    ]

    for model_name in selected_models:
        model_info = trained_models[model_name]
        model = model_info["model"]
        train_metrics = compute_eval_metrics(X_train, y_train, model, ranking_metrics)
        val_metrics = compute_eval_metrics(X_val, y_val, model, ranking_metrics)
        rows.append(
            {
                "Model": model_name,
                "Target": model_info["target"],
                "Features": ", ".join(model_info["features"]),
                "Train RMSE": train_metrics["root_mean_squared_error"],
                "Validation RMSE": val_metrics["root_mean_squared_error"],
                "Train MAE": train_metrics["mean_absolute_error"],
                "Validation MAE": val_metrics["mean_absolute_error"],
                "Train R2": train_metrics["r2_score"],
                "Validation R2": val_metrics["r2_score"],
            }
        )

    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        return comparison_df

    comparison_df["RMSE Rank"] = comparison_df["Validation RMSE"].rank(method="dense", ascending=True)
    comparison_df["MAE Rank"] = comparison_df["Validation MAE"].rank(method="dense", ascending=True)
    comparison_df["R2 Rank"] = comparison_df["Validation R2"].rank(method="dense", ascending=False)
    comparison_df["Overall Rank"] = (
        comparison_df["RMSE Rank"] + comparison_df["MAE Rank"] + comparison_df["R2 Rank"]
    )
    comparison_df = comparison_df.sort_values(
        by=["Overall Rank", "Validation RMSE", "Validation MAE", "Validation R2"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    return comparison_df


def build_recommendation_text(best_row):
    return (
        f"{best_row['Model']} is recommended for deployment because it achieved the lowest "
        f"overall validation rank. Its validation RMSE is {best_row['Validation RMSE']:.3f}, "
        f"its validation MAE is {best_row['Validation MAE']:.3f}, and its validation R2 is "
        f"{best_row['Validation R2']:.3f}. RMSE is prioritized because the deployed app predicts "
        f"a continuous review score, while MAE and R2 are used as secondary checks for "
        f"interpretability and explanatory power."
    )


def deploy_model(model_name, comparison_df, trained_models, train_state):
    deployed_row = comparison_df.loc[comparison_df["Model"] == model_name].iloc[0]
    model_info = trained_models[model_name]
    st.session_state[DEPLOYMENT_KEY] = {
        "model_name": model_name,
        "task_type": model_info["task_type"],
        "target": model_info["target"],
        "features": model_info["features"],
        "hyperparameters": model_info["hyperparameters"],
        "split_pct": model_info.get("split_pct"),
        "rows_used": model_info.get("rows_used"),
        "feature_defaults": train_state.get("feature_defaults", {}),
        "train_metrics": {
            "rmse": float(deployed_row["Train RMSE"]),
            "mae": float(deployed_row["Train MAE"]),
            "r2": float(deployed_row["Train R2"]),
        },
        "validation_metrics": {
            "rmse": float(deployed_row["Validation RMSE"]),
            "mae": float(deployed_row["Validation MAE"]),
            "r2": float(deployed_row["Validation R2"]),
        },
        "selection_rule": (
            "Lowest validation RMSE was prioritized because the application predicts a "
            "continuous score; validation MAE and validation R2 were used as secondary checks."
        ),
        "justification": build_recommendation_text(deployed_row),
    }
    st.session_state["deployed_model_name"] = model_name


def get_deployed_model():
    deployment = st.session_state.get(DEPLOYMENT_KEY)
    trained_models = st.session_state.get(TRAINED_MODELS_KEY, {})
    if deployment is None:
        return None
    model_name = deployment["model_name"]
    model_info = trained_models.get(model_name)
    if model_info is None:
        return None
    return model_info["model"]


df, train_state = get_dataset_and_state()

if df is not None and train_state is not None:
    trained_models = st.session_state.get(TRAINED_MODELS_KEY, {})
    regression_models = {
        name: info
        for name, info in trained_models.items()
        if info["task_type"] == "regression" and name in REGRESSION_MODEL_NAMES
    }

    st.markdown("## Validation Setup")
    setup_col1, setup_col2, setup_col3 = st.columns(3)
    setup_col1.metric("Training rows", len(train_state["X_train"]))
    setup_col2.metric("Validation rows", len(train_state["X_val"]))
    setup_col3.metric("Target", train_state["target"])
    st.caption(
        f"Current feature set: {', '.join(train_state['features'])}. "
        "These values come from the training page and are reused here for fair model comparison."
    )

    if not regression_models:
        st.warning("No trained regression models were found. Train at least one candidate on the Train Model page first.")
        st.stop()

    metric_select = st.multiselect(
        "Select metrics to highlight in the plots and tables",
        options=list(METRICS_MAP.keys()),
        default=list(METRICS_MAP.keys()),
    )

    selected_models = st.multiselect(
        "Select trained regression models for evaluation",
        options=list(regression_models.keys()),
        default=list(regression_models.keys()),
    )

    review_plot = st.multiselect(
        "Select outputs to generate",
        options=["Metric Results", "Learning Curve"],
        default=["Metric Results"],
    )

    if selected_models and review_plot:
        if st.button("Evaluate Selected Models"):
            comparison_df = build_comparison_table(selected_models, regression_models, train_state)
            st.session_state[COMPARISON_KEY] = comparison_df

    comparison_df = st.session_state.get(COMPARISON_KEY)
    if isinstance(comparison_df, pd.DataFrame) and not comparison_df.empty:
        st.markdown("## Model Comparison")
        display_columns = ["Model"]
        if "Metric Results" in review_plot:
            metric_column_map = {
                "root_mean_squared_error": ["Train RMSE", "Validation RMSE"],
                "mean_absolute_error": ["Train MAE", "Validation MAE"],
                "r2_score": ["Train R2", "Validation R2"],
            }
            for metric in metric_select:
                display_columns.extend(metric_column_map[metric])
            display_columns.extend(["Overall Rank"])
            st.dataframe(comparison_df[display_columns], use_container_width=True)

        best_row = comparison_df.iloc[0]
        st.success(build_recommendation_text(best_row))

        if "Learning Curve" in review_plot:
            st.markdown("## Learning Curves")
            for model_name in selected_models:
                if model_name not in regression_models:
                    continue
                figure = plot_learning_curve(
                    train_state["X_train"],
                    train_state["X_val"],
                    train_state["y_train"],
                    train_state["y_val"],
                    regression_models[model_name]["model"],
                    metric_select,
                    model_name,
                )
                if figure is not None:
                    st.plotly_chart(figure, use_container_width=True)

        st.markdown("## Deploy a Model")
        deployment_choice = st.selectbox(
            "Select the regression model to deploy",
            options=comparison_df["Model"].tolist(),
            index=0,
        )
        st.caption(
            "Pressing the deploy button stores the chosen trained model and its validation summary in session state. "
            "The prediction interface below and the Critical Analysis page both read from that deployed state."
        )

        if st.button("Deploy Selected Model"):
            deploy_model(deployment_choice, comparison_df, regression_models, train_state)
            st.success(f"{deployment_choice} is now the deployed model for this Streamlit session.")

    deployment = st.session_state.get(DEPLOYMENT_KEY)
    deployed_model = get_deployed_model()
    if deployment is not None and deployed_model is not None:
        st.markdown("## Deployed Model Card")
        st.write(f"**Model:** {deployment['model_name']}")
        st.write(f"**Target:** {deployment['target']}")
        st.write(f"**Features:** {', '.join(deployment['features'])}")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Validation RMSE", f"{deployment['validation_metrics']['rmse']:.3f}")
        metric_col2.metric("Validation MAE", f"{deployment['validation_metrics']['mae']:.3f}")
        metric_col3.metric("Validation R2", f"{deployment['validation_metrics']['r2']:.3f}")
        st.info(deployment["selection_rule"])

        st.markdown("## Prediction Playground")
        st.caption(
            "This front-end form is connected directly to the deployed back-end model. "
            "When the user presses the prediction button, the numeric inputs are assembled into a feature vector "
            "and passed to the selected model's `predict()` method."
        )

        feature_defaults = deployment.get("feature_defaults", {})
        with st.form("deployed_prediction_form"):
            user_inputs = []
            for feature in deployment["features"]:
                default_value = float(feature_defaults.get(feature, 0.0))
                user_inputs.append(
                    st.number_input(
                        feature,
                        value=default_value,
                        format="%.4f",
                        key=f"deployed_{feature}",
                    )
                )
            predict_button = st.form_submit_button("Predict Score")

        if predict_button:
            feature_vector = np.array([user_inputs], dtype=float)
            predicted_score = float(np.asarray(deployed_model.predict(feature_vector)).reshape(-1)[0])
            st.metric(f"Predicted {deployment['target']}", f"{predicted_score:.3f}")
            st.write(
                "The prediction above was generated by the deployed model stored in `st.session_state`, "
                "which is the same object selected from the validation comparison table."
            )
