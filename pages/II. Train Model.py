import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures


SESSION_DATA_KEY = "game_df"
LEGACY_SESSION_DATA_KEY = "house_df"
TRAIN_STATE_KEY = "train_state"
TRAINED_MODELS_KEY = "trained_models"

REGRESSION_MODEL_NAMES = [
    "Multiple Linear Regression",
    "Polynomial Regression",
    "Ridge Regression",
    "Lasso Regression",
]
CLASSIFICATION_MODEL_NAMES = ["Naive Bayes"]


st.markdown("# Practical Applications of Machine Learning (PAML)")
st.markdown("### Game Score Prediction and Critics Analysis")
st.title("Train Model")
st.caption(
    "This page prepares the modeling dataset, trains candidate models, and stores "
    "them in Streamlit session state for evaluation and deployment on the next page."
)


def split_dataset(X, y, number, random_state=45):
    """Split features and targets into train and validation sets."""
    return train_test_split(
        X,
        y,
        test_size=number / 100,
        random_state=random_state,
    )


def to_column_vector(values):
    """Convert input into a 2D column vector."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


class LinearRegression:
    def __init__(self, learning_rate=0.001, num_iterations=500):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.cost_history = []
        self.feature_mean_ = None
        self.feature_std_ = None
        self.W = None

    def _fit_scaler(self, X):
        self.feature_mean_ = X.mean(axis=0)
        self.feature_std_ = X.std(axis=0)
        self.feature_std_[self.feature_std_ == 0] = 1.0

    def normalize(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.feature_mean_) / self.feature_std_

    def predict(self, X):
        X_norm = self.normalize(X)
        return X_norm @ self.W[1:] + self.W[0]

    def update_weights(self):
        X_norm = self.normalize(self.X)
        y_pred = self.predict(self.X)
        error = y_pred - self.Y
        m = len(X_norm)

        dW0 = (2 / m) * np.sum(error)
        dW_rest = (2 / m) * (X_norm.T @ error)
        gradient = np.vstack(([[dW0]], dW_rest))

        self.W = self.W - self.learning_rate * gradient
        self.cost_history.append(float(np.mean((self.Y - y_pred) ** 2)))
        return self

    def fit(self, X, Y):
        self.X = np.asarray(X, dtype=float)
        self.Y = to_column_vector(Y)
        self._fit_scaler(self.X)
        self.W = np.zeros((self.X.shape[1] + 1, 1))
        self.cost_history = []

        for _ in range(self.num_iterations):
            self.update_weights()
        return self

    def get_weights(self, model_name, features):
        weights = pd.DataFrame(
            {
                "term": ["bias"] + list(features),
                "weight": self.W.reshape(-1),
            }
        )
        st.write(model_name)
        st.dataframe(weights, use_container_width=True)
        return {model_name: weights}


class PolynomialRegression(LinearRegression):
    def __init__(self, degree, learning_rate, num_iterations):
        super().__init__(learning_rate=learning_rate, num_iterations=num_iterations)
        self.degree = degree
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=True)
        self.poly_mean_ = None
        self.poly_std_ = None

    def _transform(self, X, fit=False):
        X = np.asarray(X, dtype=float)
        transformed = self.poly.fit_transform(X) if fit else self.poly.transform(X)
        transformed = transformed.astype(float)

        if fit:
            self.poly_mean_ = transformed[:, 1:].mean(axis=0)
            self.poly_std_ = transformed[:, 1:].std(axis=0)
            self.poly_std_[self.poly_std_ == 0] = 1.0

        if transformed.shape[1] > 1:
            transformed[:, 1:] = (transformed[:, 1:] - self.poly_mean_) / self.poly_std_
        return transformed

    def fit(self, X, Y):
        self.X = self._transform(X, fit=True)
        self.Y = to_column_vector(Y)
        self.W = np.zeros((self.X.shape[1], 1))
        self.cost_history = []

        for _ in range(self.num_iterations):
            y_pred = self.X @ self.W
            error = y_pred - self.Y
            gradient = (2 / len(self.X)) * (self.X.T @ error)
            self.W = self.W - self.learning_rate * gradient
            self.cost_history.append(float(np.mean((self.Y - y_pred) ** 2)))
        return self

    def predict(self, X):
        transformed = self._transform(X, fit=False)
        return transformed @ self.W

    def get_weights(self, model_name, features):
        feature_names = self.poly.get_feature_names_out(features)
        weights = pd.DataFrame(
            {
                "term": feature_names,
                "weight": self.W.reshape(-1),
            }
        )
        st.write(model_name)
        st.dataframe(weights, use_container_width=True)
        return {model_name: weights}


class RidgeRegression(LinearRegression):
    def __init__(self, learning_rate, num_iterations, l2_penalty):
        super().__init__(learning_rate=learning_rate, num_iterations=num_iterations)
        self.l2_penalty = l2_penalty

    def update_weights(self):
        X_norm = self.normalize(self.X)
        y_pred = self.predict(self.X)
        error = y_pred - self.Y
        m = len(X_norm)

        dW0 = (2 / m) * np.sum(error)
        dW_rest = (2 / m) * (X_norm.T @ error) + (2 * self.l2_penalty / m) * self.W[1:]
        gradient = np.vstack(([[dW0]], dW_rest))

        self.W = self.W - self.learning_rate * gradient
        penalty = self.l2_penalty * np.sum(self.W[1:] ** 2)
        self.cost_history.append(float(np.mean((self.Y - y_pred) ** 2) + penalty))
        return self


class LassoRegression(LinearRegression):
    def __init__(self, learning_rate, num_iterations, l1_penalty):
        super().__init__(learning_rate=learning_rate, num_iterations=num_iterations)
        self.l1_penalty = l1_penalty

    def update_weights(self):
        X_norm = self.normalize(self.X)
        y_pred = self.predict(self.X)
        error = y_pred - self.Y
        m = len(X_norm)

        dW0 = (2 / m) * np.sum(error)
        dW_rest = (2 / m) * (X_norm.T @ error) + (self.l1_penalty / m) * np.sign(self.W[1:])
        gradient = np.vstack(([[dW0]], dW_rest))

        self.W = self.W - self.learning_rate * gradient
        penalty = self.l1_penalty * np.sum(np.abs(self.W[1:]))
        self.cost_history.append(float(np.mean((self.Y - y_pred) ** 2) + penalty))
        return self


class NaiveBayes:
    def __init__(self):
        self.model = GaussianNB()
        self.model_name = "Naive Bayes"
        self.cost_history = []

    def fit(self, X, Y):
        self.model.fit(np.asarray(X, dtype=float), np.asarray(Y).reshape(-1))
        return self

    def predict(self, X):
        return self.model.predict(np.asarray(X, dtype=float))

    def predict_probability(self, X):
        return self.model.predict_proba(np.asarray(X, dtype=float))[:, 1]

    def get_weights(self):
        st.info(
            "Gaussian Naive Bayes does not expose a simple coefficient vector like the "
            "regression models. Use the evaluation page to inspect its predictive behavior."
        )
        return None


def load_dataset(source):
    df = pd.read_csv(source)
    st.session_state[SESSION_DATA_KEY] = df
    st.session_state[LEGACY_SESSION_DATA_KEY] = df
    return df


def get_active_dataset():
    df = None
    if SESSION_DATA_KEY in st.session_state:
        df = st.session_state[SESSION_DATA_KEY]
    elif LEGACY_SESSION_DATA_KEY in st.session_state:
        df = st.session_state[LEGACY_SESSION_DATA_KEY]
    else:
        uploaded = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
        if uploaded is not None:
            df = load_dataset(uploaded)
    return df


def prepare_supervised_data(df, target, features):
    required_columns = list(dict.fromkeys(features + [target]))
    modeling_df = df[required_columns].dropna().copy()
    X = modeling_df[features].to_numpy(dtype=float)
    y = modeling_df[target].to_numpy(dtype=float).reshape(-1, 1)
    return modeling_df, X, y


def store_training_state(modeling_df, X_train, X_val, y_train, y_val, target, features, split_pct, random_state):
    st.session_state["target"] = target
    st.session_state["feature"] = features
    st.session_state["X_train"] = X_train
    st.session_state["X_val"] = X_val
    st.session_state["y_train"] = y_train
    st.session_state["y_val"] = y_val
    st.session_state["X_train_df"] = pd.DataFrame(X_train, columns=features)
    st.session_state["X_val_df"] = pd.DataFrame(X_val, columns=features)
    st.session_state["y_train_df"] = pd.DataFrame(y_train, columns=[target])
    st.session_state["y_val_df"] = pd.DataFrame(y_val, columns=[target])
    st.session_state[TRAIN_STATE_KEY] = {
        "target": target,
        "features": features,
        "split_pct": split_pct,
        "random_state": random_state,
        "rows_used": int(len(modeling_df)),
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "feature_defaults": modeling_df[features].median(numeric_only=True).to_dict(),
    }


def build_model(model_name, hyperparameters):
    if model_name == "Multiple Linear Regression":
        return LinearRegression(
            learning_rate=hyperparameters["learning_rate"],
            num_iterations=hyperparameters["num_iterations"],
        )
    if model_name == "Polynomial Regression":
        return PolynomialRegression(
            degree=hyperparameters["degree"],
            learning_rate=hyperparameters["learning_rate"],
            num_iterations=hyperparameters["num_iterations"],
        )
    if model_name == "Ridge Regression":
        return RidgeRegression(
            learning_rate=hyperparameters["learning_rate"],
            num_iterations=hyperparameters["num_iterations"],
            l2_penalty=hyperparameters["l2_penalty"],
        )
    if model_name == "Lasso Regression":
        return LassoRegression(
            learning_rate=hyperparameters["learning_rate"],
            num_iterations=hyperparameters["num_iterations"],
            l1_penalty=hyperparameters["l1_penalty"],
        )
    if model_name == "Naive Bayes":
        return NaiveBayes()
    raise ValueError(f"Unsupported model: {model_name}")


def register_trained_model(model_name, model, task_type, target, features, hyperparameters):
    trained_models = st.session_state.get(TRAINED_MODELS_KEY, {})
    train_state = st.session_state.get(TRAIN_STATE_KEY, {})
    trained_models[model_name] = {
        "model": model,
        "task_type": task_type,
        "target": target,
        "features": features,
        "hyperparameters": hyperparameters,
        "rows_used": train_state.get("rows_used"),
        "split_pct": train_state.get("split_pct"),
        "random_state": train_state.get("random_state"),
    }
    st.session_state[TRAINED_MODELS_KEY] = trained_models
    st.session_state[model_name] = model


def plot_cost_history(model_name, model):
    if not getattr(model, "cost_history", None):
        st.info(f"{model_name} does not expose an iterative cost history.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, len(model.cost_history) + 1),
            y=model.cost_history,
            mode="lines",
            name=model_name,
        )
    )
    fig.update_layout(
        title=f"{model_name} Cost Curve",
        xaxis_title="Iteration",
        yaxis_title="Cost",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def show_model_registry():
    trained_models = st.session_state.get(TRAINED_MODELS_KEY, {})
    if not trained_models:
        st.info("No models have been trained yet.")
        return

    rows = []
    for model_name, info in trained_models.items():
        rows.append(
            {
                "Model": model_name,
                "Task": info["task_type"],
                "Target": info["target"],
                "Features": ", ".join(info["features"]),
                "Rows used": info.get("rows_used"),
                "Validation split (%)": info.get("split_pct"),
                "Hyperparameters": str(info.get("hyperparameters", {})),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


df = get_active_dataset()

if df is not None:
    numeric_columns = list(df.select_dtypes(include="number").columns)
    st.markdown("## Modeling Dataset")
    st.write(f"Rows available: `{df.shape[0]}` | Numeric columns: `{len(numeric_columns)}`")
    st.dataframe(df.head(10), use_container_width=True)

    if len(numeric_columns) < 2:
        st.warning("The dataset needs at least two numeric columns to train a predictive model.")
        st.stop()

    default_target = "Avg_Reviews" if "Avg_Reviews" in numeric_columns else numeric_columns[0]
    target = st.selectbox(
        "Select the numeric target to predict",
        options=numeric_columns,
        index=numeric_columns.index(default_target),
    )
    
    default_features = [
        column for column in ["Percentage_non_male_num", "Sexualization"]
        if column in numeric_columns and column != target
    ]
    
    if not default_features:
        default_features = [
            column for column in numeric_columns if column != target
        ][:4]
    
    features = st.multiselect(
        "Select numeric input features",
        options=[column for column in numeric_columns if column != target],
        default=default_features,
        )
        
        # Data leakage warning
    features = st.multiselect(
        "Select numeric input features",
        options=[column for column in numeric_columns if column != target],
        default=default_features,
    )
    
    leakage_features = ["Metacritic", "IGN", "GameSpot", "Destructoid"]
    
    selected_leakage_features = [
        col for col in leakage_features if col in features
    ]
    
    if selected_leakage_features:
        st.warning(
            "⚠️ Data leakage risk warning: "
            f"{', '.join(selected_leakage_features)} are also critics' review scores. "
            "Predicting one review score from another can give misleadingly high "
            "performance and may cause data leakage."
        )
    
    split_col, random_col = st.columns(2)
    with split_col:
        split_pct = st.slider("Validation split (%)", min_value=10, max_value=40, value=30, step=5)
    with random_col:
        random_state = st.number_input("Random state", min_value=0, value=45, step=1)

    if not features:
        st.info("Select at least one feature to prepare the training split.")
        st.stop()

    modeling_df, X, y = prepare_supervised_data(df, target, features)
    if len(modeling_df) < 8:
        st.warning("Not enough complete rows remain after dropping missing values for the selected columns.")
        st.stop()

    X_train, X_val, y_train, y_val = split_dataset(X, y, split_pct, random_state=int(random_state))
    store_training_state(
        modeling_df=modeling_df,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        target=target,
        features=features,
        split_pct=split_pct,
        random_state=int(random_state),
    )

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Complete rows used", len(modeling_df))
    summary_col2.metric("Training rows", len(X_train))
    summary_col3.metric("Validation rows", len(X_val))
    st.caption(
        "The current split is stored in session state and reused by the evaluation and deployment page."
    )

    st.markdown("## Train Candidate Models")
    st.info(
        "Train several candidates here, then compare them on the Test Model page. "
        "The deployed model should be the one with the best validation tradeoff, not just the lowest training error."
    )

    candidate_models = st.multiselect(
        "Select models to train",
        options=REGRESSION_MODEL_NAMES + CLASSIFICATION_MODEL_NAMES,
        default=["Multiple Linear Regression", "Ridge Regression", "Lasso Regression"],
    )

    if "Multiple Linear Regression" in candidate_models:
        with st.expander("Multiple Linear Regression", expanded=True):
            learning_rate = st.number_input(
                "Learning rate",
                min_value=0.0001,
                max_value=1.0,
                value=0.01,
                step=0.0001,
                format="%.4f",
                key="linear_lr",
            )
            num_iterations = st.number_input(
                "Gradient descent iterations",
                min_value=10,
                max_value=10000,
                value=500,
                step=10,
                key="linear_iter",
            )
            if st.button("Train Multiple Linear Regression", key="train_linear_button"):
                model = build_model(
                    "Multiple Linear Regression",
                    {
                        "learning_rate": float(learning_rate),
                        "num_iterations": int(num_iterations),
                    },
                )
                model.fit(X_train, y_train)
                register_trained_model(
                    "Multiple Linear Regression",
                    model,
                    task_type="regression",
                    target=target,
                    features=features,
                    hyperparameters={
                        "learning_rate": float(learning_rate),
                        "num_iterations": int(num_iterations),
                    },
                )
                st.success("Multiple Linear Regression trained and saved to session state.")
                plot_cost_history("Multiple Linear Regression", model)

    if "Polynomial Regression" in candidate_models:
        with st.expander("Polynomial Regression"):
            degree = st.number_input(
                "Polynomial degree",
                min_value=2,
                max_value=6,
                value=2,
                step=1,
                key="poly_degree",
            )
            learning_rate = st.number_input(
                "Learning rate ",
                min_value=0.00001,
                max_value=1.0,
                value=0.001,
                step=0.00001,
                format="%.5f",
                key="poly_lr",
            )
            num_iterations = st.number_input(
                "Gradient descent iterations ",
                min_value=10,
                max_value=10000,
                value=400,
                step=10,
                key="poly_iter",
            )
            if st.button("Train Polynomial Regression", key="train_poly_button"):
                model = build_model(
                    "Polynomial Regression",
                    {
                        "degree": int(degree),
                        "learning_rate": float(learning_rate),
                        "num_iterations": int(num_iterations),
                    },
                )
                model.fit(X_train, y_train)
                register_trained_model(
                    "Polynomial Regression",
                    model,
                    task_type="regression",
                    target=target,
                    features=features,
                    hyperparameters={
                        "degree": int(degree),
                        "learning_rate": float(learning_rate),
                        "num_iterations": int(num_iterations),
                    },
                )
                st.success("Polynomial Regression trained and saved to session state.")
                plot_cost_history("Polynomial Regression", model)

    if "Ridge Regression" in candidate_models:
        with st.expander("Ridge Regression"):
            learning_rate = st.number_input(
                "Learning rate  ",
                min_value=0.0001,
                max_value=1.0,
                value=0.01,
                step=0.0001,
                format="%.4f",
                key="ridge_lr",
            )
            num_iterations = st.number_input(
                "Gradient descent iterations  ",
                min_value=10,
                max_value=10000,
                value=500,
                step=10,
                key="ridge_iter",
            )
            l2_penalty = st.number_input(
                "L2 penalty",
                min_value=0.0,
                max_value=10.0,
                value=0.5,
                step=0.1,
                key="ridge_penalty",
            )
            if st.button("Train Ridge Regression", key="train_ridge_button"):
                model = build_model(
                    "Ridge Regression",
                    {
                        "learning_rate": float(learning_rate),
                        "num_iterations": int(num_iterations),
                        "l2_penalty": float(l2_penalty),
                    },
                )
                model.fit(X_train, y_train)
                register_trained_model(
                    "Ridge Regression",
                    model,
                    task_type="regression",
                    target=target,
                    features=features,
                    hyperparameters={
                        "learning_rate": float(learning_rate),
                        "num_iterations": int(num_iterations),
                        "l2_penalty": float(l2_penalty),
                    },
                )
                st.success("Ridge Regression trained and saved to session state.")
                plot_cost_history("Ridge Regression", model)

    if "Lasso Regression" in candidate_models:
        with st.expander("Lasso Regression"):
            learning_rate = st.number_input(
                "Learning rate   ",
                min_value=0.0001,
                max_value=1.0,
                value=0.001,
                step=0.0001,
                format="%.4f",
                key="lasso_lr",
            )
            num_iterations = st.number_input(
                "Gradient descent iterations   ",
                min_value=10,
                max_value=10000,
                value=500,
                step=10,
                key="lasso_iter",
            )
            l1_penalty = st.number_input(
                "L1 penalty",
                min_value=0.0,
                max_value=10.0,
                value=0.5,
                step=0.1,
                key="lasso_penalty",
            )
            if st.button("Train Lasso Regression", key="train_lasso_button"):
                model = build_model(
                    "Lasso Regression",
                    {
                        "learning_rate": float(learning_rate),
                        "num_iterations": int(num_iterations),
                        "l1_penalty": float(l1_penalty),
                    },
                )
                model.fit(X_train, y_train)
                register_trained_model(
                    "Lasso Regression",
                    model,
                    task_type="regression",
                    target=target,
                    features=features,
                    hyperparameters={
                        "learning_rate": float(learning_rate),
                        "num_iterations": int(num_iterations),
                        "l1_penalty": float(l1_penalty),
                    },
                )
                st.success("Lasso Regression trained and saved to session state.")
                plot_cost_history("Lasso Regression", model)

    if "Naive Bayes" in candidate_models:
        with st.expander("Naive Bayes Classifier"):
            st.caption(
                "This classifier is kept as a supplementary analysis tool. "
                "It predicts whether a score is high or low rather than estimating an exact review score."
            )
            threshold = st.slider(
                "High-score threshold",
                min_value=0.0,
                max_value=10.0,
                value=7.5,
                step=0.1,
                key="nb_threshold",
            )
            nb_features = st.multiselect(
                "Select numeric features for Naive Bayes",
                options=[column for column in numeric_columns if column != target],
                default=features,
                key="nb_features",
            )
            if st.button("Train Naive Bayes", key="train_nb_button"):
                if not nb_features:
                    st.warning("Select at least one feature for Naive Bayes.")
                else:
                    nb_df, X_nb, y_reg = prepare_supervised_data(df, target, nb_features)
                    y_nb = (y_reg.reshape(-1) >= threshold).astype(int)
                    X_nb_train, X_nb_val, y_nb_train, y_nb_val = split_dataset(
                        X_nb,
                        y_nb,
                        split_pct,
                        random_state=int(random_state),
                    )
                    model = build_model("Naive Bayes", {})
                    model.fit(X_nb_train, y_nb_train)
                    register_trained_model(
                        "Naive Bayes",
                        model,
                        task_type="classification",
                        target=target,
                        features=nb_features,
                        hyperparameters={"threshold": float(threshold)},
                    )
                    st.session_state["naive_bayes_eval_state"] = {
                        "X_val": X_nb_val,
                        "y_val": y_nb_val,
                        "threshold": float(threshold),
                    }
                    st.success("Naive Bayes trained and saved to session state.")

    st.markdown("## Trained Model Registry")
    show_model_registry()
    st.write("Continue to **Test Model** to compare validation metrics and deploy the best regression model.")
