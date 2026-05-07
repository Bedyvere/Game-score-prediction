import os
import re
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import OrdinalEncoder


st.markdown("# Practical Applications of Machine Learning (PAML)")
st.markdown("### Game Score Prediction and Critics Analysis")
st.markdown("# Exploratory Dataset Analysis")


DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
DEFAULT_DATASET_PATH = os.path.abspath(
    os.path.join(DATASET_DIR, "merged_grivg_data.csv")
)

SESSION_DATA_KEY = "game_df"
LEGACY_SESSION_DATA_KEY = "house_df"
SESSION_RAW_KEY = "raw_game_df"          # cleaned original dataset
SESSION_ORIGINAL_KEY = "original_game_df" # before cleaning


@st.cache_data
def read_csv_data(source):
    return pd.read_csv(source)


def clean_column_name(column):
    """Make column names easier to use in code."""
    column = str(column).strip()
    column = re.sub(r"[\s\-]+", "_", column)
    column = re.sub(r"[^0-9a-zA-Z_]", "", column)
    column = re.sub(r"_+", "_", column)
    return column.strip("_")


def parse_percentage(series):
    """Convert values like '45%' into 0.45 for modeling."""
    cleaned = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "Unknown": np.nan})
    )
    values = pd.to_numeric(cleaned, errors="coerce")

    # The GRIVG dataset stores percentages like 18%, 56%, etc.
    # For ML, 0.18 is usually easier to use than 18.
    if values.dropna().max() > 1:
        values = values / 100
    return values


def preprocess_merged_dataset(df):
    """Basic automatic cleaning for the merged GRIVG dataset."""
    cleaned = df.copy()

    # Clean column names first, so later code can safely use underscore names.
    cleaned.columns = [clean_column_name(col) for col in cleaned.columns]

    # Remove empty columns and useless exported index columns.
    cleaned = cleaned.dropna(axis=1, how="all")
    cleaned = cleaned.loc[:, ~cleaned.columns.str.startswith("Unnamed")]

    # Clean extra spaces inside text values.
    for column in cleaned.select_dtypes(include="object").columns:
        cleaned[column] = cleaned[column].astype(str).str.strip()
        cleaned[column] = cleaned[column].replace(
            {"": np.nan, "nan": np.nan, "None": np.nan, "Unknown": np.nan}
        )

    # Convert columns that should be numeric.
    numeric_columns = [
        "Playable",
        "Sexualization",
        "Protagonist",
        "Protagonist_Non_Male",
        "Relevant_males",
        "Relevant_no_males",
        "Total_team",
        "female_team",
        "Metacritic",
        "Destructoid",
        "IGN",
        "GameSpot",
        "Avg_Reviews",
        "Sexualized_clothing",
        "Trophy",
        "Damsel_in_Distress",
        "Sexualized_Cutscenes",
        "Total",
        "PEGI",
    ]
    for column in numeric_columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    # Convert percentage columns into 0-1 numeric values.
    if "Percentage_non_male" in cleaned.columns:
        cleaned["Percentage_non_male_num"] = parse_percentage(
            cleaned["Percentage_non_male"]
        )
    if "Team_percentage" in cleaned.columns:
        cleaned["Team_percentage_num"] = parse_percentage(cleaned["Team_percentage"])

    # Create simple 0/1 flags.
    if "Customizable_main" in cleaned.columns:
        cleaned["Customizable_main_flag"] = cleaned["Customizable_main"].map(
            {"Yes": 1, "No": 0, "Non_Binary": 1, "Non-Binary": 1}
        )

    if "Romantic_Interest" in cleaned.columns:
        cleaned["Romantic_Interest_flag"] = cleaned["Romantic_Interest"].map(
            {"Yes": 1, "Opt": 1, "No": 0}
        )

    # Create Age_numeric from exact ages first, then use Age_range as a backup.
    if "Age" in cleaned.columns:
        cleaned["Age_numeric"] = pd.to_numeric(cleaned["Age"], errors="coerce")

    if "Age_range" in cleaned.columns:
        age_label_map = {
            "Infant": 3,
            "Child": 10,
            "Teenager": 16,
            "Young_adult": 21,
            "Young_Adult": 21,
            "Young adult": 21,
            "Adult": 30,
            "Middle_aged": 50,
            "Middle_Aged": 50,
            "Middle-aged": 50,
            "Elderly": 70,
        }
        if "Age_numeric" not in cleaned.columns:
            cleaned["Age_numeric"] = np.nan
        cleaned["Age_numeric"] = cleaned["Age_numeric"].fillna(
            cleaned["Age_range"].map(age_label_map)
        )

    # Create release year and month.
    if "Release" in cleaned.columns:
        release_dt = pd.to_datetime(cleaned["Release"], format="%b-%y", errors="coerce")
        cleaned["Release_Date"] = release_dt
        cleaned["Release_Year"] = release_dt.dt.year
        cleaned["Release_Month"] = release_dt.dt.month

    return cleaned


def load_default_dataset():
    original_df = read_csv_data(DEFAULT_DATASET_PATH)
    cleaned_df = preprocess_merged_dataset(original_df)
    st.session_state[SESSION_ORIGINAL_KEY] = original_df.copy()
    st.session_state[SESSION_RAW_KEY] = cleaned_df.copy()
    st.session_state[SESSION_DATA_KEY] = cleaned_df.copy()
    st.session_state[LEGACY_SESSION_DATA_KEY] = cleaned_df.copy()
    return cleaned_df


def initialize_dataset():
    if SESSION_DATA_KEY not in st.session_state:
        return load_default_dataset()
    st.session_state[LEGACY_SESSION_DATA_KEY] = st.session_state[SESSION_DATA_KEY].copy()
    return st.session_state[SESSION_DATA_KEY]


def set_current_df(df):
    st.session_state[SESSION_DATA_KEY] = df.copy()
    st.session_state[LEGACY_SESSION_DATA_KEY] = df.copy()


def reset_current_df():
    st.session_state[SESSION_DATA_KEY] = st.session_state[SESSION_RAW_KEY].copy()
    st.session_state[LEGACY_SESSION_DATA_KEY] = st.session_state[SESSION_RAW_KEY].copy()


def summarize_missing_data(df):
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    return {
        "num_columns_with_missing": int((df.isna().sum() > 0).sum()),
        "total_missing_values": int(df.isna().sum().sum()),
        "top_missing": missing.head(10),
    }


def compute_correlation(df, features):
    if len(features) < 2:
        return None, []
    correlation = df[features].corr()
    summary = []
    for f1, f2 in combinations(features, 2):
        cor = correlation.loc[f1, f2]
        if pd.isna(cor):
            continue
        strength = "strongly" if abs(cor) >= 0.5 else "weakly"
        direction = "positively" if cor >= 0 else "negatively"
        summary.append(f"- {f1} and {f2} are {strength} {direction} correlated: {cor:.2f}")
    return correlation, summary


def remove_features(df, removed_features):
    if not removed_features:
        return df
    return df.drop(columns=removed_features, errors="ignore")


def impute_dataset(df, numeric_method, categorical_fill_value="Missing"):
    updated = df.copy()
    numeric_columns = updated.select_dtypes(include="number").columns
    categorical_columns = updated.select_dtypes(exclude="number").columns

    if numeric_method == "Zero":
        updated[numeric_columns] = updated[numeric_columns].fillna(0)
    elif numeric_method == "Mean":
        updated[numeric_columns] = updated[numeric_columns].fillna(updated[numeric_columns].mean())
    elif numeric_method == "Median":
        updated[numeric_columns] = updated[numeric_columns].fillna(updated[numeric_columns].median())
    elif numeric_method == "Drop Rows":
        updated = updated.dropna()

    if numeric_method != "Drop Rows" and len(categorical_columns) > 0:
        updated[categorical_columns] = updated[categorical_columns].fillna(categorical_fill_value)

    return updated


def get_outlier_appropriate_columns(df):
    candidates = []
    excluded = {}
    for column in df.select_dtypes(include="number").columns:
        series = df[column].dropna()
        if len(series) == 0:
            excluded[column] = "all NaN"
            continue
        if series.nunique() <= 2:
            excluded[column] = "binary or near-binary"
            continue
        if series.std() == 0:
            excluded[column] = "zero variance"
            continue
        candidates.append(column)
    return candidates, excluded


def remove_outliers(df, features, method):
    updated = df.copy()
    removed_rows = {}
    for feature in features:
        series = updated[feature].dropna()
        if len(series) == 0:
            continue

        before = len(updated)
        if method == "IQR":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
        else:
            mean = series.mean()
            std = series.std()
            if std == 0:
                continue
            lower = mean - 3 * std
            upper = mean + 3 * std

        updated = updated[(updated[feature].isna()) | ((updated[feature] >= lower) & (updated[feature] <= upper))]
        removed_rows[feature] = before - len(updated)

    return updated, removed_rows


def one_hot_encode_feature(df, features):
    if not features:
        return df
    return pd.get_dummies(df, columns=features, dummy_na=False)


def integer_encode_feature(df, features):
    if not features:
        return df
    updated = df.copy()
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoded = encoder.fit_transform(updated[features].fillna("Missing"))
    for idx, feature in enumerate(features):
        updated[f"{feature}_int"] = encoded[:, idx].astype(int)
    return updated


def create_feature(df, math_select, math_feature_select, new_feature_name):
    updated = df.copy()
    if not new_feature_name:
        return updated

    if math_select in {"square root", "ceil", "floor"}:
        feature = math_feature_select[0]
        if math_select == "square root":
            updated[new_feature_name] = np.where(
                updated[feature] >= 0, np.sqrt(updated[feature]), np.nan
            )
        elif math_select == "ceil":
            updated[new_feature_name] = np.ceil(updated[feature])
        elif math_select == "floor":
            updated[new_feature_name] = np.floor(updated[feature])
        return updated

    left_feature, right_feature = math_feature_select
    if math_select == "add":
        updated[new_feature_name] = updated[left_feature] + updated[right_feature]
    elif math_select == "subtract":
        updated[new_feature_name] = updated[left_feature] - updated[right_feature]
    elif math_select == "multiply":
        updated[new_feature_name] = updated[left_feature] * updated[right_feature]
    elif math_select == "divide":
        denominator = updated[right_feature].replace(0, np.nan)
        updated[new_feature_name] = updated[left_feature] / denominator
    return updated


def compute_descriptive_stats(df, features):
    if not features:
        return pd.DataFrame()
    return df[features].describe().T[["mean", "50%", "min", "max", "std"]].rename(
        columns={"50%": "median"}
    )


def scale_features(df, features, scaling_method):
    updated = df.copy()
    for feature in features:
        series = updated[feature]
        if scaling_method == "Standardization":
            std = series.std()
            updated[f"{feature}_std"] = np.nan if std == 0 else (series - series.mean()) / std
        elif scaling_method == "Normalization":
            value_range = series.max() - series.min()
            updated[f"{feature}_norm"] = np.nan if value_range == 0 else (series - series.min()) / value_range
        elif scaling_method == "Log1p":
            updated[f"{feature}_log1p"] = np.where(series >= 0, np.log1p(series), np.nan)
    return updated


def get_categorical_columns(df):
    return list(df.select_dtypes(include=["object", "category", "bool"]).columns)


def show_project_quick_charts(df):
    st.markdown("### 3. Project-related quick charts")
    st.caption("These are simple charts connected to our game score and gender representation question.")

    chart_count = 0

    if {"Percentage_non_male_num", "Avg_Reviews"}.issubset(df.columns):
        fig = px.scatter(
            df,
            x="Percentage_non_male_num",
            y="Avg_Reviews",
            color="Gender" if "Gender" in df.columns else None,
            hover_name="Title" if "Title" in df.columns else None,
            title="Non-male character percentage vs average review score",
        )
        st.plotly_chart(fig, use_container_width=True)
        chart_count += 1

    if {"Gender", "Avg_Reviews"}.issubset(df.columns):
        fig = px.box(
            df,
            x="Gender",
            y="Avg_Reviews",
            points="outliers",
            title="Average review score distribution by character gender",
        )
        st.plotly_chart(fig, use_container_width=True)
        chart_count += 1

    if {"Release_Year", "Avg_Reviews"}.issubset(df.columns):
        yearly_score = (
            df.groupby("Release_Year", as_index=False)["Avg_Reviews"]
            .mean()
            .dropna()
        )
        fig = px.line(
            yearly_score,
            x="Release_Year",
            y="Avg_Reviews",
            markers=True,
            title="Average review score by release year",
        )
        st.plotly_chart(fig, use_container_width=True)
        chart_count += 1

    if {"Release_Year", "Percentage_non_male_num"}.issubset(df.columns):
        yearly_gender = (
            df.groupby("Release_Year", as_index=False)["Percentage_non_male_num"]
            .mean()
            .dropna()
        )
        fig = px.line(
            yearly_gender,
            x="Release_Year",
            y="Percentage_non_male_num",
            markers=True,
            title="Average non-male character percentage by release year",
        )
        st.plotly_chart(fig, use_container_width=True)
        chart_count += 1

    if chart_count == 0:
        st.info("The current dataset does not include the columns needed for the quick project charts.")


st.markdown("### Import Dataset")
source_option = st.radio(
    "Select data source",
    ["Use game data in /datasets", "Upload another CSV"],
    horizontal=True,
)

if source_option == "Upload another CSV":
    uploaded_file = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
    if uploaded_file is not None:
        original_df = read_csv_data(uploaded_file)
        uploaded_df = preprocess_merged_dataset(original_df)
        st.session_state[SESSION_ORIGINAL_KEY] = original_df.copy()
        st.session_state[SESSION_RAW_KEY] = uploaded_df.copy()
        st.session_state[SESSION_DATA_KEY] = uploaded_df.copy()
        st.session_state[LEGACY_SESSION_DATA_KEY] = uploaded_df.copy()
        df = st.session_state[SESSION_DATA_KEY]
    else:
        st.info("Please upload a CSV file first.")
        st.stop()
else:
    if st.session_state.get("current_source") != "default":
        load_default_dataset()
        st.session_state["current_source"] = "default"
    df = initialize_dataset()

left_control, right_control = st.columns([1, 1])
with left_control:
    if st.button("Reset to original cleaned data"):
        reset_current_df()
        df = st.session_state[SESSION_DATA_KEY]
with right_control:
    st.download_button(
        "Download current dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="processed_game_dataset.csv",
        mime="text/csv",
    )


if df is not None:
    st.markdown("### 1. Dataset Overview")
    st.write(f"Rows: `{df.shape[0]}` | Columns: `{df.shape[1]}`")
    st.dataframe(df.head(20), use_container_width=True)

    missing_summary = summarize_missing_data(df)
    original_df = st.session_state.get(SESSION_ORIGINAL_KEY)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Rows", df.shape[0])
    metric_col2.metric("Columns", df.shape[1])
    metric_col3.metric("Missing values", missing_summary["total_missing_values"])
    metric_col4.metric("Numeric columns", len(df.select_dtypes(include="number").columns))

    if original_df is not None:
        st.markdown("### 2. Cleaning result")
        clean_col1, clean_col2, clean_col3 = st.columns(3)
        clean_col1.metric("Rows before cleaning", original_df.shape[0])
        clean_col2.metric("Columns before cleaning", original_df.shape[1])
        clean_col3.metric("Columns after cleaning", df.shape[1])

    with st.expander("Data types and missing values"):
        dtype_df = pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "missing_count": df.isna().sum(),
                "missing_rate": (df.isna().mean() * 100).round(2),
            }
        )
        st.dataframe(dtype_df, use_container_width=True)
        if not missing_summary["top_missing"].empty:
            st.write("Top columns with missing values")
            st.dataframe(
                missing_summary["top_missing"].rename("missing_count").to_frame(),
                use_container_width=True,
            )

    st.markdown("### Automatic preprocessing applied for game data")
    st.markdown(
        """
        - Removed fully empty / `Unnamed` columns
        - Cleaned column names into simple underscore style
        - Trimmed column names and string values
        - Converted review and indicator columns to numeric
        - Created `Percentage_non_male_num` and `Team_percentage_num` as 0-1 numeric values
        - Created `Age_numeric`, `Release_Year`, and `Release_Month`
        - Created binary flags for `Customizable_main` and `Romantic_Interest`, with `Opt` counted as 1
        """
    )

    show_project_quick_charts(df)

    st.markdown("### 4. Custom visualizations")
    numeric_columns = list(df.select_dtypes(include="number").columns)
    categorical_columns = get_categorical_columns(df)

    st.sidebar.header("Visualization Controls")
    chart_select = st.sidebar.selectbox(
        "Type of chart",
        ["Scatter", "Histogram", "Box", "Bar"],
    )

    if numeric_columns:
        if chart_select == "Scatter":
            x_axis = st.sidebar.selectbox("X axis", numeric_columns, key="scatter_x")
            y_axis = st.sidebar.selectbox("Y axis", numeric_columns, key="scatter_y")
            color_axis = st.sidebar.selectbox("Color by", [None] + categorical_columns, key="scatter_color")
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_axis, hover_data=df.columns)
            st.plotly_chart(fig, use_container_width=True)

        if chart_select == "Histogram":
            x_axis = st.sidebar.selectbox("Feature", numeric_columns, key="hist_x")
            color_axis = st.sidebar.selectbox("Color by", [None] + categorical_columns, key="hist_color")
            fig = px.histogram(df, x=x_axis, color=color_axis, marginal="box", nbins=30)
            st.plotly_chart(fig, use_container_width=True)

        if chart_select == "Box":
            y_axis = st.sidebar.selectbox("Numeric feature", numeric_columns, key="box_y")
            x_axis = st.sidebar.selectbox("Group by", [None] + categorical_columns, key="box_x")
            fig = px.box(df, x=x_axis, y=y_axis, points="outliers")
            st.plotly_chart(fig, use_container_width=True)

        if chart_select == "Bar":
            if categorical_columns:
                x_axis = st.sidebar.selectbox("Category", categorical_columns, key="bar_x")
                value_counts = (
                    df[x_axis]
                    .fillna("Missing")
                    .value_counts()
                    .reset_index()
                )
                value_counts.columns = [x_axis, "count"]
                fig = px.bar(value_counts, x=x_axis, y="count")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categorical columns available for bar chart.")
    else:
        st.warning("No numeric columns available for visualization.")

    st.markdown("### 5. Remove irrelevant features")
    removed_features = st.multiselect("Select columns to drop", df.columns)
    if st.button("Apply column removal"):
        df = remove_features(df, removed_features)
        set_current_df(df)
        st.success("Selected columns removed from current dataset.")

    st.markdown("### 6. Handle missing values")
    impute_method = st.selectbox(
        "Numeric missing-value strategy",
        ["Mean", "Median", "Zero", "Drop Rows"],
    )
    categorical_fill_value = st.text_input(
        "Fill value for categorical columns",
        value="Missing",
    )
    if st.button("Apply missing-value handling"):
        df = impute_dataset(df, impute_method, categorical_fill_value)
        set_current_df(df)
        st.success("Missing-value handling applied.")

    st.markdown("### 7. Encode categorical features")
    string_columns = get_categorical_columns(df)
    encode_col1, encode_col2 = st.columns(2)

    with encode_col1:
        ordinal_features = st.multiselect(
            "Columns for integer encoding",
            string_columns,
            key="ordinal_features",
        )
        if st.button("Apply integer encoding"):
            df = integer_encode_feature(df, ordinal_features)
            set_current_df(df)
            st.success("Integer encoding completed.")

    with encode_col2:
        one_hot_features = st.multiselect(
            "Columns for one-hot encoding",
            string_columns,
            key="one_hot_features",
        )
        if st.button("Apply one-hot encoding"):
            df = one_hot_encode_feature(df, one_hot_features)
            set_current_df(df)
            st.success("One-hot encoding completed.")

    st.markdown("### 8. Feature scaling")
    current_numeric_columns = list(df.select_dtypes(include="number").columns)
    if current_numeric_columns:
        scale_method = st.selectbox(
            "Scaling method",
            ["Standardization", "Normalization", "Log1p"],
        )
        scale_features_select = st.multiselect(
            "Select numeric features to scale",
            current_numeric_columns,
        )
        if st.button("Apply scaling"):
            df = scale_features(df, scale_features_select, scale_method)
            set_current_df(df)
            st.success("Feature scaling completed.")
    else:
        st.info("No numeric columns available for scaling.")

    st.markdown("### 9. Create new features")
    numeric_columns = list(df.select_dtypes(include="number").columns)
    if numeric_columns:
        math_select = st.selectbox(
            "Mathematical operation",
            ["add", "subtract", "multiply", "divide", "square root", "ceil", "floor"],
        )

        if math_select in {"square root", "ceil", "floor"}:
            unary_feature = st.selectbox("Select one numeric feature", numeric_columns, key="unary_feature")
            new_feature_name = st.text_input("New feature name", key="new_feature_unary")
            if st.button("Create feature", key="create_unary_feature"):
                df = create_feature(df, math_select, [unary_feature], new_feature_name)
                set_current_df(df)
                st.success("New feature created.")
        else:
            math_feature_1 = st.selectbox("Feature 1", numeric_columns, key="math_feature_1")
            math_feature_2 = st.selectbox("Feature 2", numeric_columns, key="math_feature_2")
            new_feature_name = st.text_input("New feature name", key="new_feature_binary")
            if st.button("Create feature", key="create_binary_feature"):
                df = create_feature(
                    df,
                    math_select,
                    [math_feature_1, math_feature_2],
                    new_feature_name,
                )
                set_current_df(df)
                st.success("New feature created.")
    else:
        st.info("No numeric columns available for feature creation.")

    st.markdown("### 10. Remove outliers")
    outlier_method = st.selectbox("Outlier detection method", ["IQR", "STD"])
    appropriate_columns, excluded_columns = get_outlier_appropriate_columns(df)
    if excluded_columns:
        with st.expander("Excluded columns"):
            st.json(excluded_columns)

    if appropriate_columns:
        outlier_features = st.multiselect(
            "Select features for outlier removal",
            appropriate_columns,
        )
        if st.button("Apply outlier removal"):
            df, removed_rows = remove_outliers(df, outlier_features, outlier_method)
            set_current_df(df)
            if removed_rows:
                st.write(pd.DataFrame.from_dict(removed_rows, orient="index", columns=["rows_removed"]))
            st.success("Outlier removal completed.")
    else:
        st.info("No numeric columns are suitable for outlier removal.")

    st.markdown("### 11. Descriptive statistics")
    stats_numeric_columns = list(df.select_dtypes(include="number").columns)
    if stats_numeric_columns:
        stats_feature_select = st.multiselect(
            "Select numeric features",
            stats_numeric_columns,
            key="stats_features",
        )
        stats_df = compute_descriptive_stats(df, stats_feature_select)
        if not stats_df.empty:
            st.dataframe(stats_df, use_container_width=True)
    else:
        st.info("No numeric columns available for descriptive statistics.")

    st.markdown("### 12. Correlation analysis")
    correlation_numeric_columns = list(df.select_dtypes(include="number").columns)
    if correlation_numeric_columns:
        correlation_features = st.multiselect(
            "Select numeric features for correlation",
            correlation_numeric_columns,
            key="correlation_features",
        )
        correlation_df, correlation_summary = compute_correlation(df, correlation_features)
        if correlation_df is not None:
            heatmap = px.imshow(
                correlation_df,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                aspect="auto",
            )
            st.plotly_chart(heatmap, use_container_width=True)
            if 2 <= len(correlation_features) <= 6:
                scatter_fig = px.scatter_matrix(df, dimensions=correlation_features)
                st.plotly_chart(scatter_fig, use_container_width=True)
            for line in correlation_summary:
                st.write(line)
    else:
        st.info("No numeric columns available for correlation analysis.")

    st.markdown("### 13. Current processed dataset")
    st.dataframe(df, use_container_width=True)
    st.info("The current processed dataframe has been stored in `st.session_state['game_df']`.")
