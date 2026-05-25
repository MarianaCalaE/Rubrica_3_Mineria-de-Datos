# -*- coding: utf-8 -*-
"""
================================================================================
IRIS SPECIES CLASSIFICATION - Streamlit Dashboard
================================================================================
Integrantes:
  - Yairineth Camargo
  - Mariana Cala
  - Edgardo Pacheco

Description:
    Interactive dashboard that covers the full data mining pipeline for the
    classic Iris dataset: exploratory analysis, feature engineering,
    model training, performance evaluation, and live prediction.

Run with:
    streamlit run datamining_dashboard.py

Requirements:
    pip install streamlit scikit-learn pandas numpy matplotlib seaborn
================================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# 1. IMPORTING LIBRARIES AND ENVIRONMENT SETUP
# ──────────────────────────────────────────────────────────────────────────────
# In this initial section, the essential software components required for the
# development of the entire data mining pipeline are loaded. The objective is
# to configure a robust working environment that provides advanced tools for
# structured data manipulation, statistical visualization generation, variable
# preprocessing, and the methodological partitioning of the dataset.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
# Sets the browser tab title, icon and default layout before any other
# Streamlit element is rendered.

st.set_page_config(
    page_title="Iris Classification Dashboard",
    page_icon="🌸",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# 2. DATASET LOADING AND STRUCTURING  (Load Dataset)
# ──────────────────────────────────────────────────────────────────────────────
# In this section, the data is formally imported into the execution environment.
# The native Scikit-Learn dataset is converted into a tabular structure
# (DataFrame). This transformation facilitates data manipulation, visual
# inspection, and exploratory analysis in the subsequent stages of the project.
#
# @st.cache_data is used so the dataset is loaded only once across re-runs,
# improving dashboard performance significantly.

@st.cache_data
def load_data():
    """Load the Iris dataset and return it as a clean DataFrame."""
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "species"})
    return df, iris.target_names


# ──────────────────────────────────────────────────────────────────────────────
# 13. FEATURE SCALING & MODEL TRAINING  (Feature Scaling + Modeling)
# ──────────────────────────────────────────────────────────────────────────────
# As previously observed in the boxplots, the variables within the dataset
# operate on different ranges and magnitudes. Many Machine Learning algorithms
# are sensitive to these scale differences. Standardization is applied to
# ensure that all features contribute with equal mathematical weight during
# the training process.
#
# Four fundamental algorithms are evaluated:
#   - Logistic Regression : robust, linear, highly interpretable.
#   - K-Nearest Neighbors : non-parametric, instance-based, distance-driven.
#   - Decision Tree       : hierarchical classifier based on logical rules.
#   - Random Forest       : ensemble (Bagging) of multiple decision trees.
#
# Performance is measured with a *macro average* so every species receives
# equal weight in the global metrics.
#
# @st.cache_data prevents re-training on every Streamlit interaction.

@st.cache_data
def train_models(_df):
    """
    Split the dataset, scale features, train all four classifiers,
    and return the evaluation results alongside the split data.
    """
    # ── 11. Feature Selection: separate X (features) and y (target) ──────────
    X = _df.drop("species", axis=1)
    y = _df["species"]

    # ── 12. Dataset Splitting (Train-Test Split) ──────────────────────────────
    # The dataset is split 80 % training / 20 % testing.
    # stratify=y guarantees that each species is proportionally represented
    # in both subsets, preventing class imbalance during evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 13. StandardScaler: zero mean, unit variance ─────────────────────────
    # fit_transform on training data; transform (only) on test data to prevent
    # data leakage from the test set into the scaling parameters.
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Model Definition with basic hyperparameters to prevent overfitting ────
    modelos = {
        "Logistic Regression":       LogisticRegression(max_iter=200, random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree":             DecisionTreeClassifier(max_depth=3, random_state=42),
        "Random Forest":             RandomForestClassifier(
                                         n_estimators=100, max_depth=3, random_state=42
                                     ),
    }

    # ── Training, Prediction and Metric Calculation loop ─────────────────────
    # Each model is fitted on the scaled training matrix, then predictions are
    # generated on the scaled test set. Metrics use macro average so every
    # class contributes equally regardless of sample count.
    resultados = {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_train_sc, y_train)
        y_pred = modelo.predict(X_test_sc)

        resultados[nombre] = {
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, average="macro"), 4),
            "Recall":    round(recall_score(y_test, y_pred, average="macro"), 4),
            "F1-Score":  round(f1_score(y_test, y_pred, average="macro"), 4),
            "Matriz":    confusion_matrix(y_test, y_pred),
            "Modelo":    modelo,
        }

    return resultados, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler


# ── Load data and train models at startup ─────────────────────────────────────
df, target_names = load_data()
resultados, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler = train_models(df)

# Convenience mappings used throughout the dashboard
SPECIES_NAMES = list(target_names)                      # ['setosa', 'versicolor', 'virginica']
SPECIES_MAP   = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Navigation
# ──────────────────────────────────────────────────────────────────────────────
# The sidebar provides a persistent navigation menu so the user can jump between
# the four main sections of the pipeline without losing the cached state.

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/"
        "Iris_versicolor_3.jpg/320px-Iris_versicolor_3.jpg",
        use_container_width=True,
    )
    st.title("🌸 Iris Dashboard")
    st.markdown(
        "**Integrantes:**\n"
        "- Yairineth Camargo\n"
        "- Mariana Cala\n"
        "- Edgardo Pacheco"
    )
    st.divider()
    seccion = st.radio(
        "Navegar a:",
        [
            "📋 Dataset Overview",
            "📊 Exploratory Analysis",
            "🤖 Model Training & Metrics",
            "🔮 Predict New Sample",
        ],
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
# Covers steps 2–7 of the original notebook:
#   2. Dataset Loading    3. Shape Review       4. Data Types & Info
#   5. Missing Values     6. Duplicate Rows     7. Descriptive Statistics

if seccion == "📋 Dataset Overview":

    st.title("📋 Dataset Overview")
    st.markdown(
        "Complete inspection of the Iris dataset: shape, types, "
        "missing values, duplicates and descriptive statistics."
    )

    # ── 3. Dataset Shape Review ───────────────────────────────────────────────
    # The .shape property returns (rows, columns).
    # 150 rows = individual flower samples; 5 columns = 4 features + target.
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Rows",    df.shape[0])
    k2.metric("Total Columns", df.shape[1])

    # ── 5. Missing Values Analysis ────────────────────────────────────────────
    # The count returns zero missing entries — no imputation is required.
    k3.metric("Missing Values", int(df.isnull().sum().sum()))

    # ── 6. Duplicate Rows Identification ─────────────────────────────────────
    # One duplicated row is detected. Given that only a single duplicate exists
    # its retention or removal would not significantly affect model generalization.
    k4.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        # ── 2. Dataset Loading — first rows ──────────────────────────────────
        st.subheader("First 10 rows")
        st.markdown(
            "The DataFrame displays sepal and petal dimensions (cm) as "
            "predictive variables. The `species` column contains the target class."
        )
        st.dataframe(df.head(10), use_container_width=True)

        # ── 4. Data Types & Info ──────────────────────────────────────────────
        st.subheader("Data Types")
        st.markdown(
            "All four feature columns are `float64` and the target `species` "
            "is `int64`. No type corrections are required."
        )
        dtype_df = df.dtypes.reset_index()
        dtype_df.columns = ["Column", "Type"]
        st.dataframe(dtype_df, use_container_width=True)

    with col2:
        # ── 7. Descriptive Statistical Analysis ──────────────────────────────
        # Petal dimensions show substantially greater dispersion compared to
        # sepal measurements — a strong indicator that petal features contain
        # differentiating patterns that classifiers can leverage effectively.
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe().T.round(4), use_container_width=True)

        # Class distribution table
        st.subheader("Class Distribution")
        dist = df["species"].value_counts().reset_index()
        dist.columns  = ["Species (code)", "Count"]
        dist["Species"] = dist["Species (code)"].map(SPECIES_MAP)
        st.dataframe(dist[["Species", "Count"]], use_container_width=True)

    # ── Class Balance Bar Chart ───────────────────────────────────────────────
    st.subheader("Class Balance")
    fig, ax = plt.subplots(figsize=(5, 3))
    counts = [df[df["species"] == i].shape[0] for i in range(3)]
    ax.bar(SPECIES_NAMES, counts, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("Count")
    ax.set_title("Samples per Species")
    st.pyplot(fig)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
# Covers steps 8–10 of the original notebook:
#   8.  Boxplots — distribution and outlier detection
#   9.  Histograms — frequency distribution and normality check
#   10. Correlation Heatmap + Pairplot — multivariable relationship analysis

elif seccion == "📊 Exploratory Analysis":

    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📦 Boxplots",
        "📈 Histograms",
        "🔥 Correlation Heatmap",
        "🔵 Pairplot",
    ])

    # ── 8. Boxplots — Distribution & Outlier Visualization ───────────────────
    # Boxplots are used to visually evaluate symmetry, dispersion, and outliers
    # that could negatively affect Machine Learning model performance.
    with tab1:
        st.subheader("Distribution & Outliers (Boxplots)")
        st.markdown(
            "Visualizes spread, symmetry, and potential outliers per feature. "
            "Each whisker spans 1.5× the interquartile range; points beyond "
            "that are flagged as outliers."
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df, ax=ax)
        ax.set_title("Boxplots — All Features")
        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "**Key findings:**\n"
            "- Different scales: `petal length` and `sepal length` occupy higher ranges.\n"
            "- `sepal width` is the **only** feature displaying outliers above and below its whiskers.\n"
            "- `petal length` is the most spread-out variable, showing the highest variability."
        )

    # ── 9. Histograms — Frequency Distribution Analysis ──────────────────────
    # Histograms examine whether measurements follow a bell-shaped (normal)
    # distribution or exhibit multimodal behaviour before training begins.
    with tab2:
        st.subheader("Frequency Distribution (Histograms)")
        st.markdown(
            "Each histogram shows how the measurements are distributed across "
            "the dataset. A bimodal shape indicates that two distinct groups "
            "exist within the same feature."
        )

        # Build subplots explicitly so Streamlit can render them correctly.
        # df.hist() accepts an 'ax' array — we create a 2×2 grid for the
        # four numeric feature columns and pass those axes directly.
        feature_cols = [c for c in df.columns if c != "species"]   # 4 features
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
        axes_flat = axes.flatten()

        for i, col in enumerate(feature_cols):
            axes_flat[i].hist(
                df[col], bins=15, color="#4C72B0", edgecolor="white"
            )
            axes_flat[i].set_title(col)
            axes_flat[i].set_xlabel("Value (cm)")
            axes_flat[i].set_ylabel("Frequency")

        plt.suptitle("Frequency Distribution — All Features", fontsize=13, y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "**Key finding:** Petal features show a clear bimodal gap — "
            "*Iris setosa* separates naturally from the other two species, "
            "suggesting high discriminative power for classifiers."
        )

    # ── 10. Correlation Heatmap ───────────────────────────────────────────────
    # Relationships between numerical variables are identified through the use
    # of a correlation matrix using Pearson coefficients.
    with tab3:
        st.subheader("Correlation Heatmap")
        st.markdown(
            "Pearson correlation coefficients between every pair of numeric "
            "features. Values close to **+1** indicate strong positive linear "
            "relationship; values close to **−1** indicate the opposite."
        )

        numeric_df = df.select_dtypes(include="number")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f"
        )
        ax.set_title("Feature Correlation Matrix")
        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "**Key finding:** `petal length` and `petal width` are highly "
            "correlated (≈ 0.96), confirming that petal features are the "
            "primary drivers of species separation."
        )

    # ── 10. Pairplot — Multivariable Relationship Analysis ───────────────────
    # A scatterplot matrix allows us to simultaneously observe individual
    # distributions and correlations between pairs of variables, color-coded
    # by species.
    with tab4:
        st.subheader("Multivariable Pairplot")
        st.markdown(
            "Each point is colored by species. The **diagonal** shows the "
            "univariate distribution (KDE) for each feature."
        )

        df_plot = df.copy()
        df_plot["species_name"] = df_plot["species"].map(SPECIES_MAP)

        pair_fig = sns.pairplot(
            df_plot,
            hue="species_name",
            palette={
                "Setosa":     "#4C72B0",
                "Versicolor": "#DD8452",
                "Virginica":  "#55A868",
            },
        )
        st.pyplot(pair_fig.figure)
        plt.close()

        st.info(
            "**Key finding:** *Setosa* is linearly separable from "
            "*Versicolor* and *Virginica* in nearly all feature combinations. "
            "Versicolor and Virginica overlap slightly in sepal measurements."
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL TRAINING & METRICS
# ══════════════════════════════════════════════════════════════════════════════
# Covers the full Modeling and Evaluation phase:
#   2.1 Model Configuration      2.2 Training & Confusion Matrices
#   2.3 Performance Comparison   2.4 Conclusions & Model Selection

elif seccion == "🤖 Model Training & Metrics":

    st.title("🤖 Model Training & Evaluation")

    # ── 12. Train / Test Split Details ───────────────────────────────────────
    # The stratify parameter guarantees equal class representation in both
    # subsets: 10 samples per species in the test set (30 total).
    with st.expander("ℹ️ Train / Test Split Details", expanded=False):
        c1, c2 = st.columns(2)
        c1.metric("Training samples", X_train.shape[0])
        c2.metric("Testing  samples", X_test.shape[0])

        st.markdown("**Class distribution — Training set**")
        train_dist = (
            y_train.value_counts()
            .rename(index=SPECIES_MAP)
            .rename("Count")
            .reset_index()
            .rename(columns={"index": "Species"})
        )
        st.dataframe(train_dist, use_container_width=True)

    st.divider()

    # ── 2.3 Performance Comparison Table ─────────────────────────────────────
    # Accuracy, Precision, Recall and F1-Score consolidated for all models.
    # Green highlights the best value per metric; red the lowest.
    st.subheader("📊 Performance Comparison Table")
    df_comp = pd.DataFrame(resultados).T.drop(columns=["Matriz", "Modelo"])
    df_comp = df_comp.astype(float)

    st.dataframe(
        df_comp.style
            .format("{:.4f}")
            .highlight_max(axis=0, color="#d4edda")
            .highlight_min(axis=0, color="#f8d7da"),
        use_container_width=True,
    )
    st.caption("🟢 Green = best value per column  |  🔴 Red = lowest value per column")

    st.divider()

    # ── Metrics Bar Chart ─────────────────────────────────────────────────────
    # Interactive selector allows comparing any single metric across all models.
    st.subheader("📈 Metrics Bar Chart")
    metric_sel = st.selectbox(
        "Select metric to compare:",
        ["Accuracy", "Precision", "Recall", "F1-Score"],
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    valores = [resultados[m][metric_sel] for m in resultados]
    colores = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars    = ax.bar(list(resultados.keys()), valores, color=colores, edgecolor="white")
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel(metric_sel)
    ax.set_title(f"{metric_sel} — All Models")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
    plt.xticks(rotation=15, ha="right")
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # ── 2.2 Confusion Matrices ────────────────────────────────────────────────
    # The confusion matrix maps specific successes and misclassification errors
    # for each individual classifier on the 30-sample test set.
    st.subheader("🔲 Confusion Matrices")
    st.markdown(
        "Select a model below to inspect how it classified each species. "
        "Diagonal cells = correct predictions; off-diagonal = errors."
    )
    modelo_sel = st.selectbox("Select model:", list(resultados.keys()))

    cm = resultados[modelo_sel]["Matriz"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Setosa", "Versicolor", "Virginica"],
        yticklabels=["Setosa", "Versicolor", "Virginica"],
    )
    ax.set_title(f"Confusion Matrix — {modelo_sel}")
    ax.set_ylabel("True Class")
    ax.set_xlabel("Predicted Class")
    st.pyplot(fig)
    plt.close(fig)

    # Per-model metric cards
    stats = resultados[modelo_sel]
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Accuracy",  stats["Accuracy"])
    s2.metric("Precision", stats["Precision"])
    s3.metric("Recall",    stats["Recall"])
    s4.metric("F1-Score",  stats["F1-Score"])

    st.divider()

    # ── 2.4 Conclusions & Model Selection (Ockham's Razor) ───────────────────
    # Given a perfect technical tie between Logistic Regression, KNN and
    # Random Forest, parsimony (Ockham's Razor) selects the simplest model.
    with st.expander("✅ Conclusions & Model Selection (Ockham's Razor)", expanded=True):
        st.markdown(
            """
**Results Analysis**

- **Logistic Regression, KNN and Random Forest** achieved perfect performance
  (**1.0000 / 100 %**) across all evaluated metrics.
- **Decision Tree** was the only model exhibiting a margin of error (**0.9667**),
  misclassifying one actual *Versicolor* sample as *Virginica* — a consequence
  of the rigid orthogonal boundaries of a single isolated tree in subtle
  data-overlapping zones.

---

### ✅ Selected Model: Logistic Regression

Applying the principle of parsimony (*Ockham's Razor*): when predictive
capabilities are equal, the alternative with the **lowest structural and
algorithmic complexity** must always be prioritized.

| Criterion | Advantage |
|-----------|-----------|
| **Interpretability** | Coefficients reveal direct feature importance (sepal / petal weights) |
| **Efficiency** | Minimal memory & CPU vs. Random Forest (100 trees) or KNN (full-distance per query) |
| **Robustness** | Linear nature reduces overfitting risk on unseen data |
            """
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PREDICT NEW SAMPLE
# ══════════════════════════════════════════════════════════════════════════════
# Interactive panel: the user adjusts four sliders (sepal / petal measurements)
# and all four trained models return an instant species prediction together with
# the class probability breakdown.

elif seccion == "🔮 Predict New Sample":

    st.title("🔮 Predict a New Sample")
    st.markdown(
        "Adjust the flower measurements below and get an **instant species "
        "prediction** from all four trained classifiers."
    )

    # ── Input sliders ─────────────────────────────────────────────────────────
    # Ranges match the observed min/max values in the Iris dataset so the
    # StandardScaler produces meaningful standardized values.
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
        sepal_width  = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)
    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
        petal_width  = st.slider("Petal Width (cm)",  0.1, 2.5, 1.2, 0.1)

    # ── Standardize input using the fitted scaler ─────────────────────────────
    # The same StandardScaler fitted on X_train is reused here to guarantee
    # that the new sample is transformed with identical parameters.
    sample    = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    sample_sc = scaler.transform(sample)

    st.divider()

    # ── Predictions by Model ──────────────────────────────────────────────────
    # Each model returns the predicted class and its class probabilities.
    # A progress bar visualises the probability for each species.
    st.subheader("Predictions by Model")

    cols = st.columns(len(resultados))
    for col, (nombre, datos) in zip(cols, resultados.items()):
        pred    = datos["Modelo"].predict(sample_sc)[0]
        proba   = (
            datos["Modelo"].predict_proba(sample_sc)[0]
            if hasattr(datos["Modelo"], "predict_proba")
            else None
        )
        especie = SPECIES_MAP[pred]
        emoji   = {"Setosa": "🔵", "Versicolor": "🟠", "Virginica": "🟢"}[especie]
        col.metric(label=nombre, value=f"{emoji} {especie}")
        if proba is not None:
            for i, p in enumerate(proba):
                col.progress(float(p), text=f"{SPECIES_MAP[i]}: {p:.1%}")

    st.divider()

    # ── Input Summary Table ───────────────────────────────────────────────────
    st.subheader("Input Summary")
    input_df = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ],
    )
    st.dataframe(input_df, use_container_width=True)
