"""Train and compare car price regression models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".matplotlib"))

import joblib
import matplotlib
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from data_preprocessing import (
        CATEGORICAL_FEATURES,
        FEATURE_COLUMNS,
        NUMERICAL_FEATURES,
        TARGET_COLUMN,
        build_preprocessor,
        clean_dataset,
        load_dataset,
        split_features_target,
    )
except ImportError:
    from src.data_preprocessing import (
        CATEGORICAL_FEATURES,
        FEATURE_COLUMNS,
        NUMERICAL_FEATURES,
        TARGET_COLUMN,
        build_preprocessor,
        clean_dataset,
        load_dataset,
        split_features_target,
    )


DATA_PATH = ROOT_DIR / "dataset" / "car_data.csv"
MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
REPORT_PATH = ROOT_DIR / "reports" / "evaluation_results.txt"
GRAPH_DIR = ROOT_DIR / "reports" / "graphs"


def evaluate_model(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Calculate regression metrics for a model."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2 Score": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": mse**0.5,
    }


def get_models() -> Dict[str, object]:
    """Return candidate regression models."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=12,
            min_samples_leaf=6,
            random_state=42,
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=250,
            max_depth=16,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=220,
            learning_rate=0.055,
            max_depth=4,
            random_state=42,
        ),
    }


def create_pipeline(model: object) -> Pipeline:
    """Attach preprocessing to a regression estimator."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def get_transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Return names after one-hot encoding for interpretability charts."""
    preprocessor = pipeline.named_steps["preprocessor"]
    categorical_encoder = preprocessor.named_transformers_["categorical"].named_steps[
        "encoder"
    ]
    categorical_names = categorical_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    return [*NUMERICAL_FEATURES, *categorical_names.tolist()]


def save_eda_graphs(df: pd.DataFrame) -> None:
    """Generate professional EDA visuals."""
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="viridis")

    numeric_df = df.select_dtypes(include="number")
    plt.figure(figsize=(11, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="viridis", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of Numerical Features", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "correlation_heatmap.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(df[TARGET_COLUMN], kde=True, bins=35, color="#2563eb")
    plt.title("Distribution of Car Prices", fontsize=16, weight="bold")
    plt.xlabel("Price")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "price_distribution.png", dpi=220)
    plt.close()

    plt.figure(figsize=(12, 6))
    brand_price = df.groupby("brand")[TARGET_COLUMN].median().sort_values(ascending=False)
    sns.barplot(
        x=brand_price.index,
        y=brand_price.values,
        hue=brand_price.index,
        palette="mako",
        legend=False,
    )
    plt.title("Median Price by Brand", fontsize=16, weight="bold")
    plt.xlabel("Brand")
    plt.ylabel("Median Price")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "brand_price_comparison.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="mileage",
        y=TARGET_COLUMN,
        hue="fuel_type",
        alpha=0.75,
        palette="Set2",
    )
    plt.title("Mileage vs Price by Fuel Type", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "mileage_price_relationship.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="transmission", y=TARGET_COLUMN, hue="fuel_type")
    plt.title("Price Spread by Transmission and Fuel Type", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "transmission_fuel_price_boxplot.png", dpi=220)
    plt.close()


def save_model_comparison(results: pd.DataFrame) -> None:
    """Save a model comparison chart."""
    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=results,
        x="R2 Score",
        y="Model",
        hue="Model",
        palette="crest",
        legend=False,
    )
    plt.title("Model Comparison by R2 Score", fontsize=16, weight="bold")
    plt.xlabel("R2 Score")
    plt.ylabel("Model")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "model_comparison.png", dpi=220)
    plt.close()


def save_feature_importance(best_pipeline: Pipeline) -> None:
    """Save feature importance or coefficient impact chart."""
    model = best_pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        label = "Importance"
    elif hasattr(model, "coef_"):
        values = abs(model.coef_)
        label = "Absolute Coefficient"
    else:
        return

    importances = pd.DataFrame(
        {
            "Feature": get_transformed_feature_names(best_pipeline),
            "Importance": values,
        }
    ).sort_values("Importance", ascending=False)

    importances.to_csv(GRAPH_DIR / "feature_importance.csv", index=False)
    plt.figure(figsize=(11, 7))
    sns.barplot(
        data=importances.head(15),
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="flare",
        legend=False,
    )
    plt.xlabel(label)
    plt.title("Top Feature Impacts", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "feature_importance.png", dpi=220)
    plt.close()


def write_evaluation_report(
    df: pd.DataFrame,
    results: pd.DataFrame,
    best_model_name: str,
) -> None:
    """Write a polished text report for internship submission."""
    best_row = results.loc[results["Model"] == best_model_name].iloc[0]
    if best_model_name == "Linear Regression":
        selection_reason = (
            "Selection Reason: Linear Regression achieved the highest R2 Score "
            "and the lowest error values on the holdout test set. The generated "
            "dataset contains strong additive pricing signals such as depreciation, "
            "mileage impact, horsepower contribution, and brand premium, which a "
            "linear model captures efficiently while remaining highly interpretable."
        )
    else:
        selection_reason = (
            "Selection Reason: The best model achieved the highest R2 Score while "
            "maintaining competitive MAE and RMSE on the holdout test set. "
            "Tree-based ensemble models are well suited to car pricing because they "
            "capture nonlinear effects such as depreciation, mileage bands, brand "
            "premiums, fuel technology, and performance differences."
        )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as report:
        report.write("CAR PRICE PREDICTION - MODEL EVALUATION REPORT\n")
        report.write("=" * 62 + "\n\n")
        report.write(f"Dataset rows after cleaning: {len(df)}\n")
        report.write(f"Features used: {', '.join(FEATURE_COLUMNS)}\n")
        report.write(f"Target variable: {TARGET_COLUMN}\n\n")
        report.write("Model Performance:\n")
        report.write(results.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
        report.write("\n\n")
        report.write(f"Selected Best Model: {best_model_name}\n")
        report.write(f"{selection_reason}\n\n")
        report.write(
            "Best Model Metrics:\n"
            f"R2 Score: {best_row['R2 Score']:.4f}\n"
            f"MAE: {best_row['MAE']:,.2f}\n"
            f"MSE: {best_row['MSE']:,.2f}\n"
            f"RMSE: {best_row['RMSE']:,.2f}\n"
        )


def train_and_save() -> Tuple[Pipeline, pd.DataFrame]:
    """Train all candidate models, save reports, and persist the best pipeline."""
    df = clean_dataset(load_dataset(DATA_PATH))
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    results = []
    trained_pipelines: Dict[str, Pipeline] = {}
    for model_name, model in get_models().items():
        pipeline = create_pipeline(model)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = evaluate_model(y_test, predictions)
        results.append({"Model": model_name, **metrics})
        trained_pipelines[model_name] = pipeline

    results_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False)
    best_model_name = results_df.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_model_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    save_eda_graphs(df)
    save_model_comparison(results_df)
    save_feature_importance(best_pipeline)
    write_evaluation_report(df, results_df, best_model_name)
    results_df.to_csv(ROOT_DIR / "reports" / "model_metrics.csv", index=False)

    print("Training completed successfully.")
    print(f"Best model: {best_model_name}")
    print(results_df.to_string(index=False))
    return best_pipeline, results_df


if __name__ == "__main__":
    train_and_save()
