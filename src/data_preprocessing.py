"""Data loading, cleaning, feature engineering, and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "price"
CURRENT_YEAR = 2026

NUMERICAL_FEATURES = [
    "year",
    "mileage",
    "engine_size",
    "horsepower",
    "car_age",
    "power_to_engine_ratio",
    "mileage_per_year",
]

CATEGORICAL_FEATURES = ["brand", "fuel_type", "transmission"]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def generate_sample_dataset(csv_path: str | Path, n_rows: int = 900) -> pd.DataFrame:
    """Create a realistic car price CSV for portfolio demonstration."""
    rng = np.random.default_rng(42)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    brand_profile = {
        "Toyota": (14500, 0.88),
        "Honda": (14000, 0.86),
        "Hyundai": (11500, 0.80),
        "Ford": (13000, 0.75),
        "BMW": (30000, 0.70),
        "Mercedes-Benz": (34000, 0.72),
        "Audi": (32000, 0.70),
        "Tesla": (39000, 0.82),
        "Kia": (12000, 0.78),
        "Nissan": (12500, 0.76),
        "Volkswagen": (15500, 0.77),
        "Chevrolet": (13200, 0.73),
    }

    brands = np.array(list(brand_profile.keys()))
    fuel_types = np.array(["Petrol", "Diesel", "Hybrid", "Electric"])
    transmissions = np.array(["Manual", "Automatic"])

    rows = []
    for _ in range(n_rows):
        brand = rng.choice(brands)
        base_price, retention = brand_profile[brand]
        year = int(rng.integers(2011, 2025))
        car_age = CURRENT_YEAR - year

        fuel_type = rng.choice(fuel_types, p=[0.43, 0.27, 0.18, 0.12])
        transmission = rng.choice(transmissions, p=[0.38, 0.62])
        if fuel_type == "Electric":
            engine_size = 0.0
            horsepower = int(rng.normal(310, 65))
        else:
            engine_size = round(float(rng.uniform(1.0, 5.0)), 1)
            horsepower = int(70 + engine_size * rng.normal(62, 11))

        annual_mileage = max(4500, rng.normal(11800, 4200))
        mileage = int(max(2500, annual_mileage * car_age + rng.normal(0, 9000)))

        fuel_adjustment = {
            "Petrol": 0,
            "Diesel": 1800,
            "Hybrid": 4200,
            "Electric": 7200,
        }[fuel_type]
        transmission_adjustment = 2200 if transmission == "Automatic" else 0
        performance_adjustment = horsepower * 68 + engine_size * 1150
        depreciation = (car_age * 1850) + (mileage * 0.055)
        brand_retention = retention * 2600
        noise = rng.normal(0, 2600)

        price = (
            base_price
            + fuel_adjustment
            + transmission_adjustment
            + performance_adjustment
            + brand_retention
            - depreciation
            + noise
        )
        price = int(max(3500, price))

        rows.append(
            {
                "brand": brand,
                "year": year,
                "mileage": mileage,
                "fuel_type": fuel_type,
                "transmission": transmission,
                "engine_size": engine_size,
                "horsepower": horsepower,
                "price": price,
            }
        )

    df = pd.DataFrame(rows)

    # Add a small amount of realistic messiness for preprocessing demonstration.
    duplicate_rows = df.sample(12, random_state=7)
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    for column in ["mileage", "engine_size", "horsepower", "fuel_type"]:
        missing_indices = df.sample(frac=0.018, random_state=len(column)).index
        df.loc[missing_indices, column] = np.nan

    df.to_csv(csv_path, index=False)
    return df


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the dataset, generating a demo dataset if the CSV is missing."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return generate_sample_dataset(csv_path)
    return pd.read_csv(csv_path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names, remove invalid records, and handle duplicates."""
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip().str.lower().str.replace(" ", "_")
    cleaned = cleaned.drop_duplicates()

    text_columns = ["brand", "fuel_type", "transmission"]
    for column in text_columns:
        cleaned[column] = cleaned[column].astype("string").str.strip().str.title()
        cleaned[column] = cleaned[column].replace({pd.NA: np.nan}).astype(object)

    numeric_columns = ["year", "mileage", "engine_size", "horsepower", TARGET_COLUMN]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=[TARGET_COLUMN, "brand", "year"])
    cleaned = cleaned[(cleaned["year"] >= 1995) & (cleaned["year"] <= CURRENT_YEAR)]
    cleaned = cleaned[cleaned[TARGET_COLUMN] > 0]
    cleaned = cleaned[cleaned["mileage"].isna() | (cleaned["mileage"] >= 0)]
    cleaned = cleaned[cleaned["horsepower"].isna() | (cleaned["horsepower"] > 0)]
    cleaned = cleaned[cleaned["engine_size"].isna() | (cleaned["engine_size"] >= 0)]
    cleaned = cleaned.replace({pd.NA: np.nan})

    return cleaned.reset_index(drop=True)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-driven features that improve model signal."""
    featured = df.copy()
    featured["car_age"] = (CURRENT_YEAR - featured["year"]).clip(lower=0)
    featured["power_to_engine_ratio"] = np.where(
        featured["engine_size"].fillna(0) > 0,
        featured["horsepower"] / featured["engine_size"],
        featured["horsepower"],
    )
    featured["mileage_per_year"] = featured["mileage"] / featured["car_age"].clip(lower=1)
    return featured


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return model-ready X and y dataframes."""
    featured = add_engineered_features(df)
    return featured[FEATURE_COLUMNS], featured[TARGET_COLUMN]


def build_preprocessor() -> ColumnTransformer:
    """Build a reusable sklearn preprocessing transformer."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def prepare_user_input(
    brand: str,
    year: int,
    mileage: float,
    fuel_type: str,
    transmission: str,
    engine_size: float,
    horsepower: float,
) -> pd.DataFrame:
    """Build a single-row dataframe with engineered features for prediction."""
    raw_input = pd.DataFrame(
        [
            {
                "brand": brand,
                "year": year,
                "mileage": mileage,
                "fuel_type": fuel_type,
                "transmission": transmission,
                "engine_size": engine_size,
                "horsepower": horsepower,
            }
        ]
    )
    return add_engineered_features(raw_input)[FEATURE_COLUMNS]
