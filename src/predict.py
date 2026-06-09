"""Prediction helpers for the trained car price model."""

from __future__ import annotations

from pathlib import Path

import joblib

try:
    from data_preprocessing import CURRENT_YEAR, prepare_user_input
except ImportError:
    from src.data_preprocessing import CURRENT_YEAR, prepare_user_input


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model artifact does not exist."""


def load_model(model_path: str | Path = MODEL_PATH):
    """Load the saved sklearn pipeline."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise ModelNotFoundError(
            "Model file not found. Run `python src/train_model.py` before prediction."
        )
    return joblib.load(model_path)


def validate_inputs(
    year: int,
    mileage: float,
    engine_size: float,
    horsepower: float,
) -> None:
    """Validate user-provided car specifications."""
    if year < 1995 or year > CURRENT_YEAR:
        raise ValueError(f"Year must be between 1995 and {CURRENT_YEAR}.")
    if mileage < 0:
        raise ValueError("Mileage cannot be negative.")
    if engine_size < 0:
        raise ValueError("Engine size cannot be negative.")
    if horsepower <= 0:
        raise ValueError("Horsepower must be greater than zero.")


def predict_price(
    brand: str,
    year: int,
    mileage: float,
    fuel_type: str,
    transmission: str,
    engine_size: float,
    horsepower: float,
    model_path: str | Path = MODEL_PATH,
) -> float:
    """Predict car price from user inputs."""
    validate_inputs(year, mileage, engine_size, horsepower)
    model = load_model(model_path)
    input_df = prepare_user_input(
        brand=brand,
        year=year,
        mileage=mileage,
        fuel_type=fuel_type,
        transmission=transmission,
        engine_size=engine_size,
        horsepower=horsepower,
    )
    prediction = model.predict(input_df)[0]
    return float(max(0, prediction))
