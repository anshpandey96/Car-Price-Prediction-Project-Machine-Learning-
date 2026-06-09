"""Professional Streamlit app for car price prediction."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_preprocessing import CURRENT_YEAR, clean_dataset, load_dataset
from src.predict import ModelNotFoundError, predict_price


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "dataset" / "car_data.csv"
REPORT_PATH = ROOT_DIR / "reports" / "evaluation_results.txt"
METRICS_PATH = ROOT_DIR / "reports" / "model_metrics.csv"
GRAPH_DIR = ROOT_DIR / "reports" / "graphs"


st.set_page_config(
    page_title="Car Price Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    :root {
        --primary: #0f766e;
        --accent: #f59e0b;
        --ink: #111827;
        --muted: #6b7280;
        --panel: #ffffff;
        --soft: #f8fafc;
    }

    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
        color: var(--ink);
    }

    [data-testid="stSidebar"] {
        background: #0f172a;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] span {
        color: #f8fafc;
    }

    .hero {
        padding: 1.6rem 0 0.7rem 0;
        border-bottom: 1px solid #dbe3ec;
        margin-bottom: 1rem;
    }

    .hero h1 {
        font-size: 2.4rem;
        line-height: 1.1;
        margin: 0;
        letter-spacing: 0;
        color: #0f172a;
    }

    .hero p {
        margin-top: 0.7rem;
        color: var(--muted);
        max-width: 880px;
        font-size: 1rem;
    }

    .metric-card {
        background: var(--panel);
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.1rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .metric-value {
        color: #0f172a;
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: 0.3rem;
    }

    .prediction-box {
        background: #0f766e;
        color: white;
        border-radius: 8px;
        padding: 1.3rem;
        margin-top: 0.7rem;
    }

    .prediction-box h2 {
        color: white;
        margin: 0;
        font-size: 2rem;
    }

    .section-title {
        color: #0f172a;
        margin-top: 1rem;
        font-size: 1.25rem;
        font-weight: 800;
    }

    .stButton > button {
        background: #f59e0b;
        border: 0;
        color: #111827;
        font-weight: 800;
        border-radius: 8px;
        width: 100%;
        padding: 0.75rem 1rem;
    }

    .stDownloadButton > button {
        border-radius: 8px;
        border: 1px solid #0f766e;
        color: #0f766e;
        font-weight: 700;
        width: 100%;
    }
</style>
"""


@st.cache_data
def get_dataset() -> pd.DataFrame:
    """Load cleaned data for the dashboard."""
    return clean_dataset(load_dataset(DATA_PATH))


@st.cache_data
def get_metrics() -> pd.DataFrame:
    """Load model metrics if available."""
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    return pd.DataFrame()


def read_report() -> str:
    """Read evaluation report text."""
    if REPORT_PATH.exists():
        return REPORT_PATH.read_text(encoding="utf-8")
    return "Evaluation report not found. Run `python src/train_model.py` first."


def render_graph(image_name: str, caption: str) -> None:
    """Render a generated graph with graceful fallback."""
    image_path = GRAPH_DIR / image_name
    if image_path.exists():
        st.image(str(image_path), caption=caption, use_container_width=True)
    else:
        st.info(f"{caption} will appear after running the training pipeline.")


def build_prediction_report(prediction: float, inputs: dict[str, object]) -> str:
    """Create a downloadable prediction report."""
    lines = [
        "CAR PRICE PREDICTION REPORT",
        "=" * 42,
        f"Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Input Details:",
    ]
    lines.extend(f"- {key}: {value}" for key, value in inputs.items())
    lines.extend(
        [
            "",
            f"Predicted Price: ${prediction:,.2f}",
            "",
            "Note: This estimate is produced by a machine learning model trained on",
            "historical vehicle attributes. Final market value may vary based on",
            "location, accident history, service records, ownership, and demand.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Run the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    df = get_dataset()
    metrics = get_metrics()

    st.markdown(
        """
        <div class="hero">
            <h1>Car Price Intelligence Dashboard</h1>
            <p>
                A production-style machine learning project for estimating used car
                prices using brand, depreciation, mileage, performance, fuel type,
                and transmission patterns.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    brand_options = sorted(df["brand"].dropna().unique().tolist())
    fuel_options = sorted(df["fuel_type"].dropna().unique().tolist())
    transmission_options = sorted(df["transmission"].dropna().unique().tolist())

    with st.sidebar:
        st.header("Vehicle Inputs")
        brand = st.selectbox("Car Brand", brand_options, index=brand_options.index("Toyota"))
        year = st.slider("Manufacturing Year", 1995, CURRENT_YEAR, 2020)
        mileage = st.number_input("Mileage", min_value=0, value=42000, step=1000)
        fuel_type = st.selectbox("Fuel Type", fuel_options)
        transmission = st.selectbox("Transmission", transmission_options)
        engine_size = st.number_input("Engine Size (L)", min_value=0.0, value=2.0, step=0.1)
        horsepower = st.number_input("Horsepower", min_value=1, value=160, step=5)
        predict_button = st.button("Predict Price")

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Records</div>"
            f"<div class='metric-value'>{len(df):,}</div></div>",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Brands</div>"
            f"<div class='metric-value'>{df['brand'].nunique()}</div></div>",
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Median Price</div>"
            f"<div class='metric-value'>${df['price'].median():,.0f}</div></div>",
            unsafe_allow_html=True,
        )
    with col_d:
        best_r2 = metrics["R2 Score"].max() if not metrics.empty else 0
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Best R2</div>"
            f"<div class='metric-value'>{best_r2:.3f}</div></div>",
            unsafe_allow_html=True,
        )

    left_panel, right_panel = st.columns([0.95, 1.05], gap="large")

    with left_panel:
        st.markdown("<div class='section-title'>Prediction System</div>", unsafe_allow_html=True)
        if predict_button:
            input_payload = {
                "Brand": brand,
                "Year": year,
                "Mileage": mileage,
                "Fuel Type": fuel_type,
                "Transmission": transmission,
                "Engine Size": engine_size,
                "Horsepower": horsepower,
            }
            try:
                prediction = predict_price(
                    brand=brand,
                    year=year,
                    mileage=mileage,
                    fuel_type=fuel_type,
                    transmission=transmission,
                    engine_size=engine_size,
                    horsepower=horsepower,
                )
                st.markdown(
                    f"<div class='prediction-box'><span>Estimated Market Price</span>"
                    f"<h2>${prediction:,.2f}</h2></div>",
                    unsafe_allow_html=True,
                )
                report = build_prediction_report(prediction, input_payload)
                st.download_button(
                    "Download Prediction Report",
                    data=report,
                    file_name="car_price_prediction_report.txt",
                    mime="text/plain",
                )
            except ModelNotFoundError as exc:
                st.error(str(exc))
            except ValueError as exc:
                st.warning(str(exc))
            except Exception as exc:  # pragma: no cover - UI safety net
                st.error(f"Prediction failed because of an unexpected issue: {exc}")
        else:
            st.info("Set vehicle specifications in the sidebar and run prediction.")

        st.markdown("<div class='section-title'>Model Information</div>", unsafe_allow_html=True)
        if not metrics.empty:
            st.dataframe(metrics, use_container_width=True, hide_index=True)
        with st.expander("Read Evaluation Summary"):
            st.text(read_report())

    with right_panel:
        st.markdown("<div class='section-title'>Feature Importance</div>", unsafe_allow_html=True)
        render_graph("feature_importance.png", "Top drivers behind price predictions")

    st.markdown("<div class='section-title'>Advanced EDA and Model Diagnostics</div>", unsafe_allow_html=True)
    tab_eda, tab_models, tab_relationships = st.tabs(
        ["Dataset Insights", "Model Comparison", "Feature Relationships"]
    )

    with tab_eda:
        chart_a, chart_b = st.columns(2)
        with chart_a:
            render_graph("price_distribution.png", "Target price distribution")
        with chart_b:
            render_graph("correlation_heatmap.png", "Correlation heatmap")

    with tab_models:
        chart_c, chart_d = st.columns(2)
        with chart_c:
            render_graph("model_comparison.png", "Regression model comparison")
        with chart_d:
            if not metrics.empty:
                st.bar_chart(metrics.set_index("Model")[["MAE", "RMSE"]])
            else:
                st.info("Metric chart will appear after model training.")

    with tab_relationships:
        chart_e, chart_f = st.columns(2)
        with chart_e:
            render_graph("brand_price_comparison.png", "Median resale value by brand")
        with chart_f:
            render_graph("mileage_price_relationship.png", "Mileage depreciation pattern")
        render_graph("transmission_fuel_price_boxplot.png", "Fuel and transmission price spread")


if __name__ == "__main__":
    main()
