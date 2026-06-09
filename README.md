<<<<<<< HEAD
# Car Price Prediction with Machine Learning

An industry-style machine learning project that predicts used car prices from brand, year, mileage, fuel type, transmission, engine size, and horsepower. The project includes advanced exploratory data analysis, feature engineering, multiple regression models, model comparison, feature importance analysis, a saved production pipeline, and a polished Streamlit application.

## Resume-Worthy Project Description

Built an end-to-end car price intelligence system using Python, Pandas, Scikit-learn, Matplotlib, Seaborn, and Streamlit. The solution cleans noisy vehicle data, engineers depreciation and performance features, compares four regression algorithms, selects the best-performing model using holdout evaluation, and deploys an interactive Streamlit dashboard with prediction reports and explainability visuals.

## Project Objective

The objective is to estimate realistic car resale prices using machine learning and provide a professional workflow similar to a real analytics product:

- Clean and preprocess a CSV-based vehicle dataset.
- Analyze pricing trends across brand, mileage, year, fuel type, and performance.
- Train and compare multiple regression models.
- Save the best model as a reusable prediction pipeline.
- Deploy a user-friendly Streamlit app for price prediction and model interpretation.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Joblib / Pickle-compatible model persistence
- Jupyter Notebook

## Project Structure

```text
Car-Price-Prediction/
|
|-- dataset/
|   `-- car_data.csv
|
|-- notebooks/
|   `-- Car_Price_Prediction.ipynb
|
|-- src/
|   |-- data_preprocessing.py
|   |-- train_model.py
|   `-- predict.py
|
|-- models/
|   `-- best_model.pkl
|
|-- reports/
|   |-- graphs/
|   |-- evaluation_results.txt
|   `-- model_metrics.csv
|
|-- app.py
|-- requirements.txt
|-- README.md
`-- screenshots/
```

## Workflow

1. Load the car price CSV dataset.
2. Remove duplicates and clean invalid values.
3. Handle missing numerical and categorical values through a Scikit-learn pipeline.
4. Engineer domain-specific features:
   - `car_age`
   - `power_to_engine_ratio`
   - `mileage_per_year`
5. Perform advanced EDA:
   - Statistical summary
   - Correlation heatmap
   - Price distribution
   - Brand price comparison
   - Mileage-price relationship
   - Fuel and transmission price spread
6. Train and compare:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
7. Evaluate using:
   - R2 Score
   - MAE
   - MSE
   - RMSE
8. Save the best model pipeline to `models/best_model.pkl`.
9. Launch the Streamlit app for prediction and visualization.

## Installation Steps

```bash
git clone <your-repository-url>
cd Car-Price-Prediction
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train the Model

```bash
python src/train_model.py
```

This command creates:

- `dataset/car_data.csv` if the dataset does not already exist.
- `models/best_model.pkl`
- `reports/evaluation_results.txt`
- `reports/model_metrics.csv`
- EDA and model charts inside `reports/graphs/`

## Run the Streamlit App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Prediction Inputs

The application accepts:

- Car Brand
- Year
- Mileage
- Fuel Type
- Transmission
- Engine Size
- Horsepower

The output is a predicted car price and a downloadable prediction report.

## Results

The training pipeline compares all models and automatically selects the best one based on R2 Score, with MAE and RMSE used as supporting metrics. In most runs on the included dataset, ensemble models such as Random Forest or Gradient Boosting perform best because they capture nonlinear depreciation, brand premiums, mileage effects, and performance interactions.

Detailed results are available in:

```text
reports/evaluation_results.txt
reports/model_metrics.csv
```

## Key Features

- Complete end-to-end ML workflow
- Robust data preprocessing
- Missing value handling
- Duplicate removal
- Advanced EDA charts
- Derived feature engineering
- Four model comparison
- Feature importance visualization
- Saved production-ready pipeline
- Professional Streamlit dashboard
- Sidebar-driven prediction interface
- Downloadable prediction report
- GitHub-ready documentation

## Future Improvements

- Use a larger real-world marketplace dataset.
- Add location, ownership history, accident history, and service records.
- Track experiments with MLflow.
- Add model monitoring for prediction drift.
- Deploy the app on Streamlit Community Cloud or Hugging Face Spaces.
- Add automated unit tests and CI checks.

## Author

Final-year engineering portfolio project for machine learning internship submission.
=======
# Car-Price-Prediction-Project-Machine-Learning-
A machine learning project to predict car prices based on multiple features such as brand, model, year, mileage, fuel type, transmission, and engine specifications. The project demonstrates end‑to‑end workflow including data preprocessing, feature engineering, model training, evaluation, and deployment‑ready scripts.
>>>>>>> 3d63632613b7eb422db1232f6913d4938ca3111a
