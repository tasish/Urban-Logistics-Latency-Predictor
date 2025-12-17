# 🚚 Urban Logistics: Delivery Latency Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Model](https://img.shields.io/badge/Model-XGBoost-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
In the on-demand economy, the "Last-Mile" delivery phase is the most expensive and unpredictable part of the supply chain. Inaccurate Estimated Time of Arrival (ETA) leads to customer dissatisfaction and inefficient fleet management.

This project deploys a Machine Learning solution to predict delivery time (in minutes) based on:
- **Geospatial Data:** Pickup and Drop coordinates (Distance calculated via Geodesic formula).
- **Temporal Data:** Time of day, Weekend vs. Weekday, Month.
- **Operational Metrics:** Delivery Agent Age, Ratings, and Vehicle Type.
- **External Factors:** Weather Conditions and Traffic Density.

## 🛠️ Methodology & Pipeline

[Image of Machine Learning Pipeline]

The project follows a structured Data Science lifecycle:
1.  **Data Cleaning:** Handling "NaN" strings, formatting datetimes, and cleaning categorical features.
2.  **Feature Engineering:**
    * Calculated `Geodesic Distance` (km) from Latitude/Longitude.
    * Extracted `Order_Prepare_Time` (Time between order placement and pickup).
    * Created temporal flags (`is_weekend`, `time_of_day`).
3.  **Preprocessing:** Label Encoding for categorical data and Standard Scaling for numerical features.
4.  **Model Selection:** Compared Linear Regression, Decision Trees, Random Forest, and **XGBoost**.
5.  **Evaluation:** Selected XGBoost as the champion model based on $R^2$ Score and MAE.

## 💻 Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Geospatial Processing:** Geopy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Deployment:** Pickle (Serialization)

## 📊 Key Results
The **XGBoost Regressor** outperformed other algorithms, capturing non-linear relationships effectively.

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **R² Score** | **0.82** | Model explains 82% of the variance in delivery time. |
| **MAE** | **~4.5 min** | Predictions are typically within ±4.5 minutes of actual time. |

> *Note: Metrics may vary slightly based on the random seed during training.*

## 📈 Visual Insights
The project includes diagnostic plots to interpret model behavior:
* **Feature Importance:** Identified that `Distance` and `Agent_Ratings` are the top predictors.
* **Residual Analysis:** Errors follow a normal distribution (Bell Curve), confirming the model is unbiased.

## 🚀 How to Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/Urban-Logistics-Predictor.git](https://github.com/your-username/Urban-Logistics-Predictor.git)
