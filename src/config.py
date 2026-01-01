import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "Food_Delivery_Dataset.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Mappings (Label Encoding)
WEATHER_MAP = {
    'conditions Cloudy': 0, 
    'conditions Fog': 1, 
    'conditions Sandstorms': 2, 
    'conditions Stormy': 3, 
    'conditions Sunny': 4, 
    'conditions Windy': 5,
    # Fallback for app input if "conditions " prefix is omitted
    'Cloudy': 0, 'Fog': 1, 'Sandstorms': 2, 'Stormy': 3, 'Sunny': 4, 'Windy': 5
}

TRAFFIC_MAP = {'High': 0, 'Jam': 1, 'Low': 2, 'Medium': 3}
VEHICLE_MAP = {'bicycle': 0, 'electric_scooter': 1, 'motorcycle': 2, 'scooter': 3}
FESTIVAL_MAP = {'No': 0, 'Yes': 1}
CITY_MAP = {'Metropolitian': 0, 'Semi-Urban': 1, 'Urban': 2}

# Traffic Density Factors (Higher value = Slower traffic)
# Used for Feature Interaction: Distance * Traffic_Factor
TRAFFIC_FACTOR_MAP = {
    'Low': 1.0,
    'Medium': 1.2,
    'High': 1.5,
    'Jam': 2.0
}

# Feature Columns (Order matters for Model)
FEATURES = [
    'Agent_Age', 
    'Agent_Rating', 
    'Weatherconditions', 
    'Road_traffic_density', 
    'Vehicle_condition', 
    'Type_of_vehicle', 
    'multiple_deliveries', 
    'Festival', 
    'City', 
    'Prep_Time',     # Engineered
    'Distance_km',   # Engineered
    'Traffic_Adj_Dist' # Engineered Interaction
]
