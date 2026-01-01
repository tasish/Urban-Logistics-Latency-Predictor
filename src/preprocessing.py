import pandas as pd
import numpy as np
from geopy.distance import geodesic
import src.config as config

def clean_data(df):
    """
    Basic data cleaning: drop NaNs, fix data types.
    """
    
    # 0. Rename Columns
    df = df.rename(columns={
        'Restaurant_latitude': 'Pickup_Latitude',
        'Restaurant_longitude': 'Pickup_Longitude',
        'Delivery_location_latitude': 'Drop_Latitude',
        'Delivery_location_longitude': 'Drop_Longitude',
        'Delivery_person_ID': 'Agent_ID',
        'Delivery_person_Age': 'Agent_Age',
        'Delivery_person_Ratings': 'Agent_Rating',
    })

    # 1. Clean all string columns first
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    
    # 2. Replace 'NaN' string with actual numpy nan
    df = df.replace('NaN', float('nan'))
    
    # 3. Drop critical missing values
    subset_cols = ['Agent_Age', 'Agent_Rating', 'Time_Orderd', 'Time_Order_picked', 'Weatherconditions', 'Road_traffic_density', 'City', 'Festival', 'multiple_deliveries']
    df = df.dropna(subset=subset_cols)
    
    # 4. Convert types safely
    df['Agent_Age'] = df['Agent_Age'].astype(float).astype(int)
    df['Agent_Rating'] = df['Agent_Rating'].astype(float)
    df['multiple_deliveries'] = df['multiple_deliveries'].astype(str).replace('NaN', '0').astype(float).astype(int)
    
    return df

def calculate_distance(row):
    """
    Calculate Geodesic distance between Pickup and Drop coordinates.
    """
    try:
        coords_1 = (row['Pickup_Latitude'], row['Pickup_Longitude'])
        coords_2 = (row['Drop_Latitude'], row['Drop_Longitude'])
        return geodesic(coords_1, coords_2).km
    except:
        return 0

def calculate_prep_time(row):
    """
    Calculate Order Preparation Time (Picked - Ordered) in minutes.
    """
    try:
        # Handling the time format requires care as dates are separate
        # Assuming same day for simplicity unless it crosses midnight
        t1 = pd.to_datetime(row['Time_Orderd'], format='%H:%M:%S').time()
        t2 = pd.to_datetime(row['Time_Order_picked'], format='%H:%M:%S').time()
        
        # Convert to minutes
        t1_min = t1.hour * 60 + t1.minute
        t2_min = t2.hour * 60 + t2.minute
        
        diff = t2_min - t1_min
        if diff < 0: return diff + 1440 # Next day
        return diff
    except:
        return 15 # Default average

def feature_engineering(df):
    """
    Apply all feature engineering steps.
    """
    # 1. Distance
    df['Distance_km'] = df.apply(calculate_distance, axis=1)
    
    # 2. Prep Time
    df['Prep_Time'] = df.apply(calculate_prep_time, axis=1)
    
    # 3. Encoding
    df['Weather_Encoded'] = df['Weatherconditions'].map(config.WEATHER_MAP)
    df['Traffic_Encoded'] = df['Road_traffic_density'].map(config.TRAFFIC_MAP)
    df['Vehicle_Encoded'] = df['Type_of_vehicle'].map(config.VEHICLE_MAP)
    df['Festival_Encoded'] = df['Festival'].map(config.FESTIVAL_MAP)
    df['City_Encoded'] = df['City'].map(config.CITY_MAP)
    
    # 4. Interaction Feature (Traffic Adjusted Distance)
    # This addresses the user's concern about distance conceptualization
    df['Traffic_Factor'] = df['Road_traffic_density'].map(config.TRAFFIC_FACTOR_MAP)
    df['Traffic_Adj_Dist'] = df['Distance_km'] * df['Traffic_Factor']
    
    # 5. Extract Target if exists
    if 'Time_taken(min)' in df.columns:
        df['Time_taken'] = df['Time_taken(min)'].astype(str).str.extract('(\d+)').astype(int)
        
    return df

def get_final_features(df):
    """
    Select and order features for the model.
    """
    feature_cols = [
        'Agent_Age',
        'Agent_Rating',
        'Weather_Encoded',
        'Traffic_Encoded',
        'Vehicle_condition',
        'Vehicle_Encoded',
        'multiple_deliveries',
        'Festival_Encoded',
        'City_Encoded',
        'Prep_Time',
        'Distance_km',
        'Traffic_Adj_Dist'
    ]
    return df[feature_cols]
