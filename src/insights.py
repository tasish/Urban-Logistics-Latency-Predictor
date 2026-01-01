import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import os
from folium.plugins import HeatMap
import src.config as config
import src.preprocessing as preprocessing
import warnings

warnings.filterwarnings('ignore')

def generate_hotspot_map(df):
    print("Generating Delivery Hotspot Map...")
    
    # Filter for valid lat/lon
    df_map = df[(df['Pickup_Latitude'] != 0) & (df['Pickup_Longitude'] != 0)]
    
    # Focus on the most busy city for clarity
    top_city = df_map['City'].mode()[0]
    df_city = df_map[df_map['City'] == top_city].sample(2000, random_state=42) # Sample for performance
    
    # Center map
    center_lat = df_city['Pickup_Latitude'].mean()
    center_lon = df_city['Pickup_Longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='CartoDB dark_matter')
    
    # Heatmap
    HeatMap(data=df_city[['Pickup_Latitude', 'Pickup_Longitude']], radius=10).add_to(m)
    
    if not os.path.exists('assets'): os.makedirs('assets')
    m.save('assets/delivery_hotspots.html')
    print("Saved 'assets/delivery_hotspots.html' (Open this in browser)")

def analyze_efficiency(df):
    print("Generating Efficiency Charts...")
    
    # 1. Prep Time Analysis
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Prep_Time'], bins=30, kde=True, color='orange')
    plt.title('Order Preparation Time Distribution')
    plt.xlabel('Prep Time (min)')
    if not os.path.exists('assets'): os.makedirs('assets')
    plt.savefig('assets/prep_time_dist.png')
    
    # 2. Age vs Performance
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=pd.cut(df['Agent_Age'], bins=[15, 25, 35, 50], labels=['18-25', '26-35', '36-50']), 
                y=df['Time_taken'])
    plt.title('Delivery Time by Agent Age Group')
    plt.savefig('assets/performance_by_age.png')
    
    print("Saved 'assets/prep_time_dist.png' and 'assets/performance_by_age.png'")

def run_insights():
    print("Starting Logic & Insights Analysis...")
    
    # Load & Preprocess
    df = pd.read_csv(config.DATA_PATH)
    df = preprocessing.clean_data(df)
    df = preprocessing.feature_engineering(df)
    
    generate_hotspot_map(df)
    analyze_efficiency(df)
    
    print("Insights Generation Complete!")

if __name__ == "__main__":
    run_insights()
