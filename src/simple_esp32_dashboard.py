import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import threading
import json
import random
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from api.mistral_soil_analysis import MistralSoilAnalysis
from utils.soil_health_interface import SoilHealthPredictor
from models.plant_recommendation import PlantRecommendationSystem

# Set page configuration
st.set_page_config(page_title="Next-Gen Farming Dashboard", layout="wide", initial_sidebar_state="expanded")

# Initialize session state variables
def init_session_state():
    if "connected" not in st.session_state:
        st.session_state.connected = False
        
    if "esp32_url" not in st.session_state:
        st.session_state.esp32_url = ""
        
    if "current_data" not in st.session_state:
        st.session_state.current_data = {}
        
    if "connection_thread" not in st.session_state:
        st.session_state.connection_thread = None
        
    if "stop_connection_thread" not in st.session_state:
        st.session_state.stop_connection_thread = False
        
    if "connection_error_count" not in st.session_state:
        st.session_state.connection_error_count = 0
        
    if "historical_data" not in st.session_state:
        st.session_state.historical_data = []
        
    if "fertility_prediction" not in st.session_state:
        st.session_state.fertility_prediction = None
        
    if "crop_recommendations" not in st.session_state:
        st.session_state.crop_recommendations = []
        
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
        
    if "mistral_api_key" not in st.session_state:
        st.session_state.mistral_api_key = ""
        
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "dashboard"

# Initialize session state
init_session_state()

# Function to fetch data from ESP32
def fetch_data(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raise error if bad status
        data = response.json()
        
        # Add timestamp
        data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Reset error count on successful fetch
        st.session_state.connection_error_count = 0
        
        # Add to historical data for plotting
        if len(st.session_state.historical_data) >= 100:
            st.session_state.historical_data.pop(0)  # Remove oldest data point if we have too many
        st.session_state.historical_data.append(data)
        
        return data, None
    except Exception as e:
        # Increment error count
        st.session_state.connection_error_count += 1
        return None, str(e)

# Function to continuously monitor connection and fetch data
def connection_monitor_thread():
    while not st.session_state.stop_connection_thread:
        if st.session_state.connected and st.session_state.esp32_url:
            data, error = fetch_data(st.session_state.esp32_url)
            
            if data:
                st.session_state.current_data = data
            elif error:
                print(f"Connection error: {error}")
                
                # If we've had too many consecutive errors, disconnect
                if st.session_state.connection_error_count >= 5:
                    print("Too many consecutive errors, disconnecting...")
                    st.session_state.connected = False
                    # Force a rerun to update the UI
                    st.rerun()
                    break
        
        # Sleep between requests to avoid overwhelming the ESP32
        time.sleep(2)

# Function to connect to ESP32
def connect_to_esp32(url):
    if not url.strip():
        st.error("⚠️ Please enter a valid URL.")
        return False
    
    # Ensure URL has proper format
    if not url.startswith("http"):
        url = f"http://{url}"
    
    # Remove trailing slashes
    url = url.rstrip('/')
    
    # If URL doesn't end with /readings, add it
    if not url.endswith("/readings"):
        url = f"{url}/readings"
    
    st.session_state.esp32_url = url
    
    # Test connection
    with st.spinner("Connecting to ESP32..."):
        data, error = fetch_data(url)
        
        if data:
            st.session_state.connected = True
            st.session_state.current_data = data
            
            # Start monitoring thread
            st.session_state.stop_connection_thread = False
            thread = threading.Thread(target=connection_monitor_thread)
            thread.daemon = True
            thread.start()
            st.session_state.connection_thread = thread
            
            st.success("✅ Connected to ESP32 successfully!")
            return True
        else:
            st.error(f"❌ Could not connect to ESP32. Error: {error}")
            return False

# Function to disconnect from ESP32
def disconnect_esp32():
    st.session_state.connected = False
    
    # Stop the connection thread if it exists
    if st.session_state.connection_thread:
        st.session_state.stop_connection_thread = True
        st.session_state.connection_thread = None
    
    st.session_state.esp32_url = ""
    st.success("✅ Disconnected from ESP32")

# Fake prediction model functions
def generate_fake_historical_data(days=30):
    """Generate fake historical data for plotting"""
    data = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    current_date = start_date
    while current_date <= end_date:
        # Generate random sensor values with some trend
        day_progress = (current_date - start_date).days / days
        
        # Create some seasonal patterns
        seasonal_factor = np.sin(day_progress * np.pi * 2) * 0.2 + 1
        
        entry = {
            'timestamp': current_date.strftime("%Y-%m-%d %H:%M:%S"),
            'temperature': max(15, min(35, 25 + np.sin(day_progress * np.pi * 4) * 5 * seasonal_factor)),
            'moisture': max(20, min(90, 60 + np.cos(day_progress * np.pi * 2) * 15 * seasonal_factor)),
            'ph': max(5.0, min(8.0, 6.5 + np.sin(day_progress * np.pi * 3) * 0.8)),
            'nitrogen': max(10, min(140, 80 + np.sin(day_progress * np.pi) * 30 * seasonal_factor)),
            'phosphorus': max(5, min(50, 25 + np.cos(day_progress * np.pi * 2) * 10 * seasonal_factor)),
            'potassium': max(20, min(200, 100 + np.sin(day_progress * np.pi * 3) * 40 * seasonal_factor))
        }
        data.append(entry)
        current_date += timedelta(hours=random.randint(3, 8))  # Random intervals
    
    return data

def predict_soil_fertility(soil_data):
    """Fake prediction model for soil fertility"""
    # Extract soil parameters or use defaults
    temperature = soil_data.get('temperature', 25)
    moisture = soil_data.get('moisture', 50)
    ph = soil_data.get('ph', 6.5)
    nitrogen = soil_data.get('nitrogen', 80)
    phosphorus = soil_data.get('phosphorus', 25)
    potassium = soil_data.get('potassium', 100)
    
    # Calculate a simple fertility score (0-100)
    # This is a simplified model for demonstration
    ph_score = 100 - abs(ph - 6.8) * 20  # Optimal pH around 6.8
    moisture_score = 100 - abs(moisture - 60) * 1.5  # Optimal moisture around 60%
    npk_score = (nitrogen / 120 * 100 + phosphorus / 30 * 100 + potassium / 120 * 100) / 3
    
    # Combine scores with weights
    fertility_score = (ph_score * 0.3 + moisture_score * 0.3 + npk_score * 0.4)
    fertility_score = max(0, min(100, fertility_score))  # Clamp between 0-100
    
    # Determine fertility category
    if fertility_score >= 80:
        category = "Excellent"
        color = "#2e7d32"  # Dark green
    elif fertility_score >= 60:
        category = "Good"
        color = "#4caf50"  # Green
    elif fertility_score >= 40:
        category = "Moderate"
        color = "#ff9800"  # Orange
    elif fertility_score >= 20:
        category = "Poor"
        color = "#f44336"  # Red
    else:
        category = "Very Poor"
        color = "#b71c1c"  # Dark red
    
    # Generate recommendations based on deficiencies
    recommendations = []
    if ph < 6.0:
        recommendations.append("Add agricultural lime to increase soil pH")
    elif ph > 7.5:
        recommendations.append("Add sulfur or organic matter to decrease soil pH")
    
    if nitrogen < 60:
        recommendations.append("Apply nitrogen-rich fertilizer or compost")
    
    if phosphorus < 15:
        recommendations.append("Apply phosphate fertilizer or bone meal")
    
    if potassium < 80:
        recommendations.append("Apply potassium-rich fertilizer or wood ash")
    
    if moisture < 40:
        recommendations.append("Increase irrigation frequency")
    elif moisture > 80:
        recommendations.append("Improve drainage or reduce irrigation")
    
    return {
        "score": fertility_score,
        "category": category,
        "color": color,
        "recommendations": recommendations,
        "parameters": {
            "temperature": temperature,
            "moisture": moisture,
            "ph": ph,
            "nitrogen": nitrogen,
            "phosphorus": phosphorus,
            "potassium": potassium
        }
    }

def get_crop_recommendations(data):
    """Get crop recommendations based on soil data"""
    # Load crop data from CSV
    try:
        crop_df = pd.read_csv("crop_recommendations.csv")
    except Exception:
        # Fallback to hardcoded recommendations if CSV can't be loaded
        return [
            {"crop": "Tomato", "score": 85, "description": "Warm-season crop that thrives in well-drained soil", "market_price": 1500},
            {"crop": "Potato", "score": 78, "description": "Cool-season crop that prefers loose, well-drained soil", "market_price": 1200},
            {"crop": "Carrot", "score": 72, "description": "Root vegetable that grows best in loose, sandy soil", "market_price": 1800},
            {"crop": "Cabbage", "score": 65, "description": "Cool-season crop that needs fertile, well-drained soil", "market_price": 1000},
            {"crop": "Onion", "score": 60, "description": "Bulb vegetable that prefers loose, well-drained soil", "market_price": 1300}
        ]
    
    # Extract soil parameters
    soil_params = {
        "temperature": data.get("temperature", data.get("temperature_ds18b20", 25)),
        "moisture": data.get("moisture", 60),
        "ph": data.get("ph", 6.5),
        "nitrogen": data.get("nitrogen", 80),
        "phosphorus": data.get("phosphorus", 25),
        "potassium": data.get("potassium", 100)
    }
    
    # Calculate compatibility scores for each crop
    recommendations = []
    for _, crop in crop_df.iterrows():
        # Calculate score based on how close soil parameters are to crop's optimal range
        temp_score = 100 - min(100, abs(soil_params["temperature"] - (crop["temperature_min"] + crop["temperature_max"]) / 2) * 5)
        ph_score = 100 - min(100, abs(soil_params["ph"] - (crop["ph_min"] + crop["ph_max"]) / 2) * 20)
        n_score = 100 - min(100, abs(soil_params["nitrogen"] - crop["n"]) / crop["n"] * 100)
        p_score = 100 - min(100, abs(soil_params["phosphorus"] - crop["p"]) / crop["p"] * 100)
        k_score = 100 - min(100, abs(soil_params["potassium"] - crop["k"]) / crop["k"] * 100)
        
        # Calculate overall score (weighted average)
        overall_score = (temp_score * 0.2 + ph_score * 0.2 + n_score * 0.2 + p_score * 0.2 + k_score * 0.2)
        
        recommendations.append({
            "crop": crop["crop"],
            "score": overall_score,
            "description": crop["description"],
            "market_price": crop["market_price"]
        })
    
    # Sort by score and return top recommendations
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations[:5]

# Main app
st.title("🌱 Next-Gen Farming Dashboard")

# Sidebar for navigation and connection settings
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/farm.png", width=80)
    st.title("Navigation")
    
    # Navigation tabs
    tabs = {
        "dashboard": "📊 Dashboard",
        "analytics": "📈 Analytics",
        "predictions": "🔮 Predictions",
        "recommendations": "🌾 Crop Recommendations",
        "chatbot": "🤖 AI Assistant"
    }
    
    for tab_id, tab_name in tabs.items():
        if st.button(tab_name, key=f"nav_{tab_id}", use_container_width=True):
            st.session_state.active_tab = tab_id
            st.rerun()
    
    st.divider()
    
    # Connection section in sidebar
    st.subheader("📡 ESP32 Connection")
    
    # Show connection status
    if st.session_state.connected:
        st.success(f"✅ Connected")
        st.caption(f"URL: {st.session_state.esp32_url}")
        
        # Disconnect button
        if st.button("Disconnect", use_container_width=True):
            disconnect_esp32()
            st.rerun()
    else:
        # Input for ESP32 API URL
        url = st.text_input("ESP32 API URL", "http://10.90.0.244/readings")
        
        # Connect button
        if st.button("Connect", use_container_width=True):
            if connect_to_esp32(url):
                # Generate some fake historical data for demo purposes
                if not st.session_state.historical_data:
                    st.session_state.historical_data = generate_fake_historical_data()
                st.rerun()
    
    # Mistral API key input (for chatbot)
    if st.session_state.active_tab == "chatbot":
        st.divider()
        st.subheader("🔑 Mistral AI API")
        api_key = st.text_input("API Key", type="password", value=st.session_state.mistral_api_key)
        if api_key != st.session_state.mistral_api_key:
            st.session_state.mistral_api_key = api_key
            st.success("API key updated")
    
    st.divider()
    st.caption("© 2023 Next-Gen Farming")
    st.caption("Version 1.0.0")

# Main content area based on active tab
if st.session_state.active_tab == "dashboard":
    # Dashboard tab content
    st.header("📊 Soil Sensor Dashboard")
    
    if st.session_state.connected and st.session_state.current_data:
        # Display current data
        data = st.session_state.current_data
        
        # Create columns for sensor readings with improved styling
        col1, col2, col3 = st.columns(3)
        
        # Display temperature if available with improved styling
        if 'temperature' in data or 'temperature_ds18b20' in data:
            temp = data.get('temperature', data.get('temperature_ds18b20', 0))
            col1.metric(
                "Temperature", 
                f"{temp:.1f} °C", 
                delta=f"{temp - 25:.1f}" if 'temperature' in data else None,
                delta_color="inverse"
            )
        
        # Display moisture if available
        if 'moisture' in data:
            moisture = data['moisture']
            col2.metric(
                "Moisture", 
                f"{moisture:.1f}%", 
                delta=f"{moisture - 60:.1f}" if 'moisture' in data else None,
                delta_color="off" if 40 <= moisture <= 70 else "inverse"
            )
        
        # Display pH if available
        if 'ph' in data:
            ph = data['ph']
            col3.metric(
                "pH", 
                f"{ph:.1f}", 
                delta=f"{ph - 6.5:.1f}" if 'ph' in data else None,
                delta_color="off" if 6.0 <= ph <= 7.0 else "inverse"
            )
        
        # Create another row for additional sensors
        if 'nitrogen' in data or 'phosphorus' in data or 'potassium' in data:
            col1, col2, col3 = st.columns(3)
            
            if 'nitrogen' in data:
                nitrogen = data['nitrogen']
                col1.metric(
                    "Nitrogen", 
                    f"{nitrogen:.1f} mg/kg",
                    delta=f"{nitrogen - 80:.1f}" if 'nitrogen' in data else None,
                    delta_color="normal" if nitrogen >= 80 else "inverse"
                )
            
            if 'phosphorus' in data:
                phosphorus = data['phosphorus']
                col2.metric(
                    "Phosphorus", 
                    f"{phosphorus:.1f} mg/kg",
                    delta=f"{phosphorus - 25:.1f}" if 'phosphorus' in data else None,
                    delta_color="normal" if phosphorus >= 25 else "inverse"
                )
            
            if 'potassium' in data:
                potassium = data['potassium']
                col3.metric(
                    "Potassium", 
                    f"{potassium:.1f} mg/kg",
                    delta=f"{potassium - 100:.1f}" if 'potassium' in data else None,
                    delta_color="normal" if potassium >= 100 else "inverse"
                )
        
        # Soil Fertility Score Card
        st.subheader("🌿 Soil Fertility Analysis")
        
        # Calculate fertility if not already done
        if not st.session_state.fertility_prediction:
            st.session_state.fertility_prediction = predict_soil_fertility(data)
        
        fertility = st.session_state.fertility_prediction
        
        # Display fertility score in a gauge chart (Plot 1: Gauge Chart)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=fertility["score"],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Soil Fertility: {fertility['category']}", 'font': {'size': 24}},
                delta={'reference': 75, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': fertility["color"]},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 20], 'color': '#b71c1c'},  # Very Poor
                        {'range': [20, 40], 'color': '#f44336'},  # Poor
                        {'range': [40, 60], 'color': '#ff9800'},  # Moderate
                        {'range': [60, 80], 'color': '#4caf50'},  # Good
                        {'range': [80, 100], 'color': '#2e7d32'}  # Excellent
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Recommendations")
            for i, rec in enumerate(fertility["recommendations"]):
                st.write(f"🔹 {rec}")
            
            if not fertility["recommendations"]:
                st.write("✅ Your soil parameters are within optimal ranges!")
        
        # Add refresh button
        if st.button("Refresh Data"):
            data, error = fetch_data(st.session_state.esp32_url)
            if data:
                st.session_state.current_data = data
                # Recalculate fertility with new data
                st.session_state.fertility_prediction = predict_soil_fertility(data)
                st.rerun()
        
        # Show raw data in expandable section
        with st.expander("Raw Sensor Data"):
            st.json(data)
        
        # Show in table
        with st.expander("📋 Data Table"):
            df = pd.DataFrame([data])
            st.dataframe(df)
    
    else:
        st.info("📡 Please connect to your ESP32 device to view sensor data.")
        
        # Demo button to show sample data
        if st.button("Load Demo Data"):
            # Generate fake data for demonstration
            fake_data = {
                'temperature': 24.5,
                'moisture': 65.2,
                'ph': 6.8,
                'nitrogen': 85,
                'phosphorus': 30,
                'potassium': 120,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.current_data = fake_data
            st.session_state.historical_data = generate_fake_historical_data()
            st.session_state.fertility_prediction = predict_soil_fertility(fake_data)
            st.session_state.crop_recommendations = get_crop_recommendations(fake_data)
            st.rerun()

elif st.session_state.active_tab == "analytics":
    # Analytics tab content
    st.header("📈 Soil Analytics")
    
    # Check if we have historical data
    if not st.session_state.historical_data:
        st.info("No historical data available. Connect to your ESP32 or load demo data from the Dashboard tab.")
    else:
        # Convert historical data to DataFrame for plotting
        df = pd.DataFrame(st.session_state.historical_data)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time range selector
        time_range = st.selectbox(
            "Select Time Range",
            ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"],
            index=2
        )
        
        # Filter data based on selected time range
        now = datetime.now()
        if time_range == "Last 24 Hours":
            df = df[df['timestamp'] > (now - timedelta(days=1))]
        elif time_range == "Last 7 Days":
            df = df[df['timestamp'] > (now - timedelta(days=7))]
        elif time_range == "Last 30 Days":
            df = df[df['timestamp'] > (now - timedelta(days=30))]
        
        # Plot 2: Time Series Line Chart
        st.subheader("Temperature and Moisture Over Time")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['temperature'],
                name="Temperature (°C)",
                line=dict(color='#FF4B4B', width=2)
            ),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['moisture'],
                name="Moisture (%)",
                line=dict(color='#1E88E5', width=2, dash='dot')
            ),
            secondary_y=True,
        )
        
        fig.update_layout(
            title_text="Temperature and Moisture Trends",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified"
        )
        
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
        fig.update_yaxes(title_text="Moisture (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot 3: NPK Bar Chart with Animation
        st.subheader("Nitrogen, Phosphorus, and Potassium Levels")
        
        # Group by date for daily averages
        df['date'] = df['timestamp'].dt.date
        daily_avg = df.groupby('date').agg({
            'nitrogen': 'mean',
            'phosphorus': 'mean',
            'potassium': 'mean'
        }).reset_index()
        
        # Convert date to string for better display
        daily_avg['date'] = daily_avg['date'].astype(str)
        
        # Melt the dataframe for easier plotting
        melted_df = pd.melt(
            daily_avg,
            id_vars=['date'],
            value_vars=['nitrogen', 'phosphorus', 'potassium'],
            var_name='nutrient',
            value_name='value'
        )
        
        # Create animated bar chart
        fig = px.bar(
            melted_df,
            x='nutrient',
            y='value',
            color='nutrient',
            animation_frame='date',
            range_y=[0, melted_df['value'].max() * 1.2],
            labels={'value': 'Concentration (mg/kg)', 'nutrient': 'Nutrient'},
            color_discrete_map={
                'nitrogen': '#4CAF50',
                'phosphorus': '#FF9800',
                'potassium': '#2196F3'
            }
        )
        
        fig.update_layout(
            title="Daily NPK Levels",
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot 4: Heatmap of Correlations
        st.subheader("Parameter Correlations")
        
        # Calculate correlations
        corr_columns = ['temperature', 'moisture', 'ph', 'nitrogen', 'phosphorus', 'potassium']
        corr_df = df[corr_columns].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.columns,
            colorscale='Viridis',
            zmin=-1, zmax=1,
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Correlation Between Soil Parameters",
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot 5: Box Plot for Parameter Distribution
        st.subheader("Parameter Distributions")
        
        # Select parameter for box plot
        selected_param = st.selectbox(
            "Select Parameter",
            ["temperature", "moisture", "ph", "nitrogen", "phosphorus", "potassium"],
            index=0
        )
        
        # Create box plot
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            y=df[selected_param],
            name=selected_param,
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(color='#1E88E5'),
            line=dict(color='#0D47A1')
        ))
        
        fig.update_layout(
            title=f"{selected_param.capitalize()} Distribution",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis_title=f"{selected_param.capitalize()} Value"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot 6: Scatter Plot Matrix
        st.subheader("Parameter Relationships")
        
        # Create scatter plot matrix
        fig = px.scatter_matrix(
            df,
            dimensions=corr_columns,
            color="moisture",
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        fig.update_layout(
            title="Scatter Plot Matrix of Soil Parameters",
            height=800,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif st.session_state.active_tab == "predictions":
    # Predictions tab content
    st.header("🔮 Soil Fertility Predictions")
    
    if not st.session_state.current_data:
        st.info("No sensor data available. Connect to your ESP32 or load demo data from the Dashboard tab.")
    else:
        # Get current data and fertility prediction
        data = st.session_state.current_data
        
        # Calculate fertility if not already done
        if not st.session_state.fertility_prediction:
            st.session_state.fertility_prediction = predict_soil_fertility(data)
        
        fertility = st.session_state.fertility_prediction
        
        # Display current soil parameters
        st.subheader("Current Soil Parameters")
        
        # Plot 7: Radar Chart of Soil Parameters
        params = fertility["parameters"]
        
        # Normalize values for radar chart
        radar_data = {
            'r': [
                params["ph"] / 14 * 100,  # pH scale is 0-14
                min(100, params["moisture"]),  # Moisture as percentage
                min(100, params["nitrogen"] / 2),  # Normalize nitrogen
                min(100, params["phosphorus"] * 2),  # Normalize phosphorus
                min(100, params["potassium"] / 2),  # Normalize potassium
                min(100, (params["temperature"] / 40) * 100)  # Normalize temperature
            ],
            'theta': ['pH', 'Moisture', 'Nitrogen', 'Phosphorus', 'Potassium', 'Temperature'],
            'fill': 'toself',
            'name': 'Current Parameters'
        }
        
        # Add optimal ranges
        optimal_data = {
            'r': [6.5 / 14 * 100, 60, 80 / 2, 25 * 2, 100 / 2, 25 / 40 * 100],
            'theta': ['pH', 'Moisture', 'Nitrogen', 'Phosphorus', 'Potassium', 'Temperature'],
            'fill': 'toself',
            'name': 'Optimal Range'
        }
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=optimal_data['r'],
            theta=optimal_data['theta'],
            fill='toself',
            name='Optimal Range',
            line=dict(color='rgba(46, 125, 50, 0.5)'),  # Green with transparency
            fillcolor='rgba(46, 125, 50, 0.2)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=radar_data['r'],
            theta=radar_data['theta'],
            fill='toself',
            name='Current Parameters',
            line=dict(color='rgba(33, 150, 243, 0.8)'),  # Blue
            fillcolor='rgba(33, 150, 243, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Soil Parameters vs. Optimal Range",
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot 8: Prediction Confidence Intervals
        st.subheader("Fertility Score Prediction")
        
        # Create a fake confidence interval around the fertility score
        score = fertility["score"]
        lower_bound = max(0, score - random.uniform(5, 15))
        upper_bound = min(100, score + random.uniform(5, 15))
        
        # Create the figure
        fig = go.Figure()
        
        # Add confidence interval as a filled area
        fig.add_trace(go.Scatter(
            x=["Fertility Score"],
            y=[lower_bound],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=["Fertility Score"],
            y=[upper_bound],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(33, 150, 243, 0.3)',
            name='95% Confidence Interval'
        ))
        
        # Add the actual score
        fig.add_trace(go.Scatter(
            x=["Fertility Score"],
            y=[score],
            mode='markers',
            marker=dict(
                color='rgb(33, 150, 243)',
                size=15,
                line=dict(color='rgb(8, 48, 107)', width=2)
            ),
            name='Predicted Score'
        ))
        
        # Update layout
        fig.update_layout(
            title="Soil Fertility Score with Confidence Interval",
            xaxis=dict(title=""),
            yaxis=dict(
                title="Score",
                range=[0, 100],
                tickvals=[0, 20, 40, 60, 80, 100],
                ticktext=["0 (Very Poor)", "20 (Poor)", "40 (Moderate)", "60 (Good)", "80 (Excellent)", "100"]
            ),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot 9: Prediction Timeline (Simulated Future Predictions)
        st.subheader("Fertility Prediction Timeline")
        
        # Generate simulated future predictions
        future_dates = pd.date_range(start=datetime.now(), periods=10, freq='M')
        
        # Create a trend with some randomness
        base_score = score
        future_scores = []
        lower_bounds = []
        upper_bounds = []
        
        for i in range(len(future_dates)):
            # Add some trend and randomness
            trend = np.sin(i / 5 * np.pi) * 10
            random_factor = random.uniform(-5, 5)
            future_score = base_score + trend + random_factor
            future_score = max(0, min(100, future_score))  # Clamp between 0-100
            
            future_scores.append(future_score)
            lower_bounds.append(max(0, future_score - random.uniform(5, 15)))
            upper_bounds.append(min(100, future_score + random.uniform(5, 15)))
        
        # Create the figure
        fig = go.Figure()
        
        # Add confidence intervals
        for i in range(len(future_dates)):
            fig.add_trace(go.Scatter(
                x=[future_dates[i], future_dates[i]],
                y=[lower_bounds[i], upper_bounds[i]],
                mode='lines',
                line=dict(width=0.5, color='rgba(33, 150, 243, 0.3)'),
                showlegend=False
            ))
        
        # Add the predicted scores
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_scores,
            mode='lines+markers',
            name='Predicted Fertility Score',
            line=dict(color='rgb(33, 150, 243)', width=2),
            marker=dict(size=8)
        ))
        
        # Add current score
        fig.add_trace(go.Scatter(
            x=[datetime.now()],
            y=[score],
            mode='markers',
            name='Current Score',
            marker=dict(
                color='rgb(255, 87, 34)',
                size=12,
                line=dict(color='rgb(183, 28, 28)', width=2)
            )
        ))
        
        # Add fertility categories as colored background regions
        fig.add_shape(
            type="rect",
            x0=future_dates[0],
            x1=future_dates[-1],
            y0=0,
            y1=20,
            fillcolor="rgba(183, 28, 28, 0.2)",
            line=dict(width=0),
            layer="below"
        )
        
        fig.add_shape(
            type="rect",
            x0=future_dates[0],
            x1=future_dates[-1],
            y0=20,
            y1=40,
            fillcolor="rgba(244, 67, 54, 0.2)",
            line=dict(width=0),
            layer="below"
        )
        
        fig.add_shape(
            type="rect",
            x0=future_dates[0],
            x1=future_dates[-1],
            y0=40,
            y1=60,
            fillcolor="rgba(255, 152, 0, 0.2)",
            line=dict(width=0),
            layer="below"
        )
        
        fig.add_shape(
            type="rect",
            x0=future_dates[0],
            x1=future_dates[-1],
            y0=60,
            y1=80,
            fillcolor="rgba(76, 175, 80, 0.2)",
            line=dict(width=0),
            layer="below"
        )
        
        fig.add_shape(
            type="rect",
            x0=future_dates[0],
            x1=future_dates[-1],
            y0=80,
            y1=100,
            fillcolor="rgba(46, 125, 50, 0.2)",
            line=dict(width=0),
            layer="below"
        )
        
        # Update layout
        fig.update_layout(
            title="Projected Soil Fertility Over Time",
            xaxis=dict(title="Date"),
            yaxis=dict(
                title="Fertility Score",
                range=[0, 100],
                tickvals=[10, 30, 50, 70, 90],
                ticktext=["Very Poor", "Poor", "Moderate", "Good", "Excellent"]
            ),
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recalculate button
        if st.button("Recalculate Predictions"):
            # Recalculate fertility with current data
            st.session_state.fertility_prediction = predict_soil_fertility(data)
            st.rerun()

elif st.session_state.active_tab == "recommendations":
    # Crop Recommendations tab content
    st.header("🌾 Crop Recommendations")
    
    if not st.session_state.current_data:
        st.info("No sensor data available. Connect to your ESP32 or load demo data from the Dashboard tab.")
    else:
        # Get current data
        data = st.session_state.current_data
        
        # Get crop recommendations if not already calculated
        if not st.session_state.crop_recommendations:
            st.session_state.crop_recommendations = get_crop_recommendations(data)
        
        recommendations = st.session_state.crop_recommendations
        
        # Display recommendations
        st.subheader("Recommended Crops for Your Soil")
        
        # Plot 10: Horizontal Bar Chart for Crop Scores
        crops = [rec["crop"] for rec in recommendations]
        scores = [rec["score"] for rec in recommendations]
        
        # Create color scale based on scores
        colors = [f'rgba(46, 125, 50, {score/100})' for score in scores]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=crops,
            x=scores,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(46, 125, 50, 1.0)', width=1)
            )
        ))
        
        fig.update_layout(
            title="Crop Compatibility Scores",
            xaxis=dict(
                title="Compatibility Score",
                range=[0, 100]
            ),
            yaxis=dict(title=""),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed recommendations
        st.subheader("Detailed Crop Information")
        
        # Create columns for crop cards
        cols = st.columns(3)
        
        for i, rec in enumerate(recommendations):
            with cols[i % 3]:
                st.markdown(f"""<div style='background-color: rgba(46, 125, 50, 0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
                <h3 style='color: #2E7D32;'>{rec['crop']}</h3>
                <p><b>Compatibility Score:</b> {rec['score']:.1f}%</p>
                <p><b>Description:</b> {rec['description']}</p>
                <p><b>Market Price:</b> ₹{rec['market_price']}/quintal</p>
                </div>""", unsafe_allow_html=True)
        
        # Recalculate button
        if st.button("Recalculate Recommendations"):
            # Recalculate recommendations with current data
            st.session_state.crop_recommendations = get_crop_recommendations(data)
            st.rerun()

elif st.session_state.active_tab == "chatbot":
    # Chatbot tab content
    st.header("🤖 Mistral AI Soil Assistant")
    
    # Check if API key is set
    if not st.session_state.mistral_api_key:
        st.warning("Please enter your Mistral AI API key in the sidebar to use the chatbot.")
    else:
        # Initialize chat interface
        if "chat_messages" not in st.session_state or not st.session_state.chat_messages:
            # Add initial system message
            st.session_state.chat_messages = [
                {"role": "system", "content": "I am a Soil Health Assistant powered by Mistral AI. I can help you with soil analysis, crop recommendations, and farming best practices."}
            ]
        
        # Display chat messages
        for message in st.session_state.chat_messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your soil health or farming practices..."):
            # Add user message to chat history
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Simulate AI response (in a real implementation, this would call the Mistral API)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # In a real implementation, we would use the MistralSoilAnalysis class here
                    # For now, we'll simulate a response based on the prompt
                    
                    # Get current soil data if available
                    soil_data = st.session_state.current_data if st.session_state.current_data else {}
                    fertility = st.session_state.fertility_prediction if st.session_state.fertility_prediction else {}
                    recommendations = st.session_state.crop_recommendations if st.session_state.crop_recommendations else []
                    
                    # Generate a contextual response based on available data
                    if "soil" in prompt.lower() and "health" in prompt.lower() and fertility:
                        response = f"Based on the current sensor readings, your soil fertility score is {fertility.get('score', 'N/A'):.1f}/100, which is considered {fertility.get('category', 'N/A')}. "
                        if fertility.get('recommendations', []):
                            response += "Here are some recommendations to improve your soil health:\n\n"
                            for rec in fertility.get('recommendations', []):
                                response += f"- {rec}\n"
                    elif "crop" in prompt.lower() and "recommend" in prompt.lower() and recommendations:
                        response = "Based on your current soil parameters, here are my top crop recommendations:\n\n"
                        for i, rec in enumerate(recommendations[:3], 1):
                            response += f"{i}. {rec['crop']} (Compatibility: {rec['score']:.1f}%) - {rec['description']}\n"
                    elif any(word in prompt.lower() for word in ["temperature", "moisture", "ph", "nitrogen", "phosphorus", "potassium"]) and soil_data:
                        response = "Here are your current soil parameters:\n\n"
                        for param in ["temperature", "moisture", "ph", "nitrogen", "phosphorus", "potassium"]:
                            if param in soil_data:
                                unit = "°C" if param == "temperature" else "mg/kg" if param in ["nitrogen", "phosphorus", "potassium"] else "%" if param == "moisture" else ""
                                response += f"- {param.capitalize()}: {soil_data[param]:.1f} {unit}\n"
                    else:
                        response = "I'm your Soil Health Assistant powered by Mistral AI. I can help you analyze your soil data, recommend crops, and provide farming best practices. What would you like to know about your soil or farming operations?"
                    
                    # Display the response
                    st.write(response)
                    
                    # Add assistant message to chat history
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.success("✅ Data refreshed successfully!")
            st.rerun()
        else:
            st.error(f"❌ Could not fetch data. Error: {error}")
else:
    if st.session_state.connected:
        st.info("⏳ Waiting for data from ESP32...")
    else:
        st.warning("⚠️ Not connected to ESP32. Please connect first.")
        
        # Add a demo button
        if st.button("Load Demo Data"):
            # Generate demo data
            demo_data = {
                "temperature": 25.4,
                "moisture": 68.7,
                "ph": 6.8,
                "nitrogen": 120,
                "phosphorus": 8.5,
                "potassium": 170,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.session_state.current_data = demo_data
            st.rerun()