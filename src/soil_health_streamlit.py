import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import serial
import serial.tools.list_ports
import json
import time
import threading
import os
from datetime import datetime
from collections import deque
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Import our custom modules
import sys
sys.path.append('.')
from soil_health_interface import SoilHealthPredictor, ESP32Interface
from plant_recommendation import PlantRecommendationSystem
from mistral_soil_analysis import MistralSoilAnalysis

# Set page configuration
st.set_page_config(
    page_title="Soil Health Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #388E3C;
        margin-bottom: 0.5rem;
    }
    .card {
        background-color: #F1F8E9;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-label {
        font-size: 1rem;
        color: #689F38;
        font-weight: bold;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
    }
    .fertility-high {
        color: #2E7D32;
        font-weight: bold;
    }
    .fertility-medium {
        color: #FFA000;
        font-weight: bold;
    }
    .fertility-low {
        color: #D32F2F;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load crop data from CSV
@st.cache_data
def load_crop_data():
    try:
        # Load crop recommendations CSV
        crop_recommendations_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop_recommendations.csv")
        df = pd.read_csv(crop_recommendations_path)
        return df
    except Exception as e:
        st.error(f"Error loading crop data: {e}")
        return pd.DataFrame()

# Initialize session state
def init_session_state():
    if 'connected' not in st.session_state:
        st.session_state.connected = False
    if 'esp32_interface' not in st.session_state:
        st.session_state.esp32_interface = None
    if 'soil_predictor' not in st.session_state:
        st.session_state.soil_predictor = SoilHealthPredictor()
    if 'plant_recommender' not in st.session_state:
        crop_recommendations_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop_recommendations.csv")
        st.session_state.plant_recommender = PlantRecommendationSystem(crop_data_path=crop_recommendations_path)
    if 'mistral_analyzer' not in st.session_state:
        st.session_state.mistral_analyzer = MistralSoilAnalysis(api_key="yQdfM8MLbX9uhInQ7id4iUTwN4h4pDLX")
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'fertility_prediction' not in st.session_state:
        st.session_state.fertility_prediction = None
    if 'history_data' not in st.session_state:
        st.session_state.history_data = []
    if 'sensor_data' not in st.session_state:
        st.session_state.sensor_data = {
            'temperature': deque(maxlen=50),
            'moisture': deque(maxlen=50),
            'nitrogen': deque(maxlen=50),
            'phosphorus': deque(maxlen=50),
            'potassium': deque(maxlen=50)
        }
    if 'timestamps' not in st.session_state:
        st.session_state.timestamps = deque(maxlen=50)
    if 'crop_data' not in st.session_state:
        st.session_state.crop_data = load_crop_data()
    if 'simulation_mode' not in st.session_state:
        st.session_state.simulation_mode = False
    if 'simulation_thread' not in st.session_state:
        st.session_state.simulation_thread = None
    if 'stop_simulation' not in st.session_state:
        st.session_state.stop_simulation = False
    if 'llm_analysis' not in st.session_state:
        st.session_state.llm_analysis = None
    if 'is_analyzing' not in st.session_state:
        st.session_state.is_analyzing = False
    # Initialize chat messages for AI Farmer Intelligence
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hello! I'm your AI Farming Assistant. How can I help you with your soil health and farming needs today?"}
        ]
    # Initialize farmer chat input
    if 'farmer_chat_input' not in st.session_state:
        st.session_state.farmer_chat_input = ""
    # Initialize farmer assistant persona
    if 'farmer_assistant' not in st.session_state:
        st.session_state.farmer_assistant = None

# Connect to ESP32
def connect_to_esp32(port=None):
    try:
        # Display connecting message
        with st.spinner(f"Connecting to ESP32{' on port ' + port if port else ''}..."):
            st.session_state.esp32_interface = ESP32Interface(port=port)
            success = st.session_state.esp32_interface.connect()
            
            if success:
                st.session_state.connected = True
                st.success(f"Connected to ESP32 successfully on port {st.session_state.esp32_interface.port}!")
                
                # Start data collection thread
                threading.Thread(target=collect_data, daemon=True).start()
                
                # Automatically start simulation when connected to ESP32
                if not st.session_state.get('simulation_mode', False):
                    start_simulation()
                    st.info("Simulation automatically started with real ESP32 data!")
                
                return True
            else:
                # Provide more specific error messages
                if port:
                    st.error(f"Failed to connect to ESP32 on port {port}. Please check if the device is connected and the firmware is running.")
                    st.info("Make sure the ESP32 is properly connected and the correct firmware is uploaded.")
                else:
                    st.error("ESP32 not found on any available port.")
                    st.info("Please check if the ESP32 is connected to your computer and the firmware is running. You can also try selecting a specific port from the dropdown.")
                
                st.session_state.connected = False
                return False
    except Exception as e:
        st.error(f"Failed to connect to ESP32: {e}")
        st.info("If you don't have an ESP32 connected, you can use the simulation mode to test the application.")
        st.session_state.connected = False
        return False

# Disconnect from ESP32
def disconnect_esp32():
    if st.session_state.esp32_interface:
        st.session_state.esp32_interface.close()
        st.session_state.connected = False
        st.session_state.esp32_interface = None
        st.info("Disconnected from ESP32")

# Generate simulated data
def generate_simulated_data():
    return {
        'temperature': np.random.uniform(20, 30),
        'moisture': np.random.uniform(40, 80),
        'nitrogen': np.random.uniform(100, 200),
        'phosphorus': np.random.uniform(10, 30),
        'potassium': np.random.uniform(200, 400),
        'ph': np.random.uniform(5.5, 7.5)
    }

# Simulation thread function
def simulation_thread_func():
    while not st.session_state.get('stop_simulation', False):
        data = generate_simulated_data()
        process_data(data)
        time.sleep(2)

# Start simulation
def start_simulation():
    # Ensure all required session state variables are initialized
    if 'timestamps' not in st.session_state:
        st.session_state.timestamps = deque(maxlen=50)
    if 'sensor_data' not in st.session_state:
        st.session_state.sensor_data = {
            'temperature': deque(maxlen=50),
            'moisture': deque(maxlen=50),
            'nitrogen': deque(maxlen=50),
            'phosphorus': deque(maxlen=50),
            'potassium': deque(maxlen=50)
        }
    
    st.session_state.simulation_mode = True
    st.session_state.stop_simulation = False
    st.session_state.simulation_thread = threading.Thread(target=simulation_thread_func, daemon=True)
    st.session_state.simulation_thread.start()
    st.success("Simulation started!")

# Stop simulation
def stop_simulation():
    st.session_state.stop_simulation = True
    st.session_state.simulation_mode = False
    if st.session_state.simulation_thread:
        st.session_state.simulation_thread.join(timeout=1)
    st.info("Simulation stopped")

# Collect data from ESP32
def collect_data():
    while st.session_state.connected and st.session_state.esp32_interface:
        try:
            data = st.session_state.esp32_interface.read_data()
            if data:
                process_data(data)
        except Exception as e:
            print(f"Error reading data: {e}")
        time.sleep(1)

# Process received data
def process_data(data):
    # Store data for trends
    timestamp = datetime.now()
    
    # Ensure timestamps and sensor_data are initialized
    if 'timestamps' not in st.session_state:
        st.session_state.timestamps = deque(maxlen=50)
    if 'sensor_data' not in st.session_state:
        st.session_state.sensor_data = {
            'temperature': deque(maxlen=50),
            'moisture': deque(maxlen=50),
            'nitrogen': deque(maxlen=50),
            'phosphorus': deque(maxlen=50),
            'potassium': deque(maxlen=50)
        }
    
    st.session_state.timestamps.append(timestamp)
    for key in st.session_state.sensor_data.keys():
        if key in data:
            st.session_state.sensor_data[key].append(data[key])
    
    # Make fertility prediction
    fertility_prediction = predict_fertility(data)
    
    # Store current data and prediction
    st.session_state.current_data = data
    st.session_state.fertility_prediction = fertility_prediction
    
    # Add to history
    history_entry = {
        'timestamp': timestamp,
        **data,
        'fertility_class': fertility_prediction['fertility_label']
    }
    st.session_state.history_data.append(history_entry)
    
    # Keep history limited to 1000 entries
    if len(st.session_state.history_data) > 1000:
        st.session_state.history_data.pop(0)

# Predict soil fertility
def predict_fertility(data):
    try:
        # Get fertility prediction from soil predictor
        result = st.session_state.soil_predictor.predict_fertility(data)
        
        # Map fertility class to label
        fertility_class = result['fertility_class']
        fertility_labels = ["Less Fertile", "Fertile", "Highly Fertile"]
        fertility_label = fertility_labels[fertility_class] if fertility_class in [0, 1, 2] else "Unknown"
        
        return {
            'fertility_class': fertility_class,
            'fertility_label': fertility_label,
            'confidence': result.get('confidence', 85.0)  # Default confidence if not provided
        }
    except Exception as e:
        print(f"Error predicting fertility: {e}")
        return {
            'fertility_class': 1,  # Default to "Fertile"
            'fertility_label': "Fertile",
            'confidence': 50.0
        }

# Get crop recommendations
def get_crop_recommendations(data, fertility_prediction):
    try:
        # Get recommendations from plant recommendation system
        fertility_class = fertility_prediction['fertility_class']  # Use numeric class instead of label
        
        # Add pH if not present (using a default value for demonstration)
        if 'ph' not in data:
            data['ph'] = 6.5  # Default neutral pH
        
        # Prepare soil data dictionary for recommendation system
        soil_data = {
            'nitrogen': data['nitrogen'],
            'phosphorus': data['phosphorus'],
            'potassium': data['potassium'],
            'ph': data['ph'],
            'temperature': data['temperature']
        }
        
        # Get recommendations
        recommendations = st.session_state.plant_recommender.get_recommendations(soil_data, fertility_class)
        
        return recommendations
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return {
            'crop_recommendations': [],
            'soil_improvement': []
        }

# Get crop price information from CSV data
def get_crop_price_info(crop_name):
    if st.session_state.crop_data.empty:
        return None
    
    # Find the crop in our recommendations CSV
    crop_data = st.session_state.crop_data[st.session_state.crop_data['crop'].str.lower() == crop_name.lower()]
    
    if crop_data.empty:
        return None
    
    # Get market price if available
    if 'market_price' in crop_data.columns:
        market_price = crop_data['market_price'].iloc[0]
        
        # Create a simulated price range (±10%)
        min_price = market_price * 0.9
        max_price = market_price * 1.1
        
        return {
            'avg_min_price': min_price,
            'avg_max_price': max_price,
            'avg_modal_price': market_price,
            'top_markets': {
                'National Market': market_price,
                'Local Market': market_price * 0.95,
                'Export Market': market_price * 1.15
            }
        }
    
    return None

# Create soil health radar chart for comprehensive visualization
def create_soil_health_radar(soil_data):
    # Define the parameters to include in the radar chart
    parameters = ['nitrogen', 'phosphorus', 'potassium', 'ph', 'moisture', 'temperature']
    
    # Filter out parameters that don't exist in soil_data
    available_params = [p for p in parameters if p in soil_data]
    
    if not available_params:
        # Return empty figure with message
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Soil Health Radar",
            annotations=[
                dict(
                    text="No soil data available",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=16)
                )
            ],
            height=400
        )
        return empty_fig
    
    # Get values for available parameters
    values = [soil_data.get(p, 0) for p in available_params]
    
    # Define ideal ranges for each parameter
    ideal_ranges = {
        'nitrogen': (140, 280),   # mg/kg
        'phosphorus': (25, 50),    # mg/kg
        'potassium': (160, 300),   # mg/kg
        'ph': (6.0, 7.5),          # pH scale
        'moisture': (50, 70),      # %
        'temperature': (20, 30)    # °C
    }
    
    # Calculate normalized values (0-1 scale)
    normalized_values = []
    for i, param in enumerate(available_params):
        if param in ideal_ranges:
            min_val, max_val = ideal_ranges[param]
            # Normalize to 0-1 scale where 1 is optimal
            if values[i] < min_val:
                # Below optimal range
                norm_val = values[i] / min_val
            elif values[i] > max_val:
                # Above optimal range (diminishing returns)
                excess = values[i] - max_val
                range_size = max_val - min_val
                norm_val = 1 - min(1, excess / range_size / 2)  # Penalty for excess
            else:
                # Within optimal range
                norm_val = 1.0
            normalized_values.append(norm_val)
        else:
            # If no ideal range is defined, use raw value
            normalized_values.append(values[i] / 100)  # Arbitrary scaling
    
    # Create radar chart
    fig = go.Figure()
    
    # Add trace for current soil values
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=available_params,
        fill='toself',
        name='Current Soil',
        line=dict(color='#4CAF50', width=2),
        fillcolor='rgba(76, 175, 80, 0.3)'
    ))
    
    # Add trace for ideal values (all 1.0)
    fig.add_trace(go.Scatterpolar(
        r=[1.0] * len(available_params),
        theta=available_params,
        fill='toself',
        name='Ideal Range',
        line=dict(color='rgba(255, 255, 255, 0.8)', width=1, dash='dash'),
        fillcolor='rgba(255, 255, 255, 0.1)'
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=False
            )
        ),
        title={
            'text': "<b>Soil Health Balance</b>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18, color='white')
        },
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.1, xanchor='center', x=0.5),
        height=400,
        margin=dict(l=80, r=80, t=80, b=80),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Create enhanced trend graph with more interactive features
def create_trend_graph(data_key, title, color):
    if len(st.session_state.timestamps) == 0 or len(st.session_state.sensor_data[data_key]) == 0:
        # Return empty figure with message
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title=title,
            annotations=[
                dict(
                    text="No data available yet",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=16)
                )
            ],
            height=300
        )
        return empty_fig
    
    # Convert deque to list for plotting
    timestamps = list(st.session_state.timestamps)
    values = list(st.session_state.sensor_data[data_key])
    
    # Calculate statistics for annotations
    avg_value = sum(values) / len(values)
    max_value = max(values)
    min_value = min(values)
    
    # Create figure
    fig = go.Figure()
    
    # Add main data trace with gradient fill
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=values,
        mode='lines+markers',
        name=data_key,
        line=dict(color=color, width=2),
        marker=dict(size=6, color=color, line=dict(width=1, color='white')),
        fill='tozeroy',
        fillcolor=f'rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.2)',
        hovertemplate='<b>Time</b>: %{x}<br><b>' + title + '</b>: %{y:.2f}<extra></extra>'
    ))
    
    # Add average line
    fig.add_trace(go.Scatter(
        x=[timestamps[0], timestamps[-1]],
        y=[avg_value, avg_value],
        mode='lines',
        name='Average',
        line=dict(color='rgba(255,255,255,0.7)', width=1.5, dash='dash'),
        hovertemplate='<b>Average</b>: %{y:.2f}<extra></extra>'
    ))
    
    # Add max and min lines
    fig.add_trace(go.Scatter(
        x=[timestamps[0], timestamps[-1]],
        y=[max_value, max_value],
        mode='lines',
        name='Maximum',
        line=dict(color='rgba(255,200,200,0.5)', width=1, dash='dot'),
        hovertemplate='<b>Maximum</b>: %{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=[timestamps[0], timestamps[-1]],
        y=[min_value, min_value],
        mode='lines',
        name='Minimum',
        line=dict(color='rgba(200,200,255,0.5)', width=1, dash='dot'),
        hovertemplate='<b>Minimum</b>: %{y:.2f}<extra></extra>'
    ))
    
    # Add annotations for statistics
    fig.add_annotation(
        x=timestamps[-1],
        y=avg_value,
        text=f"Avg: {avg_value:.2f}",
        showarrow=False,
        font=dict(size=10, color="white"),
        bgcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.7)",
        bordercolor=color,
        borderwidth=1,
        borderpad=3,
        xshift=50
    )
    
    # Update layout with improved styling
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=16, color='white')
        },
        xaxis={
            'title': 'Time',
            'gridcolor': 'rgba(211,211,211,0.3)',
            'showgrid': True,
            'rangeslider': dict(visible=True, thickness=0.05),
            'rangeselector': dict(
                buttons=list([
                    dict(count=5, label="5m", step="minute", stepmode="backward"),
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(step="all")
                ])
            )
        },
        yaxis={
            'title': 'Value',
            'gridcolor': 'rgba(211,211,211,0.3)',
            'showgrid': True
        },
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        annotations=[
            dict(
                text=f"Max: {max_value:.2f}",
                x=timestamps[values.index(max_value)],
                y=max_value,
                xshift=0,
                yshift=10,
                showarrow=False,
                font=dict(size=10, color=color)
            ),
            dict(
                text=f"Min: {min_value:.2f}",
                x=timestamps[values.index(min_value)],
                y=min_value,
                xshift=0,
                yshift=-15,
                showarrow=False,
                font=dict(size=10, color=color)
            )
        ]
    )
    
    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.05),
            type="date"
        )
    )
    
    return fig

# Main dashboard layout with enhanced UI
def main_dashboard():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #3498db;
    }
    .sub-header {
        color: #2c3e50;
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
        height: 100%;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    .metric-label {
        font-size: 1rem;
        font-weight: 600;
        color: #7f8c8d;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .connect-btn {
        background-color: #27ae60;
        color: white;
    }
    .disconnect-btn {
        background-color: #e74c3c;
        color: white;
    }
    .sim-btn {
        background-color: #3498db;
        color: white;
    }
    .status-indicator {
        display: flex;
        align-items: center;
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 0.5rem;
    }
    .status-dot {
        height: 12px;
        width: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .connected-dot {
        background-color: #27ae60;
    }
    .simulation-dot {
        background-color: #3498db;
    }
    .disconnected-dot {
        background-color: #e74c3c;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with logo
    st.markdown('<h1 class="main-header">🌱 Smart Soil Health Dashboard</h1>', unsafe_allow_html=True)
    
    # Connection controls with improved UI
    st.markdown('<div style="background-color: #f1f8ff; padding: 15px; border-radius: 10px; margin-bottom: 20px;">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if not st.session_state.connected and not st.session_state.simulation_mode:
            # Get available serial ports
            ports = [port.device for port in serial.tools.list_ports.comports()]
            selected_port = st.selectbox("📱 Select ESP32 Port", options=ports, index=0 if ports else None)
            
            # Connect button with selected port and custom styling
            if st.button("🔌 Connect to ESP32", key="connect_btn", type="primary"):
                connect_to_esp32(port=selected_port)
        elif st.session_state.connected:
            if st.button("❌ Disconnect ESP32", on_click=disconnect_esp32, key="disconnect_btn", type="secondary"):
                pass
            if st.session_state.esp32_interface and st.session_state.esp32_interface.port:
                st.markdown(f"<div class='status-indicator' style='background-color: #e8f8f5;'><div class='status-dot connected-dot'></div>Connected on: {st.session_state.esp32_interface.port}</div>", unsafe_allow_html=True)
    
    with col2:
        if not st.session_state.connected and not st.session_state.simulation_mode:
            if st.button("▶️ Start Simulation", on_click=start_simulation, key="start_sim_btn", type="primary"):
                pass
        elif st.session_state.simulation_mode:
            if st.button("⏹️ Stop Simulation", on_click=stop_simulation, key="stop_sim_btn", type="secondary"):
                pass
    
    with col3:
        if st.session_state.connected:
            st.markdown("<div class='status-indicator' style='background-color: #e8f8f5;'><div class='status-dot connected-dot'></div><strong>Connected to ESP32</strong></div>", unsafe_allow_html=True)
        elif st.session_state.simulation_mode:
            st.markdown("<div class='status-indicator' style='background-color: #e8f0f9;'><div class='status-dot simulation-dot'></div><strong>Running in simulation mode</strong></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='status-indicator' style='background-color: #fdedec;'><div class='status-dot disconnected-dot'></div><strong>Not connected</strong></div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Dashboard", "Manual Entry", "History", "Recommendations", "LLM Analysis", "AI Farmer Intelligence 🤖"])
    
    # Dashboard Tab
    with tab1:
        # Current readings section
        st.markdown('<h2 class="sub-header">Current Soil Readings</h2>', unsafe_allow_html=True)
        
        if st.session_state.current_data:
            data = st.session_state.current_data
            
            # Display current readings in cards
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">Temperature</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{data["temperature"]:.1f} °C</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">Moisture</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{data["moisture"]:.1f} %</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">Nitrogen</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{data["nitrogen"]:.1f} mg/kg</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">Phosphorus</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{data["phosphorus"]:.1f} mg/kg</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col5:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">Potassium</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{data["potassium"]:.1f} mg/kg</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Fertility prediction
            st.markdown('<h2 class="sub-header">Soil Fertility Prediction</h2>', unsafe_allow_html=True)
            
            if st.session_state.fertility_prediction:
                fertility = st.session_state.fertility_prediction
                
                # Display fertility prediction
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    
                    # Set class based on fertility
                    if fertility['fertility_label'] == "Highly Fertile":
                        fertility_class = "fertility-high"
                    elif fertility['fertility_label'] == "Fertile":
                        fertility_class = "fertility-medium"
                    else:
                        fertility_class = "fertility-low"
                    
                    st.markdown(f'<p class="metric-label">Fertility Class</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="metric-value {fertility_class}">{fertility["fertility_label"]}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p>Confidence: {fertility["confidence"]:.1f}%</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    # Create gauge chart for fertility
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=fertility['fertility_class'],
                        title={'text': "Soil Fertility"},
                        gauge={
                            'axis': {'range': [0, 2], 'tickvals': [0, 1, 2], 'ticktext': ["Less Fertile", "Fertile", "Highly Fertile"]},
                            'bar': {'color': "darkgreen"},
                            'steps': [
                                {'range': [0, 0.5], 'color': "#ffcccb"},
                                {'range': [0.5, 1.5], 'color': "#ffffcc"},
                                {'range': [1.5, 2], 'color': "#ccffcc"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': fertility['fertility_class']
                            }
                        }
                    ))
                    
                    fig.update_layout(
                        height=250,
                        margin=dict(l=20, r=20, t=50, b=20),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Soil Health Radar Chart
            st.markdown('<h2 class="sub-header">Soil Health Overview</h2>', unsafe_allow_html=True)
            
            # Create two columns for soil health visualization
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Create and display soil health radar chart
                radar_fig = create_soil_health_radar(data)
                st.plotly_chart(radar_fig, use_container_width=True)
                
                # Add soil health insights
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">Soil Health Insights</p>', unsafe_allow_html=True)
                
                # Generate insights based on soil data
                insights = []
                
                # Check nitrogen levels
                if data['nitrogen'] < 140:
                    insights.append("⚠️ <b>Low Nitrogen:</b> Consider adding nitrogen-rich fertilizers or organic matter.")
                elif data['nitrogen'] > 280:
                    insights.append("⚠️ <b>High Nitrogen:</b> Reduce nitrogen fertilization to prevent leaching and plant burn.")
                else:
                    insights.append("✅ <b>Optimal Nitrogen:</b> Levels are within the ideal range.")
                
                # Check phosphorus levels
                if data['phosphorus'] < 25:
                    insights.append("⚠️ <b>Low Phosphorus:</b> Add phosphate fertilizers or bone meal to improve levels.")
                elif data['phosphorus'] > 50:
                    insights.append("⚠️ <b>High Phosphorus:</b> Avoid additional phosphorus to prevent water pollution.")
                else:
                    insights.append("✅ <b>Optimal Phosphorus:</b> Levels are within the ideal range.")
                
                # Check potassium levels
                if data['potassium'] < 160:
                    insights.append("⚠️ <b>Low Potassium:</b> Add potash or wood ash to improve levels.")
                elif data['potassium'] > 300:
                    insights.append("⚠️ <b>High Potassium:</b> Reduce potassium fertilization.")
                else:
                    insights.append("✅ <b>Optimal Potassium:</b> Levels are within the ideal range.")
                
                # Check pH levels
                if 'ph' in data:
                    if data['ph'] < 6.0:
                        insights.append("⚠️ <b>Acidic Soil:</b> Consider adding lime to raise pH.")
                    elif data['ph'] > 7.5:
                        insights.append("⚠️ <b>Alkaline Soil:</b> Add sulfur or organic matter to lower pH.")
                    else:
                        insights.append("✅ <b>Optimal pH:</b> Soil pH is within the ideal range.")
                
                # Display insights
                for insight in insights:
                    st.markdown(f"<p>{insight}</p>", unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Create and display moisture and temperature trends
                moisture_fig = create_trend_graph('moisture', 'Moisture (%)', '#4CAF50')
                st.plotly_chart(moisture_fig, use_container_width=True)
                
                temp_fig = create_trend_graph('temperature', 'Temperature (°C)', '#FF5722')
                st.plotly_chart(temp_fig, use_container_width=True)
            
            # Trend graphs for nutrients
            st.markdown('<h2 class="sub-header">Nutrient Trends</h2>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                nitrogen_fig = create_trend_graph('nitrogen', 'Nitrogen (mg/kg)', '#2196F3')
                st.plotly_chart(nitrogen_fig, use_container_width=True)
            
            with col2:
                phosphorus_fig = create_trend_graph('phosphorus', 'Phosphorus (mg/kg)', '#FFC107')
                st.plotly_chart(phosphorus_fig, use_container_width=True)
            
            with col3:
                potassium_fig = create_trend_graph('potassium', 'Potassium (mg/kg)', '#9C27B0')
                st.plotly_chart(potassium_fig, use_container_width=True)
        else:
            st.info("No data available. Please connect to ESP32 or start simulation.")
    
    # Manual Entry Tab
    with tab2:
        st.markdown('<h2 class="sub-header">Manual Soil Data Entry</h2>', unsafe_allow_html=True)
        
        # Create form for manual data entry
        with st.form("manual_entry_form"):
            st.markdown('<p>Enter soil sensor readings manually:</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
                moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
                ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
            
            with col2:
                nitrogen = st.number_input("Nitrogen (mg/kg)", min_value=0.0, max_value=1000.0, value=150.0, step=1.0)
                phosphorus = st.number_input("Phosphorus (mg/kg)", min_value=0.0, max_value=500.0, value=20.0, step=1.0)
                potassium = st.number_input("Potassium (mg/kg)", min_value=0.0, max_value=1000.0, value=300.0, step=1.0)
            
            submit_button = st.form_submit_button("Submit Data")
            
            if submit_button:
                # Create data dictionary
                manual_data = {
                    'temperature': temperature,
                    'moisture': moisture,
                    'nitrogen': nitrogen,
                    'phosphorus': phosphorus,
                    'potassium': potassium,
                    'ph': ph
                }
                
                # Process the manually entered data
                process_data(manual_data)
                
                st.success("Manual data submitted successfully!")
                
                # Display the fertility prediction
                if st.session_state.fertility_prediction:
                    fertility = st.session_state.fertility_prediction
                    st.markdown(f"<p><b>Fertility Prediction:</b> {fertility['fertility_label']} (Confidence: {fertility['confidence']:.1f}%)</p>", unsafe_allow_html=True)
    
    # History Tab
    with tab3:
        st.markdown('<h2 class="sub-header">Historical Data</h2>', unsafe_allow_html=True)
        
        if st.session_state.history_data:
            # Convert history data to DataFrame
            history_df = pd.DataFrame(st.session_state.history_data)
            
            # Display history data
            st.dataframe(history_df, use_container_width=True)
            
            # Download button
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download History Data",
                data=csv,
                file_name="soil_health_history.csv",
                mime="text/csv"
            )
            
            # Historical trends
            st.markdown('<h2 class="sub-header">Historical Trends</h2>', unsafe_allow_html=True)
            
            # Select parameter to visualize
            param = st.selectbox(
                "Select Parameter",
                ["temperature", "moisture", "nitrogen", "phosphorus", "potassium"]
            )
            
            # Create line chart
            fig = px.line(
                history_df,
                x="timestamp",
                y=param,
                title=f"{param.capitalize()} Over Time"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No historical data available.")
    
    # Recommendations Tab
    with tab4:
        st.markdown('<h2 class="sub-header">Crop Recommendations</h2>', unsafe_allow_html=True)
        
        if st.session_state.current_data and st.session_state.fertility_prediction:
            # Get recommendations
            recommendations = get_crop_recommendations(st.session_state.current_data, st.session_state.fertility_prediction)
            
            if recommendations and 'crop_recommendations' in recommendations:
                # Display crop recommendations
                for i, crop in enumerate(recommendations['crop_recommendations']):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.markdown(f'<div class="card">', unsafe_allow_html=True)
                        st.markdown(f'<p class="metric-label">{crop["name"]}</p>', unsafe_allow_html=True)
                        st.markdown(f'<p>Score: {crop.get("score", 85):.1f}%</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f'<div class="card">', unsafe_allow_html=True)
                        st.markdown(f'<p>{crop["description"]}</p>', unsafe_allow_html=True)
                        
                        # Get price information from CSV
                        price_info = get_crop_price_info(crop["name"])
                        if price_info:
                            st.markdown(f'<p><b>Average Price:</b> ₹{price_info["avg_modal_price"]:.2f}/quintal</p>', unsafe_allow_html=True)
                            st.markdown('<p><b>Top Markets:</b></p>', unsafe_allow_html=True)
                            for market, price in price_info['top_markets'].items():
                                st.markdown(f'<p>- {market}: ₹{price:.2f}/quintal</p>', unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown("<hr>", unsafe_allow_html=True)
                
                # Soil improvement suggestions
                st.markdown('<h2 class="sub-header">Soil Improvement Suggestions</h2>', unsafe_allow_html=True)
                
                if 'soil_improvement' in recommendations:
                    for suggestion in recommendations['soil_improvement']:
                        st.markdown(f'<div class="card">', unsafe_allow_html=True)
                        st.markdown(f'<p>{suggestion}</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.info("No recommendations available.")
        else:
            st.info("No data available for recommendations.")
    
    # LLM Analysis Tab
    with tab5:
        st.markdown('<h2 class="sub-header">Soil Health Analysis with Mistral AI</h2>', unsafe_allow_html=True)
        
        # API Key input
        col1, col2 = st.columns([3, 1])
        with col1:
            api_key = st.text_input("Mistral API Key", value="yQdfM8MLbX9uhInQ7id4iUTwN4h4pDLX", type="password")
            if api_key != st.session_state.mistral_analyzer.api_key:
                st.session_state.mistral_analyzer.set_api_key(api_key)
        
        # Analysis options
        analysis_type = st.radio(
            "Analysis Type",
            ["Standard Analysis", "Streaming Analysis"],
            horizontal=True
        )
        
        # Analysis button
        if st.button("Analyze Soil Health"):
            if not st.session_state.current_data:
                st.error("No soil data available for analysis. Please collect data first.")
            elif not api_key:
                st.error("Please enter your Mistral API Key.")
            else:
                # Get recommendations for context
                recommendations = None
                if st.session_state.current_data and st.session_state.fertility_prediction:
                    recommendations = get_crop_recommendations(
                        st.session_state.current_data, 
                        st.session_state.fertility_prediction
                    )
                
                if analysis_type == "Standard Analysis":
                    perform_standard_analysis(st.session_state.current_data, 
                                             st.session_state.fertility_prediction,
                                             recommendations)
                else:
                    perform_streaming_analysis(st.session_state.current_data, 
                                              st.session_state.fertility_prediction,
                                              recommendations)
        
        # Display analysis results
        if 'llm_analysis' in st.session_state and st.session_state.llm_analysis:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(st.session_state.llm_analysis, unsafe_allow_html=False)
            st.markdown('</div>', unsafe_allow_html=True)
            
    # AI Farmer Intelligence Tab
    with tab6:
        ai_farmer_intelligence_tab()

# Perform standard Mistral AI analysis
def perform_standard_analysis(soil_data, fertility_prediction, recommendations=None):
    st.session_state.is_analyzing = True
    
    with st.spinner("Analyzing soil health data with Mistral AI... Please wait."):
        try:
            # Call Mistral AI for analysis
            result = st.session_state.mistral_analyzer.analyze_soil(
                soil_data,
                fertility_prediction,
                recommendations
            )
            
            if 'error' in result:
                st.error(f"Analysis failed: {result['error']}")
                st.session_state.llm_analysis = f"Error: {result['error']}"
            else:
                st.session_state.llm_analysis = result['analysis']
                st.success("Analysis complete!")
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.session_state.llm_analysis = f"Error: {str(e)}"
    
    st.session_state.is_analyzing = False

# Perform streaming Mistral AI analysis
def perform_streaming_analysis(soil_data, fertility_prediction, recommendations=None):
    st.session_state.is_analyzing = True
    st.session_state.llm_analysis = ""
    
    # Create a placeholder for streaming output
    analysis_placeholder = st.empty()
    
    try:
        # Start streaming analysis
        analysis_placeholder.markdown("<div class='card'>Analyzing soil health data...</div>", unsafe_allow_html=True)
        
        # Stream the analysis
        for chunk in st.session_state.mistral_analyzer.stream_analysis(
            soil_data,
            fertility_prediction,
            recommendations
        ):
            # Check if it's an error message (JSON string)
            try:
                error_data = json.loads(chunk)
                if 'error' in error_data:
                    st.error(f"Analysis failed: {error_data['error']}")
                    st.session_state.llm_analysis = f"Error: {error_data['error']}"
                    break
            except json.JSONDecodeError:
                # Not an error JSON, continue with streaming
                st.session_state.llm_analysis += chunk
                analysis_placeholder.markdown(
                    f"<div class='card'>{st.session_state.llm_analysis}</div>", 
                    unsafe_allow_html=True
                )
        
        if not 'error' in st.session_state.llm_analysis:
            st.success("Analysis complete!")
    
    except Exception as e:
        st.error(f"An error occurred during streaming analysis: {str(e)}")
        st.session_state.llm_analysis = f"Error: {str(e)}"
        analysis_placeholder.markdown(
            f"<div class='card'>{st.session_state.llm_analysis}</div>", 
            unsafe_allow_html=True
        )
    
    st.session_state.is_analyzing = False

# AI Farmer Intelligence chatbot functions
def process_farmer_chat(user_message):
    # Add user message to chat history
    st.session_state.chat_messages.append({"role": "user", "content": user_message})
    
    # Get current soil data context if available
    context = ""
    if st.session_state.current_data:
        data = st.session_state.current_data
        context = f"Current soil readings: Temperature: {data['temperature']}°C, Moisture: {data['moisture']}%, "
        context += f"Nitrogen: {data['nitrogen']} mg/kg, Phosphorus: {data['phosphorus']} mg/kg, "
        context += f"Potassium: {data['potassium']} mg/kg, pH: {data['ph']}"
        
        if st.session_state.fertility_prediction:
            fertility = st.session_state.fertility_prediction
            context += f". Soil fertility prediction: {fertility['fertility_label']} (Confidence: {fertility['confidence']:.1f}%)"
    
    # Use Mistral API to generate response
    if st.session_state.mistral_analyzer and st.session_state.mistral_analyzer.api_key:
        try:
            # Create prompt with farming expertise focus
            system_prompt = """You are an expert AI farming assistant with deep knowledge of soil health, crop management, 
            sustainable farming practices, and agricultural science. Provide helpful, practical advice to farmers based on 
            their soil data and farming questions. Be conversational but precise, and focus on actionable recommendations. 
            If soil data is provided, analyze it and give specific advice based on the readings. If no data is available, 
            provide general best practices."""
            
            # Prepare messages for API call
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add context as assistant message if available
            if context:
                messages.append({"role": "assistant", "content": f"I have access to your current soil data: {context}. How can I help you with this information?"})
            
            # Add the last few messages from chat history (up to 4 messages)
            chat_history = st.session_state.chat_messages[-5:] if len(st.session_state.chat_messages) > 5 else st.session_state.chat_messages
            for msg in chat_history:
                if msg["role"] != "system":  # Skip system messages
                    messages.append(msg)
            
            # Call Mistral API
            response = st.session_state.mistral_analyzer.client.chat(
                model="mistral-large-latest",
                messages=messages,
                temperature=0.7,
                max_tokens=1024
            )
            
            # Extract response
            assistant_response = response.choices[0].message.content
            
            # Add assistant response to chat history
            st.session_state.chat_messages.append({"role": "assistant", "content": assistant_response})
            
            return assistant_response
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
            return error_msg
    else:
        no_api_msg = "Please set up your Mistral API key in the LLM Analysis tab to use the AI Farmer Intelligence feature."
        st.session_state.chat_messages.append({"role": "assistant", "content": no_api_msg})
        return no_api_msg

# AI Farmer Intelligence Tab
def ai_farmer_intelligence_tab():
    st.markdown('<h2 class="sub-header">AI Farmer Intelligence Assistant</h2>', unsafe_allow_html=True)
    
    # Check if Mistral API key is set
    if not st.session_state.mistral_analyzer or not st.session_state.mistral_analyzer.api_key:
        st.warning("Please set up your Mistral API key in the LLM Analysis tab to use the AI Farmer Intelligence feature.")
        st.info("Once you've set up your API key, you can chat with the AI Farmer Intelligence Assistant to get personalized farming advice based on your soil data.")
        return
    
    # Chat container with custom styling
    st.markdown("""
    <style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e8f4f8;
        border-left: 5px solid #3498db;
    }
    .chat-message.assistant {
        background-color: #f0f7f0;
        border-left: 5px solid #27ae60;
    }
    .chat-message .content {
        display: flex;
        margin-top: 0.5rem;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex-grow: 1;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="content">
                        <img src="https://cdn-icons-png.flaticon.com/512/1077/1077114.png" class="avatar" alt="User"/>
                        <div class="message">{message["content"]}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant">
                    <div class="content">
                        <img src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png" class="avatar" alt="AI"/>
                        <div class="message">{message["content"]}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # User input
    with st.container():
        # Initialize the input value in session state if it doesn't exist
        if "farmer_chat_input" not in st.session_state:
            st.session_state.farmer_chat_input = ""
            
        # Define a callback function to handle the send button click
        def send_message():
            if st.session_state.farmer_chat_input:
                user_input = st.session_state.farmer_chat_input
                # Process user message and get response
                with st.spinner("AI Farmer thinking..."):
                    process_farmer_chat(user_input)
                # Clear input after processing but before the next rerun
                st.session_state.farmer_chat_input = ""
                # Rerun to update chat display
                st.rerun()
                
        # Create the text input and button
        # Use a different approach to avoid modifying session state after widget instantiation
        col1, col2 = st.columns([4, 1])
        with col1:
            st.text_input("Ask your farming question:", key="farmer_chat_input", on_change=send_message)
        with col2:
            st.button("Send", type="primary", key="send_farmer_chat", on_click=send_message)

# Main function
def main():
    # Initialize session state
    init_session_state()
    
    # Run main dashboard
    main_dashboard()

# Run the app
if __name__ == "__main__":
    main()