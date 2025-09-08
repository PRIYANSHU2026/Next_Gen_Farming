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
from utils.soil_health_interface import SoilHealthPredictor, ESP32Interface, ESP32WebInterface
from models.plant_recommendation import PlantRecommendationSystem
from api.mistral_soil_analysis import MistralSoilAnalysis

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
    /* Main Theme Colors */
    :root {
        --farm-green-dark: #2E7D32;
        --farm-green-medium: #388E3C;
        --farm-green-light: #689F38;
        --farm-brown: #795548;
        --farm-soil: #5D4037;
        --farm-wheat: #F9A825;
        --farm-sky: #64B5F6;
        --farm-bg-light: #F1F8E9;
    }
    
    /* Typography */
    .main-header {
        font-size: 2.5rem;
        color: var(--farm-green-dark);
        text-align: center;
        margin-bottom: 1rem;
        font-family: 'Roboto Slab', serif;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        background: linear-gradient(to right, var(--farm-green-dark), var(--farm-green-medium));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: var(--farm-green-medium);
        margin-bottom: 0.5rem;
        border-bottom: 2px solid var(--farm-green-light);
        padding-bottom: 5px;
        font-family: 'Roboto', sans-serif;
    }
    
    /* Cards */
    .card {
        background-color: var(--farm-bg-light);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid var(--farm-green-medium);
        transition: transform 0.2s;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* NPK Cards */
    .npk-card {
        background-color: var(--farm-bg-light);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-top: 4px solid var(--farm-green-medium);
        transition: transform 0.2s;
    }
    
    .npk-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Metrics */
    .metric-label {
        font-size: 1.1rem;
        color: var(--farm-green-light);
        font-weight: bold;
        font-family: 'Roboto', sans-serif;
        display: flex;
        align-items: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        font-family: 'Roboto Mono', monospace;
    }
    
    /* Fertility Classes */
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
    
    /* NPK Styling */
    .npk-category {
        font-size: 1.2rem;
        font-weight: bold;
        margin: 5px 0;
        padding: 5px 10px;
        border-radius: 5px;
        display: inline-block;
    }
    
    .low-category {
        background-color: rgba(211, 47, 47, 0.1);
        color: #D32F2F;
        border: 1px solid #D32F2F;
    }
    
    .medium-category {
        background-color: rgba(255, 160, 0, 0.1);
        color: #FFA000;
        border: 1px solid #FFA000;
    }
    
    .high-category {
        background-color: rgba(46, 125, 50, 0.1);
        color: #2E7D32;
        border: 1px solid #2E7D32;
    }
    
    .very-high-category {
        background-color: rgba(21, 101, 192, 0.1);
        color: #1565C0;
        border: 1px solid #1565C0;
    }
    
    .npk-health {
        font-size: 0.9rem;
        color: var(--farm-green-dark);
        margin-top: 5px;
        text-align: right;
    }
    
    .npk-recommendation-title {
        font-size: 0.9rem;
        color: var(--farm-brown);
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 5px;
    }
    
    .npk-recommendation {
        font-size: 0.85rem;
        color: var(--farm-soil);
        background-color: rgba(121, 85, 72, 0.1);
        padding: 8px;
        border-radius: 5px;
        border-left: 3px solid var(--farm-brown);
    }
    
    /* NPK Crop Items */
    .npk-crop-item {
        margin: 0.3rem 0;
        padding-left: 0.5rem;
        font-size: 0.95rem;
        color: var(--farm-soil);
    }
    
    /* NPK Cards Specific */
    .n-card {
        border-top: 4px solid #4CAF50; /* Nitrogen green */
    }
    
    .p-card {
        border-top: 4px solid #2196F3; /* Phosphorus blue */
    }
    
    .k-card {
        border-top: 4px solid #FF9800; /* Potassium orange */
    }
    
    /* Progress Bar Customization */
    .stProgress > div > div {
        background-color: var(--farm-green-medium);
    }
</style>
""", unsafe_allow_html=True)

# Load crop data from CSV
@st.cache_data
def load_crop_data():
    try:
        # Load crop recommendations CSV from project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        crop_recommendations_path = os.path.join(project_root, "crop_recommendations.csv")
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
    if 'connection_type' not in st.session_state:
        st.session_state.connection_type = "serial"  # Options: "serial", "web"
    if 'esp32_ip' not in st.session_state:
        st.session_state.esp32_ip = "http://localhost"  # Default IP address
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
def connect_to_esp32(port=None, ip_address=None, connection_type="serial"):
    try:
        # Set connection type
        st.session_state.connection_type = connection_type
        
        if connection_type == "serial":
            # Connect via serial port
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
        elif connection_type == "web":
            # Connect via web interface
            with st.spinner(f"Connecting to ESP32 Web Server at {ip_address}..."):
                st.session_state.esp32_ip = ip_address
                
                # Create ESP32WebInterface with debug info
                st.info(f"Creating ESP32WebInterface with IP: {ip_address}")
                st.session_state.esp32_interface = ESP32WebInterface(ip_address=ip_address)
                
                # Try to detect ESP32 devices on the network
                st.info("Scanning network for ESP32 devices...")
                available_ips = st.session_state.esp32_interface.scan_for_devices()
                
                if available_ips:
                    st.success(f"Found ESP32 devices at: {', '.join(available_ips)}")
                    # If the provided IP is not in the list, suggest using one of the found IPs
                    if ip_address not in available_ips and not any(ip_address in found_ip for found_ip in available_ips):
                        st.warning(f"The provided IP ({ip_address}) was not found in the scan results.")
                        st.info(f"Consider using one of the detected IPs: {', '.join(available_ips)}")
                
                # Attempt connection
                st.info(f"Attempting to connect to {ip_address}...")
                success = st.session_state.esp32_interface.connect()
                
                if success:
                    st.session_state.connected = True
                    st.success(f"Connected to ESP32 Web Server at {ip_address}!")
                    
                    # Test data reading
                    st.info("Testing data reading...")
                    test_data = st.session_state.esp32_interface.read_data()
                    if test_data:
                        st.success(f"Successfully read data: {test_data}")
                    else:
                        st.warning("Connected but couldn't read data. Will keep trying in background.")
                    
                    # Start data collection thread
                    st.info("Starting data collection thread...")
                    threading.Thread(target=collect_data, daemon=True).start()
                    
                    # Force dashboard to update
                    st.rerun()
                    
                    return True
                else:
                    st.error(f"Failed to connect to ESP32 Web Server at {ip_address}.")
                    st.info("Please check if the ESP32 is running and the IP address is correct.")
                    st.info("Make sure the ESP32 has a web server running and is accessible on your network.")
                    
                    # Ask if user wants to start simulation mode
                    if st.button("Start Simulation Mode Instead"):
                        st.session_state.connected = True  # Set connected to true for simulation
                        st.session_state.simulation_mode = True
                        st.success("Starting simulation mode with generated data!")
                        start_simulation()
                        return True
                    
                    st.session_state.connected = False
                    return False
        else:
            st.error(f"Invalid connection type: {connection_type}")
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
    # Generate initial data immediately
    data = generate_simulated_data()
    process_data(data)
    
    # Continue generating data at intervals
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
    
    # Force dashboard to update
    st.rerun()

# Stop simulation
def stop_simulation():
    st.session_state.stop_simulation = True
    st.session_state.simulation_mode = False
    if st.session_state.simulation_thread:
        st.session_state.simulation_thread.join(timeout=1)
    st.info("Simulation stopped")

# Collect data from ESP32
def collect_data():
    consecutive_failures = 0
    max_failures = 5  # Maximum consecutive failures before showing warning
    
    while st.session_state.connected and st.session_state.esp32_interface:
        try:
            print("🔄 Attempting to read data from ESP32...")
            data = st.session_state.esp32_interface.read_data()
            
            if data:
                print(f"✅ Data received from ESP32: {data}")
                
                # Check if this is simulated data
                if data.get('simulated', False):
                    print("📊 Using simulated data since real data is unavailable")
                    with st.sidebar:
                        if consecutive_failures >= max_failures:
                            st.info("Using simulated data since ESP32 data is unavailable")
                
                # Process the data regardless of source
                process_data(data)
                
                # Only reset failure counter if this was real data
                if not data.get('simulated', False):
                    consecutive_failures = 0
                else:
                    # Still count as a failure if using simulated data
                    consecutive_failures += 1
            else:
                consecutive_failures += 1
                print(f"⚠️ No data received from ESP32. Attempt {consecutive_failures}/{max_failures}")
                
                # Show warning in UI after several failed attempts
                if consecutive_failures >= max_failures:
                    with st.sidebar:
                        st.warning("Having trouble getting data from ESP32. Check connection settings.")
                        
                    # Try to force reconnect after multiple failures
                    if consecutive_failures >= max_failures * 2:
                        print("🔄 Attempting to reconnect to ESP32...")
                        st.session_state.esp32_interface.connect()
        except Exception as e:
            consecutive_failures += 1
            print(f"❌ Error reading data: {e}")
            
            # Show error in UI after several failed attempts
            if consecutive_failures >= max_failures:
                with st.sidebar:
                    st.error(f"Error reading data from ESP32: {str(e)}")
        
        # Adaptive sleep time - increase delay after failures to avoid hammering the device
        sleep_time = min(2 + (consecutive_failures * 0.5), 10)  # Max 10 seconds
        time.sleep(sleep_time)

# Create the connection sidebar
def create_connection_sidebar():
    with st.sidebar:
        st.header("📡 ESP32 Connection")
        
        # Connection type selection
        connection_type = st.radio(
            "Connection Type",
            options=["Serial", "Web"],
            index=0 if st.session_state.connection_type == "serial" else 1,
            key="connection_type_radio"
        )
        
        if connection_type == "Serial":
            # Serial connection options
            st.session_state.connection_type = "serial"
            
            # Get available ports
            import serial.tools.list_ports
            ports = [port.device for port in serial.tools.list_ports.comports()]
            
            if ports:
                selected_port = st.selectbox("Select Serial Port", options=ports)
                connect_button = st.button("Connect via Serial")
                
                if connect_button:
                    connect_to_esp32(port=selected_port, connection_type="serial")
            else:
                st.warning("No serial ports found. Please connect your ESP32 device.")
        
        else:  # Web connection
            # Web connection options
            st.session_state.connection_type = "web"
            
            # Initialize ESP32WebInterface for scanning if not already done
            if "esp32_web_scanner" not in st.session_state:
                st.session_state.esp32_web_scanner = ESP32WebInterface()
                
            # Scan button for automatic IP detection
            col1, col2 = st.columns([1, 1])
            with col1:
                scan_button = st.button("🔍 Scan for Devices")
            
            # Display scanning status
            if scan_button:
                with st.spinner("Scanning network for ESP32 devices..."):
                    available_ips = st.session_state.esp32_web_scanner.scan_for_devices()
                    if available_ips:
                        st.session_state.available_esp32_ips = available_ips
                        st.success(f"Found {len(available_ips)} ESP32 device(s)!")
                    else:
                        st.warning("No ESP32 devices found on the network.")
                        if "available_esp32_ips" not in st.session_state:
                            st.session_state.available_esp32_ips = []
            
            # Show detected devices if any
            if "available_esp32_ips" in st.session_state and st.session_state.available_esp32_ips:
                st.subheader("Detected Devices")
                selected_device = st.selectbox(
                    "Select ESP32 Device",
                    options=st.session_state.available_esp32_ips,
                    format_func=lambda x: f"ESP32 at {x}"
                )
                
                # Set the selected device as the IP address
                if selected_device:
                    st.session_state.esp32_ip = selected_device
            
            # Manual IP address input
            st.subheader("Manual Connection")
            ip_address = st.text_input(
                "ESP32 IP Address",
                value=st.session_state.esp32_ip if "esp32_ip" in st.session_state else "http://localhost",
                help="Enter the IP address of your ESP32 web server (e.g., http://192.168.1.100)"
            )
            
            connect_button = st.button("Connect via Web")
            
            if connect_button:
                connect_to_esp32(ip_address=ip_address, connection_type="web")
        
        # Disconnect button (only show if connected)
        if st.session_state.connected:
            if st.button("Disconnect"):
                disconnect_esp32()
        
        # Simulation controls
        st.header("🔄 Simulation")
        
        if not st.session_state.simulation_mode:
            if st.button("Start Simulation"):
                start_simulation()
        else:
            if st.button("Stop Simulation"):
                stop_simulation()

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
    
    # Map ESP32 data fields to expected dashboard fields
    processed_data = {}
    
    # Handle temperature (could be from DHT or DS18B20 sensor)
    if 'temperature_dht' in data:
        processed_data['temperature'] = data['temperature_dht']
    elif 'temperature_ds18b20' in data:
        processed_data['temperature'] = data['temperature_ds18b20']
    elif 'temperature' in data:
        processed_data['temperature'] = data['temperature']
    
    # Handle other fields
    if 'humidity' in data:
        processed_data['humidity'] = data['humidity']
    
    if 'moisture' in data:
        processed_data['moisture'] = data['moisture']
    
    if 'ph' in data:
        processed_data['ph'] = data['ph']
    
    if 'nitrogen' in data:
        processed_data['nitrogen'] = data['nitrogen']
    
    if 'phosphorus' in data:
        processed_data['phosphorus'] = data['phosphorus']
    
    if 'potassium' in data:
        processed_data['potassium'] = data['potassium']
    
    # Check if data is simulated
    if 'simulated' in data:
        processed_data['simulated'] = data['simulated']
    
    # Update the current data with processed data
    if not st.session_state.current_data:
        st.session_state.current_data = processed_data
    else:
        st.session_state.current_data.update(processed_data)
    
    # Store timestamp
    st.session_state.timestamps.append(timestamp)
    
    # Update sensor data for trends
    for key in st.session_state.sensor_data.keys():
        if key in processed_data:
            st.session_state.sensor_data[key].append(processed_data[key])
    
    # Make fertility prediction with NPK analysis
    fertility_prediction = predict_fertility(processed_data)
    
    # Store prediction
    st.session_state.fertility_prediction = fertility_prediction
    
    # Add to history with NPK categories
    history_entry = {
        'timestamp': timestamp,
        **processed_data,
        'fertility_class': fertility_prediction['fertility_label'],
        'n_category': fertility_prediction.get('n_category', 'Unknown'),
        'p_category': fertility_prediction.get('p_category', 'Unknown'),
        'k_category': fertility_prediction.get('k_category', 'Unknown')
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
        
        # Get NPK predictions
        npk_result = st.session_state.soil_predictor.predict_npk_levels(data)
        
        # Combine fertility and NPK predictions
        prediction = {
            'fertility_class': fertility_class,
            'fertility_label': fertility_label,
            'confidence': result.get('confidence', 85.0),  # Default confidence if not provided
            # Add NPK values, categories, recommendations and health percentages
            'n_value': npk_result.get('n_value', 0),
            'n_category': npk_result.get('n_category', 'Unknown'),
            'n_recommendation': npk_result.get('n_recommendation', ''),
            'n_health': npk_result.get('n_health', 0),
            'p_value': npk_result.get('p_value', 0),
            'p_category': npk_result.get('p_category', 'Unknown'),
            'p_recommendation': npk_result.get('p_recommendation', ''),
            'p_health': npk_result.get('p_health', 0),
            'k_value': npk_result.get('k_value', 0),
            'k_category': npk_result.get('k_category', 'Unknown'),
            'k_recommendation': npk_result.get('k_recommendation', ''),
            'k_health': npk_result.get('k_health', 0)
        }
        
        return prediction
    except Exception as e:
        print(f"Error predicting fertility: {e}")
        return {
            'fertility_class': 1,  # Default to "Fertile"
            'fertility_label': "Fertile",
            'confidence': 50.0,
            'n_value': 0, 'n_category': 'Unknown', 'n_recommendation': '', 'n_health': 0,
            'p_value': 0, 'p_category': 'Unknown', 'p_recommendation': '', 'p_health': 0,
            'k_value': 0, 'k_category': 'Unknown', 'k_recommendation': '', 'k_health': 0
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
    
    # Use the new connection sidebar with IP address input option
    create_connection_sidebar()
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Dashboard", "Manual Entry", "History", "Recommendations", "LLM Analysis", "AI Farmer Intelligence 🤖", "Help & Info 📚"])
    
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
            
            # NPK Analysis Section
            st.markdown('<h2 class="sub-header">NPK Analysis</h2>', unsafe_allow_html=True)
            
            if st.session_state.fertility_prediction:
                npk_data = st.session_state.fertility_prediction
                
                # Create three columns for N, P, K meters
                col1, col2, col3 = st.columns(3)
                
                # Nitrogen Column
                with col1:
                    st.markdown('<div class="npk-card">', unsafe_allow_html=True)
                    st.markdown('<p class="metric-label">Nitrogen (N)</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="metric-value">{npk_data["n_value"]:.1f} mg/kg</p>', unsafe_allow_html=True)
                    
                    # Add N category with appropriate styling
                    n_category = npk_data["n_category"]
                    if n_category == "Low":
                        n_color = "#FF5252"  # Red
                    elif n_category == "Medium":
                        n_color = "#FFC107"  # Amber
                    elif n_category == "High":
                        n_color = "#4CAF50"  # Green
                    elif n_category == "Very High":
                        n_color = "#2196F3"  # Blue
                    else:
                        n_color = "#9E9E9E"  # Grey
                    
                    # Display category and health percentage
                    st.markdown(f'<p class="npk-category" style="color:{n_color}">{n_category}</p>', unsafe_allow_html=True)
                    
                    # Create progress bar for N health
                    n_health = npk_data["n_health"]
                    st.progress(n_health/100)
                    st.markdown(f'<p class="npk-health">Health: {n_health}%</p>', unsafe_allow_html=True)
                    
                    # Display recommendation
                    st.markdown('<p class="npk-recommendation-title">Recommendation:</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="npk-recommendation">{npk_data["n_recommendation"]}</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Phosphorus Column
                with col2:
                    st.markdown('<div class="npk-card">', unsafe_allow_html=True)
                    st.markdown('<p class="metric-label">Phosphorus (P)</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="metric-value">{npk_data["p_value"]:.1f} mg/kg</p>', unsafe_allow_html=True)
                    
                    # Add P category with appropriate styling
                    p_category = npk_data["p_category"]
                    if p_category == "Low":
                        p_color = "#FF5252"  # Red
                    elif p_category == "Medium":
                        p_color = "#FFC107"  # Amber
                    elif p_category == "High":
                        p_color = "#4CAF50"  # Green
                    elif p_category == "Very High":
                        p_color = "#2196F3"  # Blue
                    else:
                        p_color = "#9E9E9E"  # Grey
                    
                    # Display category and health percentage
                    st.markdown(f'<p class="npk-category" style="color:{p_color}">{p_category}</p>', unsafe_allow_html=True)
                    
                    # Create progress bar for P health
                    p_health = npk_data["p_health"]
                    st.progress(p_health/100)
                    st.markdown(f'<p class="npk-health">Health: {p_health}%</p>', unsafe_allow_html=True)
                    
                    # Display recommendation
                    st.markdown('<p class="npk-recommendation-title">Recommendation:</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="npk-recommendation">{npk_data["p_recommendation"]}</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Potassium Column
                with col3:
                    st.markdown('<div class="npk-card">', unsafe_allow_html=True)
                    st.markdown('<p class="metric-label">Potassium (K)</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="metric-value">{npk_data["k_value"]:.1f} mg/kg</p>', unsafe_allow_html=True)
                    
                    # Add K category with appropriate styling
                    k_category = npk_data["k_category"]
                    if k_category == "Low":
                        k_color = "#FF5252"  # Red
                    elif k_category == "Medium":
                        k_color = "#FFC107"  # Amber
                    elif k_category == "High":
                        k_color = "#4CAF50"  # Green
                    elif k_category == "Very High":
                        k_color = "#2196F3"  # Blue
                    else:
                        k_color = "#9E9E9E"  # Grey
                    
                    # Display category and health percentage
                    st.markdown(f'<p class="npk-category" style="color:{k_color}">{k_category}</p>', unsafe_allow_html=True)
                    
                    # Create progress bar for K health
                    k_health = npk_data["k_health"]
                    st.progress(k_health/100)
                    st.markdown(f'<p class="npk-health">Health: {k_health}%</p>', unsafe_allow_html=True)
                    
                    # Display recommendation
                    st.markdown('<p class="npk-recommendation-title">Recommendation:</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="npk-recommendation">{npk_data["k_recommendation"]}</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Add NPK Help Button
                with st.expander("ℹ️ NPK Information"):
                    st.markdown("""
                    ### Understanding NPK Values
                    
                    **Nitrogen (N)**: Essential for leaf growth and green color. Measured in mg/kg.
                    - Low: < 140 mg/kg - Plants may show yellowing of older leaves
                    - Medium: 140-280 mg/kg - Adequate for most plants
                    - High: 280-560 mg/kg - Good levels for leafy vegetables
                    - Very High: > 560 mg/kg - May cause excessive vegetative growth
                    
                    **Phosphorus (P)**: Important for root development and flowering. Measured in mg/kg.
                    - Low: < 10 mg/kg - Plants may have stunted growth and poor flowering
                    - Medium: 10-20 mg/kg - Adequate for most plants
                    - High: 20-40 mg/kg - Good for flowering and fruiting plants
                    - Very High: > 40 mg/kg - May interfere with nutrient uptake
                    
                    **Potassium (K)**: Helps with overall plant health and disease resistance. Measured in mg/kg.
                    - Low: < 200 mg/kg - Plants may have weak stems and poor disease resistance
                    - Medium: 200-400 mg/kg - Adequate for most plants
                    - High: 400-800 mg/kg - Good for root crops and stress resistance
                    - Very High: > 800 mg/kg - May cause nutrient imbalances
                    
                    The **Health Percentage** indicates how optimal each nutrient level is for plant growth.
                    """)
                
                # Add Nutrient Distribution Pie Chart
                st.markdown('<h3 class="sub-header">Nutrient Distribution</h3>', unsafe_allow_html=True)
                
                # Create pie chart for NPK distribution
                if st.session_state.fertility_prediction:
                    npk_data = st.session_state.fertility_prediction
                    
                    # Extract NPK values
                    n_value = npk_data["n_value"]
                    p_value = npk_data["p_value"]
                    k_value = npk_data["k_value"]
                    
                    # Create two columns for pie chart and legend
                    pie_col1, pie_col2 = st.columns([3, 1])
                    
                    with pie_col1:
                        # Create pie chart
                        labels = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
                        values = [n_value, p_value, k_value]
                        
                        # Define colors based on NPK categories
                        colors = [
                            '#FF5252' if npk_data["n_category"] == "Low" else 
                            '#FFC107' if npk_data["n_category"] == "Medium" else 
                            '#4CAF50' if npk_data["n_category"] == "High" else '#2196F3',
                            
                            '#FF5252' if npk_data["p_category"] == "Low" else 
                            '#FFC107' if npk_data["p_category"] == "Medium" else 
                            '#4CAF50' if npk_data["p_category"] == "High" else '#2196F3',
                            
                            '#FF5252' if npk_data["k_category"] == "Low" else 
                            '#FFC107' if npk_data["k_category"] == "Medium" else 
                            '#4CAF50' if npk_data["k_category"] == "High" else '#2196F3'
                        ]
                        
                        # Calculate percentages for display
                        total = sum(values)
                        if total > 0:
                            percentages = [round((val/total)*100, 1) for val in values]
                        else:
                            # If total is zero, set equal percentages or zeros
                            percentages = [0, 0, 0]
                        
                        # Create custom hover text
                        hover_text = [
                            f"Nitrogen: {n_value:.1f} mg/kg ({percentages[0]}%)<br>Status: {npk_data['n_category']}",
                            f"Phosphorus: {p_value:.1f} mg/kg ({percentages[1]}%)<br>Status: {npk_data['p_category']}",
                            f"Potassium: {k_value:.1f} mg/kg ({percentages[2]}%)<br>Status: {npk_data['k_category']}"
                        ]
                        
                        # Create pie chart
                        fig = go.Figure(data=[go.Pie(
                            labels=labels,
                            values=values,
                            hole=.4,
                            hoverinfo='text',
                            hovertext=hover_text,
                            marker=dict(colors=colors),
                            textinfo='label+percent',
                            textfont=dict(size=12),
                            insidetextorientation='radial'
                        )])
                        
                        fig.update_layout(
                            title="Soil Nutrient Distribution",
                            height=400,
                            margin=dict(l=20, r=20, t=50, b=20),
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with pie_col2:
                        # Create legend/explanation
                        st.markdown("### Nutrient Ratio")
                        st.markdown(f"**N**: {percentages[0]}%")
                        st.markdown(f"**P**: {percentages[1]}%")
                        st.markdown(f"**K**: {percentages[2]}%")
                        
                        # Add ideal ratio information
                        st.markdown("### Ideal NPK Ratio")
                        st.markdown("General crops: 3-1-2")
                        st.markdown("Leafy greens: 4-1-2")
                        st.markdown("Root vegetables: 1-2-3")
                        st.markdown("Fruits/flowers: 1-2-2")
            
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
            
            # Add styling to the dataframe
            def highlight_npk_categories(val):
                if val == 'Low':
                    return 'background-color: #FFEBEE; color: #D32F2F; font-weight: bold'
                elif val == 'Medium':
                    return 'background-color: #FFF8E1; color: #FFA000; font-weight: bold'
                elif val == 'High':
                    return 'background-color: #E8F5E9; color: #2E7D32; font-weight: bold'
                elif val == 'Very High':
                    return 'background-color: #E3F2FD; color: #1976D2; font-weight: bold'
                return ''
            
            # Apply styling to NPK category columns - using .map instead of .applymap to avoid deprecation warning
            styled_df = history_df.style.map(
                highlight_npk_categories, 
                subset=['n_category', 'p_category', 'k_category']
            )
            
            # Display styled history data
            st.dataframe(styled_df, use_container_width=True)
            
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
            
            # Create tabs for different trend visualizations
            trend_tab1, trend_tab2, trend_tab3 = st.tabs(["Single Parameter", "Multi-Parameter Comparison", "NPK Balance Over Time"])
            
            with trend_tab1:
                # Select parameter to visualize
                param = st.selectbox(
                    "Select Parameter",
                    ["temperature", "moisture", "nitrogen", "phosphorus", "potassium"]
                )
                
                # Create line chart with improved styling
                fig = px.line(
                    history_df,
                    x="timestamp",
                    y=param,
                    title=f"{param.capitalize()} Over Time"
                )
                
                # Update chart styling
                fig.update_layout(
                    plot_bgcolor='rgba(240, 248, 235, 0.6)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_family="Roboto",
                    height=500,
                    xaxis=dict(title="Time"),
                    yaxis=dict(title=f"{param.capitalize()} Value")
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with trend_tab2:
                st.subheader("Compare Multiple Parameters")
                
                # Multi-select for parameters
                selected_params = st.multiselect(
                    "Select Parameters to Compare",
                    ["temperature", "moisture", "nitrogen", "phosphorus", "potassium"],
                    default=["nitrogen", "phosphorus", "potassium"]
                )
                
                if selected_params:
                    # Create multi-line chart
                    multi_fig = go.Figure()
                    
                    colors = {
                        "temperature": "#FF5722",
                        "moisture": "#03A9F4",
                        "nitrogen": "#2196F3",
                        "phosphorus": "#FFC107",
                        "potassium": "#9C27B0"
                    }
                    
                    for param in selected_params:
                        multi_fig.add_trace(go.Scatter(
                            x=history_df["timestamp"],
                            y=history_df[param],
                            mode='lines+markers',
                            name=param.capitalize(),
                            line=dict(color=colors.get(param, "#000000"), width=2),
                            marker=dict(size=6)
                        ))
                    
                    # Update layout
                    multi_fig.update_layout(
                        title="Parameter Comparison Over Time",
                        plot_bgcolor='rgba(240, 248, 235, 0.6)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_family="Roboto",
                        height=500,
                        xaxis=dict(title="Time"),
                        yaxis=dict(title="Value"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(multi_fig, use_container_width=True)
                    
                    # Add correlation heatmap
                    if len(selected_params) > 1:
                        st.subheader("Parameter Correlation")
                        corr_df = history_df[selected_params].corr()
                        corr_fig = px.imshow(
                            corr_df,
                            text_auto=True,
                            color_continuous_scale='Viridis',
                            aspect="auto"
                        )
                        corr_fig.update_layout(height=400)
                        st.plotly_chart(corr_fig, use_container_width=True)
                else:
                    st.info("Please select at least one parameter to display the chart")
            
            with trend_tab3:
                st.subheader("NPK Balance Over Time")
                
                # Create area chart for NPK balance
                npk_fig = go.Figure()
                
                # Add traces for N, P, K
                npk_fig.add_trace(go.Scatter(
                    x=history_df["timestamp"],
                    y=history_df["nitrogen"],
                    mode='lines',
                    name='Nitrogen',
                    line=dict(width=0.5, color='#2196F3'),
                    stackgroup='one',
                    groupnorm='percent'
                ))
                
                npk_fig.add_trace(go.Scatter(
                    x=history_df["timestamp"],
                    y=history_df["phosphorus"],
                    mode='lines',
                    name='Phosphorus',
                    line=dict(width=0.5, color='#FFC107'),
                    stackgroup='one'
                ))
                
                npk_fig.add_trace(go.Scatter(
                    x=history_df["timestamp"],
                    y=history_df["potassium"],
                    mode='lines',
                    name='Potassium',
                    line=dict(width=0.5, color='#9C27B0'),
                    stackgroup='one'
                ))
                
                # Update layout
                npk_fig.update_layout(
                    title="NPK Balance Over Time (Percentage)",
                    plot_bgcolor='rgba(240, 248, 235, 0.6)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_family="Roboto",
                    height=500,
                    xaxis=dict(title="Time"),
                    yaxis=dict(title="Percentage (%)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(npk_fig, use_container_width=True)
                
                # Add explanation
                st.info(
                    "This chart shows the relative balance of Nitrogen, Phosphorus, and Potassium over time. "
                    "The ideal NPK ratio depends on your specific crops and soil conditions."
                )
            
            # Update line color based on parameter
            if param == "nitrogen":
                fig.update_traces(line_color="#4CAF50")
            elif param == "phosphorus":
                fig.update_traces(line_color="#2196F3")
            elif param == "potassium":
                fig.update_traces(line_color="#FF9800")
            elif param == "temperature":
                fig.update_traces(line_color="#F44336")
            elif param == "moisture":
                fig.update_traces(line_color="#03A9F4")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # NPK Category Distribution
            st.markdown('<h2 class="sub-header">NPK Category Distribution</h2>', unsafe_allow_html=True)
            
            # Create three columns for N, P, K category distribution
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Count N categories
                n_counts = history_df['n_category'].value_counts().reset_index()
                n_counts.columns = ['Category', 'Count']
                
                # Create pie chart for N categories
                fig_n = px.pie(
                    n_counts, 
                    values='Count', 
                    names='Category',
                    title='Nitrogen Categories',
                    color='Category',
                    color_discrete_map={
                        'Low': '#FF5252',
                        'Medium': '#FFC107',
                        'High': '#4CAF50',
                        'Very High': '#2196F3',
                        'Unknown': '#9E9E9E'
                    }
                )
                fig_n.update_layout(title_font_color="#388E3C")
                st.plotly_chart(fig_n, use_container_width=True)
            
            with col2:
                # Count P categories
                p_counts = history_df['p_category'].value_counts().reset_index()
                p_counts.columns = ['Category', 'Count']
                
                # Create pie chart for P categories
                fig_p = px.pie(
                    p_counts, 
                    values='Count', 
                    names='Category',
                    title='Phosphorus Categories',
                    color='Category',
                    color_discrete_map={
                        'Low': '#FF5252',
                        'Medium': '#FFC107',
                        'High': '#4CAF50',
                        'Very High': '#2196F3',
                        'Unknown': '#9E9E9E'
                    }
                )
                fig_p.update_layout(title_font_color="#388E3C")
                st.plotly_chart(fig_p, use_container_width=True)
            
            with col3:
                # Count K categories
                k_counts = history_df['k_category'].value_counts().reset_index()
                k_counts.columns = ['Category', 'Count']
                
                # Create pie chart for K categories
                fig_k = px.pie(
                    k_counts, 
                    values='Count', 
                    names='Category',
                    title='Potassium Categories',
                    color='Category',
                    color_discrete_map={
                        'Low': '#FF5252',
                        'Medium': '#FFC107',
                        'High': '#4CAF50',
                        'Very High': '#2196F3',
                        'Unknown': '#9E9E9E'
                    }
                )
                fig_k.update_layout(title_font_color="#388E3C")
                st.plotly_chart(fig_k, use_container_width=True)
        else:
            st.info("No historical data available.")
    
    # Recommendations Tab
    with tab4:
        st.markdown('<h2 class="sub-header">Crop Recommendations</h2>', unsafe_allow_html=True)
        
        if st.session_state.current_data and st.session_state.fertility_prediction:
            # Get recommendations
            recommendations = get_crop_recommendations(st.session_state.current_data, st.session_state.fertility_prediction)
            
            # NPK-based Recommendations
            st.markdown('<h3 class="sub-header">NPK-Specific Recommendations</h3>', unsafe_allow_html=True)
            
            if 'npk_values' in st.session_state.fertility_prediction and 'npk_categories' in st.session_state.fertility_prediction:
                npk_values = st.session_state.fertility_prediction['npk_values']
                npk_categories = st.session_state.fertility_prediction['npk_categories']
                npk_recommendations = st.session_state.fertility_prediction.get('npk_recommendations', {})
                
                # Create three columns for N, P, K specific recommendations
                col_n, col_p, col_k = st.columns(3)
                
                with col_n:
                    st.markdown(f'<div class="npk-card n-card">', unsafe_allow_html=True)
                    st.markdown(f'<p class="metric-label">Nitrogen-Loving Crops</p>', unsafe_allow_html=True)
                    n_category = npk_categories.get('nitrogen', 'Unknown')
                    n_crops = []
                    
                    if n_category == 'Low':
                        n_crops = ['Legumes (Beans, Peas)', 'Clover', 'Alfalfa']
                    elif n_category == 'Medium':
                        n_crops = ['Tomatoes', 'Peppers', 'Squash', 'Cucumbers']
                    elif n_category == 'High':
                        n_crops = ['Corn', 'Leafy Greens', 'Cabbage', 'Broccoli']
                    elif n_category == 'Very High':
                        n_crops = ['Rice', 'Wheat', 'Sugarcane', 'Cotton']
                    
                    for crop in n_crops:
                        st.markdown(f'<p class="npk-crop-item">• {crop}</p>', unsafe_allow_html=True)
                    
                    st.markdown(f'<p class="npk-category {n_category.lower().replace(" ", "-")}-category">Current Level: {n_category}</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_p:
                    st.markdown(f'<div class="npk-card p-card">', unsafe_allow_html=True)
                    st.markdown(f'<p class="metric-label">Phosphorus-Loving Crops</p>', unsafe_allow_html=True)
                    p_category = npk_categories.get('phosphorus', 'Unknown')
                    p_crops = []
                    
                    if p_category == 'Low':
                        p_crops = ['Carrots', 'Potatoes', 'Garlic', 'Onions']
                    elif p_category == 'Medium':
                        p_crops = ['Beets', 'Radishes', 'Turnips', 'Strawberries']
                    elif p_category == 'High':
                        p_crops = ['Sunflowers', 'Soybeans', 'Peanuts', 'Flax']
                    elif p_category == 'Very High':
                        p_crops = ['Fruit Trees', 'Grapes', 'Berries', 'Melons']
                    
                    for crop in p_crops:
                        st.markdown(f'<p class="npk-crop-item">• {crop}</p>', unsafe_allow_html=True)
                    
                    st.markdown(f'<p class="npk-category {p_category.lower().replace(" ", "-")}-category">Current Level: {p_category}</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_k:
                    st.markdown(f'<div class="npk-card k-card">', unsafe_allow_html=True)
                    st.markdown(f'<p class="metric-label">Potassium-Loving Crops</p>', unsafe_allow_html=True)
                    k_category = npk_categories.get('potassium', 'Unknown')
                    k_crops = []
                    
                    if k_category == 'Low':
                        k_crops = ['Lettuce', 'Spinach', 'Herbs', 'Peas']
                    elif k_category == 'Medium':
                        k_crops = ['Beans', 'Eggplant', 'Peppers', 'Cucumbers']
                    elif k_category == 'High':
                        k_crops = ['Tomatoes', 'Potatoes', 'Sweet Potatoes', 'Squash']
                    elif k_category == 'Very High':
                        k_crops = ['Bananas', 'Citrus Fruits', 'Root Vegetables', 'Avocados']
                    
                    for crop in k_crops:
                        st.markdown(f'<p class="npk-crop-item">• {crop}</p>', unsafe_allow_html=True)
                    
                    st.markdown(f'<p class="npk-category {k_category.lower().replace(" ", "-")}-category">Current Level: {k_category}</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # NPK Balancing Tips
                st.markdown('<h3 class="sub-header">NPK Balancing Tips</h3>', unsafe_allow_html=True)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                
                # Generate balancing tips based on NPK categories
                balancing_tips = []
                
                # Nitrogen tips
                if npk_categories.get('nitrogen') == 'Low':
                    balancing_tips.append("<b>Increase Nitrogen:</b> Add composted manure, blood meal, or plant nitrogen-fixing cover crops like legumes.")
                elif npk_categories.get('nitrogen') == 'Very High':
                    balancing_tips.append("<b>Reduce Nitrogen:</b> Plant heavy nitrogen-feeding crops like corn or cabbage. Avoid adding nitrogen-rich fertilizers.")
                
                # Phosphorus tips
                if npk_categories.get('phosphorus') == 'Low':
                    balancing_tips.append("<b>Increase Phosphorus:</b> Add bone meal, rock phosphate, or fish meal to your soil.")
                elif npk_categories.get('phosphorus') == 'Very High':
                    balancing_tips.append("<b>Manage Phosphorus:</b> Avoid phosphorus-rich fertilizers and consider planting phosphorus-hungry crops.")
                
                # Potassium tips
                if npk_categories.get('potassium') == 'Low':
                    balancing_tips.append("<b>Increase Potassium:</b> Add wood ash, seaweed, or compost rich in banana peels to your soil.")
                elif npk_categories.get('potassium') == 'Very High':
                    balancing_tips.append("<b>Manage Potassium:</b> Plant potassium-loving crops like tomatoes and potatoes. Avoid adding potassium-rich amendments.")
                
                # Add general tip if all levels are medium
                if all(cat == 'Medium' for cat in npk_categories.values()):
                    balancing_tips.append("<b>Maintain Balance:</b> Your NPK levels are well-balanced. Continue with regular composting and crop rotation to maintain soil health.")
                
                # Display tips
                for tip in balancing_tips:
                    st.markdown(f'<p>{tip}</p>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
            
            # General Crop Recommendations
            st.markdown('<h3 class="sub-header">General Crop Recommendations</h3>', unsafe_allow_html=True)
            
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
    
    # Help & Info Tab
    with tab7:
        st.markdown('<h2 class="sub-header">Help & Information</h2>', unsafe_allow_html=True)
        
        # About the Dashboard
        st.markdown('<h3>About the Dashboard</h3>', unsafe_allow_html=True)
        st.markdown("""
        This Soil Health Dashboard provides real-time monitoring and analysis of soil parameters to help farmers make informed decisions about their crops and soil management practices.
        
        The dashboard collects data from soil sensors, processes it, and provides insights about soil fertility, NPK levels, crop recommendations, and soil improvement suggestions.
        """)
        
        # How to Use
        st.markdown('<h3>How to Use</h3>', unsafe_allow_html=True)
        st.markdown("""
        1. **Connect to Sensor**: Use the sidebar to connect to your ESP32 soil sensor or use the simulation mode.
        2. **Collect Data**: Click the 'Collect Data' button to gather soil parameter readings.
        3. **View Analysis**: Explore the different tabs to see soil health analysis, NPK levels, crop recommendations, and more.
        4. **Historical Data**: Track your soil health over time in the History tab.
        5. **AI Analysis**: Use the Mistral AI analysis for deeper insights about your soil health.
        """)
        
        # Soil Parameters Explained
        st.markdown('<h3>Soil Parameters Explained</h3>', unsafe_allow_html=True)
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h4>Temperature</h4>', unsafe_allow_html=True)
            st.markdown("""
            Soil temperature affects seed germination, plant growth, and microbial activity. Most crops prefer soil temperatures between 18°C and 24°C (65°F to 75°F).
            
            **Optimal Range**: 18°C - 24°C
            """)
            
            st.markdown('<h4>Moisture</h4>', unsafe_allow_html=True)
            st.markdown("""
            Soil moisture is critical for plant growth, nutrient uptake, and overall soil health. Different crops have different moisture requirements.
            
            **Optimal Range**: 30% - 60%
            """)
            
            st.markdown('<h4>NPK Levels</h4>', unsafe_allow_html=True)
            st.markdown("""
            NPK refers to the three primary nutrients essential for plant growth:
            
            **Nitrogen (N)**: Essential for leaf growth and green vegetation. Deficiency causes yellowing of leaves.
            - **Low**: 0-30 mg/kg
            - **Medium**: 31-60 mg/kg
            - **High**: 61-90 mg/kg
            - **Very High**: >90 mg/kg
            
            **Phosphorus (P)**: Important for root development, flowering, and fruiting. Deficiency causes stunted growth.
            - **Low**: 0-10 mg/kg
            - **Medium**: 11-20 mg/kg
            - **High**: 21-30 mg/kg
            - **Very High**: >30 mg/kg
            
            **Potassium (K)**: Helps in overall plant health and disease resistance. Deficiency causes brown edges on leaves.
            - **Low**: 0-100 mg/kg
            - **Medium**: 101-200 mg/kg
            - **High**: 201-300 mg/kg
            - **Very High**: >300 mg/kg
            """)
        
        with col2:
            st.markdown('<h4>pH Level</h4>', unsafe_allow_html=True)
            st.markdown("""
            Soil pH affects nutrient availability to plants. Most crops prefer slightly acidic to neutral soil (pH 6.0 to 7.0).
            
            **Optimal Range**: 6.0 - 7.0
            """)
            
            st.markdown('<h4>Fertility</h4>', unsafe_allow_html=True)
            st.markdown("""
            Soil fertility refers to the soil's ability to supply essential nutrients to plants. It's influenced by organic matter content, microbial activity, and nutrient levels.
            
            **Categories**:
            - Poor: Low nutrient availability
            - Fair: Moderate nutrient availability
            - Good: Adequate nutrient availability
            - Excellent: High nutrient availability
            """)
            
            st.markdown('<h4>NPK Health Percentage</h4>', unsafe_allow_html=True)
            st.markdown("""
            The NPK health percentage indicates how optimal your soil's NPK levels are for general plant growth:
            
            - **90-100%**: Excellent - Optimal levels for most plants
            - **70-89%**: Good - Suitable for most plants
            - **50-69%**: Fair - May need some amendments
            - **Below 50%**: Poor - Requires significant amendments
            
            Different plants have different NPK requirements, so check the NPK-specific recommendations for your intended crops.
            """)
        
        # Understanding NPK Analysis
        st.markdown('<h3>Understanding NPK Analysis</h3>', unsafe_allow_html=True)
        st.markdown("""
        The NPK Analysis section provides detailed information about your soil's macronutrient levels:
        
        1. **NPK Values**: The actual measured values of Nitrogen, Phosphorus, and Potassium in your soil.
        
        2. **Categories**: Each nutrient is categorized as Low, Medium, High, or Very High based on standard agricultural ranges.
        
        3. **Health Percentages**: Indicates how close each nutrient is to its optimal range for general plant growth.
        
        4. **Recommendations**: Specific suggestions for managing each nutrient level, whether it needs to be increased, decreased, or maintained.
        
        5. **NPK-Specific Crop Recommendations**: Suggestions for crops that would thrive with your current NPK levels.
        
        Remember that different plants have different nutrient requirements. Use the NPK-specific recommendations in the Recommendations tab to choose crops that match your soil's nutrient profile.
        """)
        
        # Troubleshooting
        st.markdown('<h3>Troubleshooting</h3>', unsafe_allow_html=True)
        st.markdown("""
        **Sensor Connection Issues**:
        - Ensure your ESP32 is properly powered and connected.
        - Check that the correct COM port is selected.
        - Try disconnecting and reconnecting the device.
        
        **Data Collection Problems**:
        - Verify that the sensors are properly inserted into the soil.
        - Ensure the soil is not too dry or too wet for accurate readings.
        - Try collecting data from different spots in your field.
        
        **NPK Reading Issues**:
        - Make sure the NPK sensor probes are clean and free from debris.
        - Calibrate your NPK sensor according to manufacturer instructions if readings seem inaccurate.
        - Take multiple readings from different soil depths for more accurate results.
        
        **For additional help, contact support at: support@soilhealth.com**
        """)

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