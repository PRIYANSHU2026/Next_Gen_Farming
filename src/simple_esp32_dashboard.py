import streamlit as st
import requests
import pandas as pd
import time
import threading
import json
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="ESP32 Sensor Dashboard", layout="centered")

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

# Main app
st.title("🌱 ESP32 Sensor Dashboard")

# Connection section
st.subheader("📡 ESP32 Connection")

# Show connection status
if st.session_state.connected:
    st.success(f"✅ Connected to: {st.session_state.esp32_url}")
    
    # Disconnect button
    if st.button("Disconnect"):
        disconnect_esp32()
        st.rerun()
else:
    # Input for ESP32 API URL
    url = st.text_input("Enter ESP32 API URL", "http://10.90.0.244/readings")
    
    # Connect button
    if st.button("Connect"):
        if connect_to_esp32(url):
            st.rerun()

# Data display section
st.subheader("📊 Sensor Data")

if st.session_state.connected and st.session_state.current_data:
    # Display current data
    data = st.session_state.current_data
    
    # Create columns for sensor readings
    cols = st.columns(3)
    
    # Display temperature if available
    if 'temperature' in data or 'temperature_ds18b20' in data:
        temp = data.get('temperature', data.get('temperature_ds18b20', 0))
        cols[0].metric("Temperature", f"{temp:.1f} °C")
    
    # Display moisture if available
    if 'moisture' in data:
        cols[1].metric("Moisture", f"{data['moisture']:.1f}%")
    
    # Display pH if available
    if 'ph' in data:
        cols[2].metric("pH", f"{data['ph']:.1f}")
    
    # Create another row for additional sensors
    if 'nitrogen' in data or 'phosphorus' in data or 'potassium' in data:
        cols = st.columns(3)
        
        if 'nitrogen' in data:
            cols[0].metric("Nitrogen", f"{data['nitrogen']:.1f} mg/kg")
        
        if 'phosphorus' in data:
            cols[1].metric("Phosphorus", f"{data['phosphorus']:.1f} mg/kg")
        
        if 'potassium' in data:
            cols[2].metric("Potassium", f"{data['potassium']:.1f} mg/kg")
    
    # Show raw data in expandable section
    with st.expander("Raw Data"):
        st.json(data)
    
    # Show in table
    st.subheader("📋 Data Table")
    df = pd.DataFrame([data])
    st.dataframe(df)
    
    # Add refresh button
    if st.button("Refresh Data"):
        data, error = fetch_data(st.session_state.esp32_url)
        if data:
            st.session_state.current_data = data
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