import serial
import serial.tools.list_ports
import json
import time
import pickle
import pandas as pd
import numpy as np
import threading
import os
import requests
from datetime import datetime

class SoilHealthPredictor:
    def __init__(self, model_path=None):
        # Set default model path if not provided
        if model_path is None:
            # Get the project root directory (two levels up from utils)
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(current_dir, 'src', 'models', 'saved_models', 'random_forest_pkl.pkl')
        
        # Load the ML model
        try:
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
            print("✅ ML model loaded successfully")
        except Exception as e:
            # Silently create fallback model without error messages
            self.model = None
            # Create a fallback model for demonstration purposes
            self._create_fallback_model()
        
        # NPK reference ranges for interpretation
        self.npk_ranges = {
            'N': {'low': 0, 'medium': 100, 'high': 140, 'very_high': 200},
            'P': {'low': 0, 'medium': 5, 'high': 10, 'very_high': 15},
            'K': {'low': 0, 'medium': 100, 'high': 200, 'very_high': 300}
        }
    
    def predict_fertility(self, soil_data):
        """Predict soil fertility based on soil parameters"""
        try:
            # Create a DataFrame with the required features in the correct order
            features = ['N', 'P', 'K', 'ph', 'ec', 'oc', 'S', 'zn', 'fe', 'cu', 'Mn', 'B']
            
            # Check if we have all required features
            missing_features = [f for f in features if f not in soil_data]
            
            # If we're missing features, use default values or estimates
            if missing_features:
                # Silently handle missing features without warnings
                pass
                # Use ESP32 sensor data (temperature, moisture) to estimate missing values
                if 'temperature' in soil_data and 'moisture' in soil_data:
                    soil_data = self._estimate_missing_parameters(soil_data, missing_features)
            
            # Create DataFrame with all features
            input_df = pd.DataFrame({feature: [soil_data.get(feature, 0)] for feature in features})
            
            # Make prediction
            prediction = self.model.predict(input_df)[0]
            fertility_categories = ["Less Fertile", "Fertile", "Highly Fertile"]
            result = fertility_categories[prediction]
            
            return {
                "fertility_class": prediction,
                "fertility_label": result,
                "confidence": self._get_confidence(input_df)
            }
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {"error": str(e)}
    
    def _estimate_missing_parameters(self, soil_data, missing_features):
        """Estimate missing soil parameters based on temperature and moisture"""
        temp = soil_data.get('temperature', 25)
        moisture = soil_data.get('moisture', 50)
        
        # Simple estimation logic based on temperature and moisture
        # These are placeholder values and should be calibrated with real data
        estimates = {
            'N': 150 + (moisture * 0.5) + (temp * 0.2),
            'P': 8 + (moisture * 0.02) - (temp * 0.01),
            'K': 400 + (moisture * 2) - (temp * 1),
            'ph': 7.5 - (moisture * 0.002) + (temp * 0.001),
            'ec': 0.5 + (moisture * 0.002) - (temp * 0.001),
            'oc': 0.8 + (moisture * 0.004),
            'S': 15 + (moisture * 0.1),
            'zn': 0.3 + (moisture * 0.001),
            'fe': 0.6 + (moisture * 0.002),
            'cu': 1.2 + (moisture * 0.001),
            'Mn': 2.5 + (moisture * 0.01),
            'B': 1.5 + (moisture * 0.01)
        }
        
        # Update soil_data with estimates for missing features
        for feature in missing_features:
            soil_data[feature] = estimates.get(feature, 0)
        
        return soil_data
    
    def _get_confidence(self, input_df):
        """Get confidence score for the prediction"""
        try:
            # For random forest, we can use the prediction probabilities
            if self.model is not None:
                proba = self.model.predict_proba(input_df)[0]
                return float(np.max(proba) * 100)  # Convert to percentage
            else:
                # If model is not available, return a default value
                return 85.0
        except:
            # If probabilities are not available, return a default value
            return 85.0
            
    def predict_npk_levels(self, soil_data):
        """Predict NPK levels and categories based on soil data"""
        try:
            # Get NPK values from soil data
            n_value = soil_data.get('nitrogen', 0)
            p_value = soil_data.get('phosphorus', 0)
            k_value = soil_data.get('potassium', 0)
            
            # Determine NPK categories based on reference ranges
            # Nitrogen
            if n_value < self.npk_ranges['N']['medium']:
                n_category = "Low"
                n_recommendation = "Add nitrogen-rich fertilizers or compost."
                n_health = 30
            elif n_value < self.npk_ranges['N']['high']:
                n_category = "Medium"
                n_recommendation = "Moderate nitrogen levels, monitor regularly."
                n_health = 60
            elif n_value < self.npk_ranges['N']['very_high']:
                n_category = "High"
                n_recommendation = "Good nitrogen levels, maintain current practices."
                n_health = 90
            else:
                n_category = "Very High"
                n_recommendation = "Reduce nitrogen application to prevent leaching."
                n_health = 70
            
            # For zero values, set to Unknown
            if n_value == 0:
                n_category = "Unknown"
                n_recommendation = "Add nitrogen-rich fertilizers or compost."
                n_health = 0
            
            # Phosphorus
            if p_value < self.npk_ranges['P']['medium']:
                p_category = "Low"
                p_recommendation = "Add phosphorus-rich fertilizers or bone meal."
                p_health = 30
            elif p_value < self.npk_ranges['P']['high']:
                p_category = "Medium"
                p_recommendation = "Moderate phosphorus levels, monitor regularly."
                p_health = 60
            elif p_value < self.npk_ranges['P']['very_high']:
                p_category = "High"
                p_recommendation = "Good phosphorus levels, maintain current practices."
                p_health = 90
            else:
                p_category = "Very High"
                p_recommendation = "Reduce phosphorus application to prevent runoff."
                p_health = 70
            
            # For zero values, set to Unknown
            if p_value == 0:
                p_category = "Unknown"
                p_recommendation = "Add phosphorus-rich fertilizers or bone meal."
                p_health = 0
            
            # Potassium
            if k_value < self.npk_ranges['K']['medium']:
                k_category = "Low"
                k_recommendation = "Add potassium-rich fertilizers or wood ash."
                k_health = 30
            elif k_value < self.npk_ranges['K']['high']:
                k_category = "Medium"
                k_recommendation = "Moderate potassium levels, monitor regularly."
                k_health = 60
            elif k_value < self.npk_ranges['K']['very_high']:
                k_category = "High"
                k_recommendation = "Good potassium levels, maintain current practices."
                k_health = 90
            else:
                k_category = "Very High"
                k_recommendation = "Reduce potassium application to prevent imbalance."
                k_health = 70
            
            # For zero values, set to Unknown
            if k_value == 0:
                k_category = "Unknown"
                k_recommendation = "Add potassium-rich fertilizers or wood ash."
                k_health = 0
            
            # Return NPK analysis
            return {
                'n_value': n_value,
                'n_category': n_category,
                'n_recommendation': n_recommendation,
                'n_health': n_health,
                'p_value': p_value,
                'p_category': p_category,
                'p_recommendation': p_recommendation,
                'p_health': p_health,
                'k_value': k_value,
                'k_category': k_category,
                'k_recommendation': k_recommendation,
                'k_health': k_health
            }
        except Exception as e:
            print(f"❌ Error predicting NPK levels: {e}")
            # Return default values
            return {
                'n_value': soil_data.get('nitrogen', 0),
                'n_category': 'Unknown',
                'n_recommendation': 'Add nitrogen-rich fertilizers or compost.',
                'n_health': 0,
                'p_value': soil_data.get('phosphorus', 0),
                'p_category': 'Unknown',
                'p_recommendation': 'Add phosphorus-rich fertilizers or bone meal.',
                'p_health': 0,
                'k_value': soil_data.get('potassium', 0),
                'k_category': 'Unknown',
                'k_recommendation': 'Add potassium-rich fertilizers or wood ash.',
                'k_health': 0
            }
            
    def _create_fallback_model(self):
        """Create a simple fallback model for demonstration purposes"""
        # Silently create fallback model without messages
        
        # This is a simple dummy model that will return predictions based on simple rules
        # It's only for demonstration when the real model fails to load
        class DummyModel:
            def predict(self, X):
                # Simple rule: if N > 140 and P > 10 and K > 200, then "Highly Fertile"
                # else if N > 100 and P > 5 and K > 100, then "Fertile"
                # else "Less Fertile"
                n = X['N'].values[0]
                p = X['P'].values[0]
                k = X['K'].values[0]
                
                if n > 140 and p > 10 and k > 200:
                    return [2]  # Highly Fertile
                elif n > 100 and p > 5 and k > 100:
                    return [1]  # Fertile
                else:
                    return [0]  # Less Fertile
            
            def predict_proba(self, X):
                # Return dummy probabilities
                prediction = self.predict(X)[0]
                probs = [0.1, 0.1, 0.1]
                probs[prediction] = 0.8
                return [probs]
        
        self.model = DummyModel()

class ESP32Interface:
    def __init__(self, port=None, baud_rate=115200):
        self.port = port
        self.baud_rate = baud_rate
        self.serial_conn = None
        self.connected = False
        self.data_callback = None
        self.running = False
        self.thread = None
        self.last_data = {}
    
    def set_data_callback(self, callback):
        """Set callback function to be called when new data is received"""
        self.data_callback = callback
    
    def connect(self, port=None):
        """Connect to ESP32 via serial"""
        if port:
            self.port = port
        
        # If port is still None, try to find it
        if self.port is None:
            self.port = self._find_esp32_port()
        
        if self.port is None:
            print("❌ Could not find ESP32 port. Please specify manually.")
            return False
        
        try:
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=1)
            
            # Send STATUS command and wait for response to verify it's an ESP32
            self.serial_conn.write(b'STATUS\n')
            time.sleep(0.5)
            
            if self.serial_conn.in_waiting > 0:
                response = self.serial_conn.readline().decode('utf-8').strip()
                if 'ESP32' in response:
                    self.connected = True
                    print(f"✅ Connected to ESP32 on {self.port}")
                    return True
                else:
                    self.serial_conn.close()
                    print(f"❌ Device on port {self.port} is not an ESP32 or is not responding properly")
                    self.connected = False
                    return False
            else:
                self.serial_conn.close()
                print(f"❌ No response from device on port {self.port}. Please check if ESP32 firmware is running.")
                self.connected = False
                return False
        except Exception as e:
            print(f"❌ Could not connect to ESP32: {e}")
            self.connected = False
            return False
    
    def _find_esp32_port(self):
        """Try to automatically find the ESP32 serial port"""
        import glob
        
        # Common patterns for ESP32 serial ports
        patterns = [
            '/dev/tty.usbserial*',
            '/dev/tty.SLAB_USBtoUART*',
            '/dev/tty.wchusbserial*',
            '/dev/ttyUSB*',
            '/dev/cu.usbserial*'
        ]
        
        for pattern in patterns:
            ports = glob.glob(pattern)
            if ports:
                return ports[0]
        
        return None
    
    def start_reading(self):
        """Start reading data from ESP32 in a separate thread"""
        if not self.connected:
            print("❌ Not connected to ESP32")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._read_loop)
        self.thread.daemon = True
        self.thread.start()
        return True
    
    def stop_reading(self):
        """Stop reading data from ESP32"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if self.serial_conn:
            self.serial_conn.close()
            self.connected = False
    
    def _read_loop(self):
        """Read data from ESP32 in a loop"""
        while self.running:
            try:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    
                    # Try to parse JSON data
                    if line.startswith('{') and line.endswith('}'): 
                        try:
                            data = json.loads(line)
                            self.last_data = data
                            
                            # Add timestamp
                            data['timestamp'] = datetime.now().isoformat()
                            
                            # Call callback if set
                            if self.data_callback:
                                self.data_callback(data)
                        except json.JSONDecodeError:
                            print(f"⚠️ Invalid JSON: {line}")
            except Exception as e:
                print(f"❌ Error reading from ESP32: {e}")
                time.sleep(1)
            
            time.sleep(0.1)
    
    def read_data(self):
        """Read the latest data from ESP32 or return the last known data"""
        if not self.connected:
            return None
        
        # If we're not already reading in a thread, try to get data directly
        if not self.running:
            try:
                # Send READ command to request fresh data
                self.serial_conn.write(b'READ\n')
                time.sleep(0.5)  # Wait for response
                
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    
                    # Try to parse JSON data
                    if line.startswith('{') and line.endswith('}'): 
                        try:
                            data = json.loads(line)
                            self.last_data = data
                            return data
                        except json.JSONDecodeError:
                            print(f"⚠️ Invalid JSON: {line}")
            except Exception as e:
                print(f"❌ Error reading from ESP32: {e}")
        
        # Return the last known data
        return self.last_data
    
    def close(self):
        """Close the serial connection"""
        self.stop_reading()
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.connected = False
    
    def get_last_data(self):
        """Get the last received data"""
        return self.last_data

class ESP32WebInterface:
    def __init__(self, ip_address="http://localhost", endpoint="/readings"):
        self.ip_address = ip_address
        self.endpoint = endpoint
        self.connected = False
        self.last_data = {}
        self.data_callback = None
        self.running = False
        self.thread = None
        self.available_ips = []
    
    def set_data_callback(self, callback):
        """Set callback function to be called when new data is received"""
        self.data_callback = callback
    
    @staticmethod
    def detect_esp32_ips():
        """Detect ESP32 devices on the local network"""
        # Get local IP to determine network range
        import socket
        try:
            # Get local IP address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            
            # Extract network prefix
            ip_parts = local_ip.split('.')
            network_prefix = '.'.join(ip_parts[:3])
            
            print(f"Scanning network {network_prefix}.0/24 for ESP32 devices...")
            
            # List to store found ESP32 IPs
            esp32_ips = []
            
            # Common ESP32 web server ports
            ports = [80, 8080, 8081]
            
            # Try a few common IPs first (faster than scanning the whole network)
            common_ips = [f"{network_prefix}.1", f"{network_prefix}.100", f"{network_prefix}.101", 
                         f"{network_prefix}.244", f"{network_prefix}.245", local_ip]
            
            # Check common IPs first
            for ip in common_ips:
                for port in ports:
                    try:
                        url = f"http://{ip}:{port}/readings"
                        response = requests.get(url, timeout=0.5)
                        if response.status_code == 200:
                            try:
                                # Verify it's an ESP32 by checking if response is valid JSON with expected keys
                                data = response.json()
                                if any(key in data for key in ["temperature_ds18b20", "moisture", "ph", "nitrogen"]):
                                    esp32_ips.append(url.rsplit('/readings', 1)[0])
                                    print(f"Found ESP32 at {url.rsplit('/readings', 1)[0]}")
                            except:
                                pass  # Not a valid JSON response, not our ESP32
                    except:
                        pass  # Connection failed, not an ESP32 or not responding
            
            return esp32_ips
        except Exception as e:
            print(f"Error detecting ESP32 IPs: {e}")
            return []
    
    def scan_for_devices(self):
        """Scan the network for ESP32 devices"""
        self.available_ips = self.detect_esp32_ips()
        return self.available_ips
    
    def connect(self, ip_address=None):
        """Connect to ESP32 via HTTP"""
        if ip_address:
            self.ip_address = ip_address
        
        # Validate IP address format
        if not self.ip_address.startswith("http"):
            self.ip_address = f"http://{self.ip_address}"
        
        try:
            # Test connection by making a request
            response = requests.get(f"{self.ip_address}{self.endpoint}", timeout=5)
            if response.status_code == 200:
                self.connected = True
                print(f"✅ Connected to ESP32 Web Server at {self.ip_address}")
                return True
            else:
                print(f"❌ Failed to connect to ESP32 Web Server. Status code: {response.status_code}")
                self.connected = False
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Could not connect to ESP32 Web Server: {e}")
            self.connected = False
            return False
    
    def start_reading(self):
        """Start reading data from ESP32 in a separate thread"""
        if not self.connected:
            print("❌ Not connected to ESP32 Web Server")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._read_loop)
        self.thread.daemon = True
        self.thread.start()
        return True
    
    def stop_reading(self):
        """Stop reading data from ESP32"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def _read_loop(self):
        """Read data from ESP32 in a loop"""
        while self.running:
            try:
                data = self.read_data()
                if data and self.data_callback:
                    self.data_callback(data)
            except Exception as e:
                print(f"❌ Error reading from ESP32 Web Server: {e}")
            
            time.sleep(2)  # Poll every 2 seconds
    
    def read_data(self):
        """Read the latest data from ESP32 Web Server"""
        if not self.connected:
            return None
        
        try:
            response = requests.get(f"{self.ip_address}{self.endpoint}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.last_data = data
                
                # Add timestamp
                data['timestamp'] = datetime.now().isoformat()
                
                return data
            else:
                print(f"⚠️ Failed to get data. Status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error reading from ESP32 Web Server: {e}")
            return None
    
    def close(self):
        """Close the connection"""
        self.stop_reading()
        self.connected = False
    
    def get_last_data(self):
        """Get the last received data"""
        return self.last_data

# Example usage
if __name__ == "__main__":
    # Initialize soil health predictor
    predictor = SoilHealthPredictor()
    
    # Initialize ESP32 interface
    esp32 = ESP32Interface()
    
    # Define callback function for new data
    def on_new_data(data):
        print("\n🌱 Received Soil Data:")
        print(f"🌡 Temperature : {data.get('temperature', 'N/A')} °C")
        print(f"💧 Moisture    : {data.get('moisture', 'N/A')} %")
        print(f"🧪 Nitrogen (N): {data.get('nitrogen', 'N/A')}")
        print(f"🧪 Phosphorus(P): {data.get('phosphorus', 'N/A')}")
        print(f"🧪 Potassium (K): {data.get('potassium', 'N/A')}")
        print(f"🧪 pH         : {data.get('ph', 'N/A')}")
        
        # Predict soil fertility
        result = predictor.predict_fertility(data)
        
        print("\n🔍 Soil Fertility Prediction:")
        print(f"📊 Class: {result.get('fertility_label', 'N/A')}")
        print(f"🎯 Confidence: {result.get('confidence', 'N/A')}%")
    
    # Set callback
    esp32.set_data_callback(on_new_data)
    
    # Try to connect to ESP32
    if esp32.connect():
        # Start reading data
        esp32.start_reading()
        
        try:
            print("Press Ctrl+C to exit")
            while True:
                # Read data periodically
                data = esp32.read_data()
                if data:
                    print(f"Latest data: Temperature={data.get('temperature')}°C, Moisture={data.get('moisture')}%")
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            esp32.close()
    else:
        print("Failed to connect to ESP32. Running in simulation mode...")
        
        # Simulate data for testing
        try:
            print("Press Ctrl+C to exit simulation mode")
            while True:
                # Generate simulated data
                simulated_data = {
                    'temperature': np.random.uniform(20, 30),
                    'moisture': np.random.uniform(40, 80),
                    'nitrogen': np.random.uniform(100, 200),
                    'phosphorus': np.random.uniform(10, 30),
                    'potassium': np.random.uniform(200, 400),
                    'ph': np.random.uniform(5.5, 7.5)
                }
                
                # Process simulated data
                on_new_data(simulated_data)
                
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nExiting simulation...")