import serial
import serial.tools.list_ports
import json
import time
import pickle
import pandas as pd
import numpy as np
import threading
import os
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
            print(f"❌ Error loading ML model: {e}")
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
                print(f"⚠️ Missing features: {missing_features}")
                # Use ESP32 sensor data (temperature, moisture) to estimate missing values
                if 'temperature' in soil_data and 'moisture' in soil_data:
                    soil_data = self._estimate_missing_parameters(soil_data, missing_features)
            
            # Create DataFrame with all features
            input_df = pd.DataFrame({feature: [soil_data.get(feature, 0)] for feature in features})
            
            # Make prediction
            prediction = self.model.predict(input_df)[0]
            fertility_categories = ["Less Fertile", "Fertile", "Highly Fertile"]
            result = fertility_categories[prediction]
            
            # Add NPK predictions
            npk_predictions = self.predict_npk_levels(soil_data)
            
            return {
                "fertility_class": prediction,
                "fertility_label": result,
                "confidence": self._get_confidence(input_df),
                **npk_predictions  # Include NPK predictions in the result
            }
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {"error": str(e)}
            
    def predict_npk_levels(self, soil_data):
        """Predict and categorize NPK levels based on sensor readings"""
        try:
            # Extract NPK values from soil data
            n_value = soil_data.get('nitrogen', soil_data.get('N', 0))
            p_value = soil_data.get('phosphorus', soil_data.get('P', 0))
            k_value = soil_data.get('potassium', soil_data.get('K', 0))
            
            # Categorize N level
            if n_value < self.npk_ranges['N']['medium']:
                n_category = "Low"
                n_recommendation = "Increase nitrogen with organic matter or nitrogen-rich fertilizers."
            elif n_value < self.npk_ranges['N']['high']:
                n_category = "Medium"
                n_recommendation = "Maintain current nitrogen levels with regular fertilization."
            elif n_value < self.npk_ranges['N']['very_high']:
                n_category = "High"
                n_recommendation = "Good nitrogen levels. Monitor to prevent excess."
            else:
                n_category = "Very High"
                n_recommendation = "Reduce nitrogen applications to prevent leaching and plant burn."
            
            # Categorize P level
            if p_value < self.npk_ranges['P']['medium']:
                p_category = "Low"
                p_recommendation = "Add phosphorus-rich amendments like bone meal or rock phosphate."
            elif p_value < self.npk_ranges['P']['high']:
                p_category = "Medium"
                p_recommendation = "Maintain phosphorus with balanced fertilization."
            elif p_value < self.npk_ranges['P']['very_high']:
                p_category = "High"
                p_recommendation = "Good phosphorus levels. Monitor to prevent excess."
            else:
                p_category = "Very High"
                p_recommendation = "Avoid additional phosphorus to prevent water pollution."
            
            # Categorize K level
            if k_value < self.npk_ranges['K']['medium']:
                k_category = "Low"
                k_recommendation = "Add potassium with wood ash, seaweed, or potassium sulfate."
            elif k_value < self.npk_ranges['K']['high']:
                k_category = "Medium"
                k_recommendation = "Maintain potassium with regular balanced fertilization."
            elif k_value < self.npk_ranges['K']['very_high']:
                k_category = "High"
                k_recommendation = "Good potassium levels. Monitor to prevent excess."
            else:
                k_category = "Very High"
                k_recommendation = "Reduce potassium applications to maintain balance with other nutrients."
            
            # Calculate health percentages (0-100%)
            n_health = min(100, max(0, (n_value / self.npk_ranges['N']['very_high']) * 100))
            p_health = min(100, max(0, (p_value / self.npk_ranges['P']['very_high']) * 100))
            k_health = min(100, max(0, (k_value / self.npk_ranges['K']['very_high']) * 100))
            
            # Return NPK predictions and recommendations
            return {
                "n_value": n_value,
                "n_category": n_category,
                "n_recommendation": n_recommendation,
                "n_health": round(n_health, 1),
                "p_value": p_value,
                "p_category": p_category,
                "p_recommendation": p_recommendation,
                "p_health": round(p_health, 1),
                "k_value": k_value,
                "k_category": k_category,
                "k_recommendation": k_recommendation,
                "k_health": round(k_health, 1),
            }
        except Exception as e:
            print(f"❌ NPK prediction error: {e}")
            return {}
    
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
            
    def _create_fallback_model(self):
        """Create a simple fallback model for demonstration purposes"""
        print("⚠️ Creating fallback model for demonstration purposes")
        
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