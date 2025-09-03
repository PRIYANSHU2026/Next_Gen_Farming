# Next Gen Farming: Soil Health Prediction System
**An IoT-based system to predict soil fertility and provide plant recommendations**

![The-5-Components-of-Healthy-Soil-1024x682](https://user-images.githubusercontent.com/103903785/235441777-7f0856c5-a7e6-4fc8-96f5-0f4d70594c04.jpg)

## Overview

The Next Gen Farming Soil Health Prediction System is an IoT-based solution that uses ESP32 microcontrollers to collect soil data, predict soil fertility, and provide plant recommendations. The system combines hardware sensors, machine learning, and an intuitive GUI dashboard to help farmers make informed decisions about their soil health and crop selection.

## Project Structure

```
Soil-Quality-Fertility-Prediction/
├── .gitignore               # Git ignore file
├── main.py                  # Main entry point for the application
├── setup.py                 # Package setup file
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── Data/                    # Data directory
│   ├── Raw Data.csv         # Original dataset
│   └── Soil Fertility Data (Modified Data).csv  # Modified dataset
├── ESP32_Soil_Sensor_Code/  # ESP32 firmware
│   └── ESP32_Soil_Sensor_Code.ino  # Arduino code for ESP32
├── notebooks/               # Jupyter notebooks
│   ├── Soil_Fertility.ipynb         # Soil fertility analysis notebook
│   └── Soil_Quality_ML_serve.ipynb  # ML model training notebook
├── src/                     # Source code
│   ├── soil_health_dashboard.py     # Dashboard implementation
│   ├── soil_health_streamlit.py     # Streamlit UI implementation
│   ├── api/                         # API integrations
│   │   ├── mistral_soil_analysis.py # Mistral AI integration
│   │   └── llm_soil_analysis.py     # LLM integration
│   ├── models/                      # ML models
│   │   └── plant_recommendation.py  # Plant recommendation system
│   └── utils/                       # Utility modules
│       └── soil_health_interface.py # ESP32 interface
└── tests/                   # Test directory
```

## System Components

### 1. ESP32 Firmware

The ESP32 microcontroller reads soil moisture and temperature data from sensors, then uses a simple prediction model to estimate NPK (Nitrogen, Phosphorus, Potassium) values. This data is sent as JSON over a serial connection to the Python interface.

**Features:**
- Soil moisture sensing using capacitive sensors
- Temperature measurement using DS18B20 sensors
- NPK prediction based on moisture and temperature correlations
- JSON-formatted data transmission

### 2. Python Interface (src/utils/soil_health_interface.py)

This module serves as the bridge between the ESP32 hardware and the machine learning model. It handles serial communication, data processing, and integration with the pre-trained soil fertility prediction model.

**Features:**
- Automatic ESP32 port detection
- Serial data reception and parsing
- Integration with pre-trained RandomForest model
- Parameter estimation for missing soil attributes

### 3. Streamlit Dashboard (src/soil_health_streamlit.py)

A Streamlit-based graphical user interface that visualizes soil health metrics, displays fertility predictions, and shows historical data trends.

**Features:**
- Real-time sensor data display
- Soil fertility classification visualization
- Historical data tracking and graphing
- Plant recommendations based on soil conditions
- Manual data entry option for testing without hardware
- Port selection for ESP32 connectivity

### 4. Plant Recommendation System (src/models/plant_recommendation.py)

This module provides crop recommendations and soil improvement suggestions based on the soil fertility classification and sensor data.

**Features:**
- Crop recommendations based on soil parameters
- Soil improvement suggestions
- Optimal growing conditions for various crops

### 5. LLM Soil Analysis (src/api/mistral_soil_analysis.py)

Integrates with Mistral AI's API to provide detailed soil health analysis and recommendations using large language models.

**Features:**
- Detailed soil health analysis
- Specific insights about NPK levels
- Recommendations for improving soil health
- Suggestions for optimal crops
- Sustainable farming practices
- Streaming response capability

## Requirements

### Software Requirements
- Python 3.8 or higher
- Streamlit
- pyserial
- matplotlib
- numpy
- pandas
- scikit-learn
- seaborn
- requests (for API integration)
- Mistral AI API key (for LLM analysis)

### Hardware Requirements
- ESP32 microcontroller
- Soil moisture sensor
- Temperature and humidity sensor (DHT22)
- pH sensor (optional)
- NPK sensor (optional)

## Data
### Data Source
Raw Data is published on [kaggle](https://www.kaggle.com/datasets/rahuljaiswalonkaggle/soil-fertility-dataset)
- [Original Dataset](https://github.com/iurwpoietknckvjndfsm-gndvkd/ML---Soil-Quality/blob/main/Data/Raw%20Data.csv)
- [Modified Dataset](https://github.com/iurwpoietknckvjndfsm-gndvkd/ML---Soil-Quality/blob/main/Data/Soil%20Fertility%20Data%20(Modified%20Data).csv)

### Atrributes
- N - ratio of Nitrogen (NH4+) content in soil 
- P - ratio of Phosphorous (P) content in soil 
- K - ratio of Potassium (K) content in soil 
- ph - soil acidity (pH)
- ec - electrical conductivity
- oc - organic carbon
- S - sulfur (S)
- zn - Zinc (Zn)
- fe - Iron (Fe)
- cu - Copper (Cu)
- Mn - Manganese (Mn)
- B - Boron (B)
- fertility: categorical (0 "Less Fertile", 1 "Fertile", 2 "Highly Fertile")

## Resource/Situational Constraints
- Lack of Data about our region
- Lack of some classes
- Lack of experience in agriculutral science

## Connectivity Instructions

### ESP32 to Python Interface Connection

1. **Upload the ESP32 Code**:
   - Open the `ESP32_Soil_Sensor_Code.ino` file in the Arduino IDE
   - Connect your ESP32 to your computer via USB
   - Select the correct board and port in the Arduino IDE
   - Upload the code to your ESP32

2. **Run the Streamlit Interface**:
   - Navigate to the project directory
   - Run `streamlit run soil_health_streamlit.py`
   - In the Streamlit interface, select the correct port from the dropdown menu
   - Click "Connect to ESP32"

3. **Manual Data Entry**:
   - If you don't have the hardware, you can use the "Manual Entry" tab
   - Enter soil parameter values manually
   - Click "Submit Data" to process the values

4. **Troubleshooting**:
   - Ensure the baud rate in the ESP32 code (115200) matches the Python interface
   - Check that the ESP32 is properly connected and powered
   - Verify that you have the correct port selected in the Streamlit interface

## Process followed 
- Searched for a dataset from another country
- Creating additional instances for the least appearing class 
- monitored by an experienced specialist

## Machine Learning Model
1. Import libraries and modules
2. Naive approach with RandomForestClassifier and raw data (accuracy = 88%)
3. Data Exploration for modified data
4. Choose a model 
    - SupportVectorClassifier 
    - RandomForestClassifier 
    - GaussianNB 
    - KNeighborsClassifier 
    - DecisionTreeClassifier
5. GridSearch
6. Train a RandomForestClassifier (accuracy = 97%)
7. Save the model

## Setup Instructions

### ESP32 Setup
- Connect the soil moisture sensor to pin 34
- Connect the DS18B20 temperature sensor to pin 4
- Upload the ESP32 firmware using Arduino IDE

### Running the Dashboard

```bash
python soil_health_dashboard.py
```

The dashboard will automatically detect and connect to the ESP32 device. If no device is found, it will run in simulation mode with randomly generated data.

### Dashboard Features

1. **Dashboard Tab:**
   - Current sensor readings
   - Soil fertility classification
   - Trend graphs for all parameters

2. **History Tab:**
   - Historical data in tabular format
   - Data export functionality

3. **Recommendations Tab:**
   - Crop recommendations based on soil fertility
   - Soil improvement suggestions

4. **LLM Analysis Tab:**
   - Detailed soil health analysis using OpenAI's API
   - Streaming or complete analysis options
   - Requires OpenAI API key

## Technical Details

### NPK Prediction

Since actual NPK sensors are not used, the system estimates NPK values based on soil moisture and temperature using general correlations:
- Higher moisture generally correlates with higher nutrient availability
- Temperature affects nutrient solubility and microbial activity

### LLM Integration

The LLM analysis module uses OpenAI's API to provide detailed soil health analysis and recommendations. It takes the following inputs:
- Soil sensor data (temperature, moisture, NPK)
- Fertility prediction results
- Crop recommendations

And provides:
- Detailed soil health analysis
- Specific insights about NPK levels
- Recommendations for improving soil health
- Suggestions for optimal crops
- Sustainable farming practices
