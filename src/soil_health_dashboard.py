import sys
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QComboBox, QFrame, QGridLayout, QTabWidget,
                             QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QMessageBox,
                             QLineEdit, QTextEdit, QScrollArea, QProgressBar)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor, QPalette, QPixmap

# Import matplotlib for plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import our soil health interface
from utils.soil_health_interface import SoilHealthPredictor, ESP32Interface
from models.plant_recommendation import PlantRecommendationSystem
from api.mistral_soil_analysis import MistralSoilAnalysis

# Define color constants
COLORS = {
    'primary': '#2E7D32',  # Dark green
    'secondary': '#81C784',  # Light green
    'accent': '#FF8F00',  # Amber
    'background': '#F1F8E9',  # Light farm green
    'text': '#33691E',  # Dark farm green
    'warning': '#F44336',  # Red
    'success': '#4CAF50',  # Green
    'info': '#2196F3',  # Blue
    'less_fertile': '#F44336',  # Red
    'fertile': '#FFC107',  # Yellow
    'highly_fertile': '#4CAF50',  # Green
    'low': '#F44336',  # Red for low NPK
    'medium': '#FFC107',  # Yellow for medium NPK
    'high': '#4CAF50',  # Green for high NPK
    'very_high': '#1B5E20',  # Dark green for very high NPK
    'border': '#8BC34A',  # Light green for borders
    'header': '#33691E',  # Dark green for headers
    'card': '#FFFFFF',  # White for card backgrounds
}

# Plant recommendations based on fertility class
PLANT_RECOMMENDATIONS = {
    0: [  # Less Fertile
        {"name": "Beans", "description": "Legumes that can fix nitrogen in soil"},
        {"name": "Peas", "description": "Adds nitrogen to soil while producing food"},
        {"name": "Clover", "description": "Cover crop that improves soil structure"},
        {"name": "Alfalfa", "description": "Deep roots break up compacted soil"},
        {"name": "Sunflowers", "description": "Tolerates poor soil conditions"}
    ],
    1: [  # Fertile
        {"name": "Tomatoes", "description": "Thrives in moderately fertile soil"},
        {"name": "Peppers", "description": "Produces well in balanced soil"},
        {"name": "Corn", "description": "Grows well with adequate nutrients"},
        {"name": "Squash", "description": "Productive in medium-fertility soil"},
        {"name": "Potatoes", "description": "Good yields in average soil conditions"}
    ],
    2: [  # Highly Fertile
        {"name": "Leafy Greens", "description": "Thrives in nutrient-rich soil"},
        {"name": "Broccoli", "description": "Heavy feeder that needs fertile soil"},
        {"name": "Cabbage", "description": "Requires high nitrogen levels"},
        {"name": "Cauliflower", "description": "Needs rich soil for best growth"},
        {"name": "Strawberries", "description": "Produces best in fertile conditions"}
    ]
}

# Matplotlib canvas for plots
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()

# Data collection thread
class DataCollectionThread(QThread):
    data_received = pyqtSignal(dict)
    connection_status = pyqtSignal(bool, str)
    
    def __init__(self):
        super().__init__()
        self.esp32 = ESP32Interface()
        self.predictor = SoilHealthPredictor()
        self.running = False
    
    def run(self):
        self.running = True
        
        # Try to connect to ESP32
        if self.esp32.connect():
            self.connection_status.emit(True, f"Connected to {self.esp32.port}")
            
            # Set callback for new data
            def on_new_data(data):
                # Add fertility prediction
                prediction = self.predictor.predict_fertility(data)
                data.update(prediction)
                self.data_received.emit(data)
            
            self.esp32.set_data_callback(on_new_data)
            self.esp32.start_reading()
            
            # Keep thread running
            while self.running:
                time.sleep(0.1)
                
            # Clean up
            self.esp32.stop_reading()
        else:
            self.connection_status.emit(False, "Failed to connect to ESP32")
            
            # Simulate data for testing if no ESP32 is connected
            while self.running:
                # Generate random data
                data = self._generate_test_data()
                
                # Add fertility prediction
                prediction = self.predictor.predict_fertility(data)
                data.update(prediction)
                
                # Emit data
                self.data_received.emit(data)
                
                # Wait before next update
                time.sleep(3)
    
    def stop(self):
        self.running = False
        self.wait()
    
    def _generate_test_data(self):
        """Generate random test data for simulation"""
        # Add some randomness but keep values in realistic ranges
        temp_base = 25 + np.sin(time.time() / 10) * 5
        moisture_base = 50 + np.sin(time.time() / 15) * 20
        
        return {
            'timestamp': datetime.now().isoformat(),
            'temperature': round(temp_base + np.random.normal(0, 1), 1),
            'moisture': round(moisture_base + np.random.normal(0, 3), 1),
            'nitrogen': round(150 + np.random.normal(0, 10), 1),
            'phosphorus': round(8 + np.random.normal(0, 1), 1),
            'potassium': round(400 + np.random.normal(0, 20), 1),
        }

# Main dashboard window
class SoilHealthDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Window properties
        self.setWindowTitle("Next Gen Farming - Soil Health Dashboard")
        self.setMinimumSize(1000, 700)
        
        # Data storage
        self.data_history = []
        self.max_history = 100  # Maximum number of data points to store
        
        # Initialize LLM analysis
        self.llm_analysis = MistralSoilAnalysis()
        self.current_llm_analysis = ""
        self.current_data = None
        self.current_fertility_prediction = None
        self.current_recommendations = None
        
        # Initialize UI
        self._init_ui()
        
        # Start data collection
        self.data_thread = DataCollectionThread()
        self.data_thread.data_received.connect(self.update_dashboard)
        self.data_thread.connection_status.connect(self.update_connection_status)
        self.data_thread.start()
    
    def _init_ui(self):
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header = self._create_header()
        main_layout.addWidget(header)
        
        # Create tabs
        self.tabs = QTabWidget()
        
        # Create tabs
        self.dashboard_tab = self._create_dashboard_tab()
        self.history_tab = self._create_history_tab()
        self.recommendations_tab = self._create_recommendations_tab()
        self.llm_analysis_tab = self._create_llm_analysis_tab()
        
        # Add tabs
        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        self.tabs.addTab(self.history_tab, "History")
        self.tabs.addTab(self.recommendations_tab, "Recommendations")
        self.tabs.addTab(self.llm_analysis_tab, "LLM Analysis")
        
        main_layout.addWidget(self.tabs)
        
        # Status bar at the bottom
        self.status_label = QLabel("Initializing...")
        main_layout.addWidget(self.status_label)
        
        # Set up update timer for plots
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)
        self.update_timer.start(5000)  # Update every 5 seconds
    
    def _create_header(self):
        header = QFrame()
        header.setStyleSheet(f"background-color: {COLORS['primary']}; color: white; border-radius: 5px;")
        header.setMaximumHeight(80)
        
        layout = QHBoxLayout(header)
        
        # Title
        title = QLabel("Soil Health Monitoring System")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: white;")
        
        # Help button
        help_button = QPushButton("?")
        help_button.setFixedSize(30, 30)
        help_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: white;
                border-radius: 15px;
                font-weight: bold;
                font-size: 16px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['header']};
            }}
        """)
        help_button.clicked.connect(self.show_help)
        
        # Status indicator
        self.connection_indicator = QLabel("⚠️ Not Connected")
        self.connection_indicator.setFont(QFont("Arial", 12))
        self.connection_indicator.setStyleSheet("color: white;")
        
        # Add to layout
        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(help_button)
        layout.addWidget(self.connection_indicator)
        
        return header
    
    def _create_dashboard_tab(self):
        tab = QWidget()
        tab.setStyleSheet(f"background-color: {COLORS['background']};")
        layout = QGridLayout(tab)
        layout.setSpacing(15)  # Add spacing between widgets
        
        # Current readings section
        readings_group = QFrame()
        readings_group.setStyleSheet(f"background-color: {COLORS['card']}; border: 2px solid {COLORS['border']}; border-radius: 10px;")
        readings_layout = QVBoxLayout(readings_group)
        
        readings_title = QLabel("Current Readings")
        readings_title.setFont(QFont("Arial", 14, QFont.Bold))
        readings_title.setStyleSheet(f"color: {COLORS['header']};")
        readings_layout.addWidget(readings_title)
        
        # Create labels for sensor readings
        self.temp_label = QLabel("Temperature: --")
        self.temp_label.setStyleSheet(f"color: {COLORS['text']}; font-size: 12px;")
        self.moisture_label = QLabel("Moisture: --")
        self.moisture_label.setStyleSheet(f"color: {COLORS['text']}; font-size: 12px;")
        self.nitrogen_label = QLabel("Nitrogen (N): --")
        self.nitrogen_label.setStyleSheet(f"color: {COLORS['text']}; font-size: 12px;")
        self.phosphorus_label = QLabel("Phosphorus (P): --")
        self.phosphorus_label.setStyleSheet(f"color: {COLORS['text']}; font-size: 12px;")
        self.potassium_label = QLabel("Potassium (K): --")
        self.potassium_label.setStyleSheet(f"color: {COLORS['text']}; font-size: 12px;")
        
        # Add labels to layout
        readings_layout.addWidget(self.temp_label)
        readings_layout.addWidget(self.moisture_label)
        readings_layout.addWidget(self.nitrogen_label)
        readings_layout.addWidget(self.phosphorus_label)
        readings_layout.addWidget(self.potassium_label)
        readings_layout.addStretch()
        
        # Soil fertility prediction section
        fertility_group = QFrame()
        fertility_group.setStyleSheet(f"background-color: {COLORS['card']}; border: 2px solid {COLORS['border']}; border-radius: 10px;")
        fertility_layout = QVBoxLayout(fertility_group)
        
        fertility_title = QLabel("Soil Fertility Prediction")
        fertility_title.setFont(QFont("Arial", 14, QFont.Bold))
        fertility_title.setStyleSheet(f"color: {COLORS['header']};")
        fertility_layout.addWidget(fertility_title)
        
        self.fertility_label = QLabel("Fertility: --")
        self.fertility_label.setFont(QFont("Arial", 16, QFont.Bold))
        
        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setStyleSheet(f"color: {COLORS['text']}; font-size: 12px;")
        
        fertility_layout.addWidget(self.fertility_label)
        fertility_layout.addWidget(self.confidence_label)
        fertility_layout.addStretch()
        
        # NPK Prediction section
        npk_group = QFrame()
        npk_group.setStyleSheet(f"background-color: {COLORS['card']}; border: 2px solid {COLORS['border']}; border-radius: 10px;")
        npk_layout = QVBoxLayout(npk_group)
        
        npk_title = QLabel("NPK Analysis & Recommendations")
        npk_title.setFont(QFont("Arial", 14, QFont.Bold))
        npk_title.setStyleSheet(f"color: {COLORS['header']};")
        npk_layout.addWidget(npk_title)
        
        # Create NPK status labels and meters
        # Nitrogen section
        n_section = QFrame()
        n_layout = QHBoxLayout(n_section)
        
        n_labels = QFrame()
        n_labels_layout = QVBoxLayout(n_labels)
        
        self.n_status_label = QLabel("Nitrogen (N): --")
        self.n_status_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.n_status_label.setToolTip("Nitrogen is essential for leaf growth and green vegetation. Measured in mg/kg.")
        
        self.n_category_label = QLabel("Status: --")
        self.n_category_label.setToolTip("Low: <100 mg/kg | Medium: 100-200 mg/kg | High: >200 mg/kg")
        
        self.n_recommendation_label = QLabel("Recommendation: --")
        self.n_recommendation_label.setWordWrap(True)
        self.n_recommendation_label.setToolTip("Specific actions to optimize nitrogen levels in your soil")
        
        n_labels_layout.addWidget(self.n_status_label)
        n_labels_layout.addWidget(self.n_category_label)
        n_labels_layout.addWidget(self.n_recommendation_label)
        
        # Create nitrogen meter
        self.n_meter = QProgressBar()
        self.n_meter.setOrientation(Qt.Vertical)
        self.n_meter.setRange(0, 100)
        self.n_meter.setValue(0)
        self.n_meter.setTextVisible(False)
        self.n_meter.setFixedWidth(30)
        self.n_meter.setToolTip("Nitrogen Health Percentage: Shows how optimal your nitrogen levels are (0-100%)")
        self.n_meter.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 20px;
            }
        """)
        
        n_layout.addWidget(n_labels, 4)
        n_layout.addWidget(self.n_meter, 1)
        
        # Phosphorus section
        p_section = QFrame()
        p_layout = QHBoxLayout(p_section)
        
        p_labels = QFrame()
        p_labels_layout = QVBoxLayout(p_labels)
        
        self.p_status_label = QLabel("Phosphorus (P): --")
        self.p_status_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.p_status_label.setToolTip("Phosphorus is crucial for root development and flowering. Measured in mg/kg.")
        
        self.p_category_label = QLabel("Status: --")
        self.p_category_label.setToolTip("Low: <5 mg/kg | Medium: 5-10 mg/kg | High: >10 mg/kg")
        
        self.p_recommendation_label = QLabel("Recommendation: --")
        self.p_recommendation_label.setWordWrap(True)
        self.p_recommendation_label.setToolTip("Specific actions to optimize phosphorus levels in your soil")
        
        p_labels_layout.addWidget(self.p_status_label)
        p_labels_layout.addWidget(self.p_category_label)
        p_labels_layout.addWidget(self.p_recommendation_label)
        
        # Create phosphorus meter
        self.p_meter = QProgressBar()
        self.p_meter.setOrientation(Qt.Vertical)
        self.p_meter.setRange(0, 100)
        self.p_meter.setValue(0)
        self.p_meter.setTextVisible(False)
        self.p_meter.setFixedWidth(30)
        self.p_meter.setToolTip("Phosphorus Health Percentage: Shows how optimal your phosphorus levels are (0-100%)")
        self.p_meter.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                width: 20px;
            }
        """)
        
        p_layout.addWidget(p_labels, 4)
        p_layout.addWidget(self.p_meter, 1)
        
        # Potassium section
        k_section = QFrame()
        k_layout = QHBoxLayout(k_section)
        
        k_labels = QFrame()
        k_labels_layout = QVBoxLayout(k_labels)
        
        self.k_status_label = QLabel("Potassium (K): --")
        self.k_status_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.k_status_label.setToolTip("Potassium enhances overall plant health and disease resistance. Measured in mg/kg.")
        
        self.k_category_label = QLabel("Status: --")
        self.k_category_label.setToolTip("Low: <300 mg/kg | Medium: 300-500 mg/kg | High: >500 mg/kg")
        
        self.k_recommendation_label = QLabel("Recommendation: --")
        self.k_recommendation_label.setWordWrap(True)
        self.k_recommendation_label.setToolTip("Specific actions to optimize potassium levels in your soil")
        
        k_labels_layout.addWidget(self.k_status_label)
        k_labels_layout.addWidget(self.k_category_label)
        k_labels_layout.addWidget(self.k_recommendation_label)
        
        # Create potassium meter
        self.k_meter = QProgressBar()
        self.k_meter.setOrientation(Qt.Vertical)
        self.k_meter.setRange(0, 100)
        self.k_meter.setValue(0)
        self.k_meter.setTextVisible(False)
        self.k_meter.setFixedWidth(30)
        self.k_meter.setToolTip("Potassium Health Percentage: Shows how optimal your potassium levels are (0-100%)")
        self.k_meter.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #FF9800;
                width: 20px;
            }
        """)
        
        k_layout.addWidget(k_labels, 4)
        k_layout.addWidget(self.k_meter, 1)
        
        # Add NPK sections to layout
        npk_layout.addWidget(n_section)
        npk_layout.addSpacing(10)
        npk_layout.addWidget(p_section)
        npk_layout.addSpacing(10)
        npk_layout.addWidget(k_section)
        
        # Plots section
        plots_group = QFrame()
        plots_group.setStyleSheet(f"background-color: {COLORS['card']}; border: 2px solid {COLORS['border']}; border-radius: 10px;")
        plots_layout = QVBoxLayout(plots_group)
        
        plots_title = QLabel("Sensor Data Trends")
        plots_title.setFont(QFont("Arial", 14, QFont.Bold))
        plots_title.setStyleSheet(f"color: {COLORS['header']};")
        plots_layout.addWidget(plots_title)
        
        # Create plot canvases
        self.temp_moisture_canvas = MplCanvas(self, width=5, height=3, dpi=100)
        self.npk_canvas = MplCanvas(self, width=5, height=3, dpi=100)
        
        plots_layout.addWidget(self.temp_moisture_canvas)
        plots_layout.addWidget(self.npk_canvas)
        
        # Add widgets to grid layout
        layout.addWidget(readings_group, 0, 0, 1, 1)
        layout.addWidget(fertility_group, 0, 1, 1, 1)
        layout.addWidget(npk_group, 0, 2, 1, 1)
        layout.addWidget(plots_group, 1, 0, 1, 3)
        
        return tab
    
    def _create_history_tab(self):
        tab = QWidget()
        tab.setStyleSheet(f"background-color: {COLORS['background']};")
        layout = QVBoxLayout(tab)
        
        # Create table for history data
        self.history_table = QTableWidget(0, 10)  # Time, Temp, Moisture, N, P, K, N Status, P Status, K Status, Fertility
        self.history_table.setHorizontalHeaderLabels(["Time", "Temp (°C)", "Moisture (%)", 
                                                     "N", "P", "K", 
                                                     "N Status", "P Status", "K Status", "Fertility"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setStyleSheet(f"background-color: {COLORS['card']}; color: {COLORS['text']};")
        
        layout.addWidget(self.history_table)
        
        return tab
    
    def _create_recommendations_tab(self):
        tab = QWidget()
        tab.setStyleSheet(f"background-color: {COLORS['background']};")
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        
        # Current soil status
        status_frame = QFrame()
        status_frame.setStyleSheet(f"background-color: {COLORS['card']}; border: 2px solid {COLORS['border']}; border-radius: 10px;")
        status_layout = QVBoxLayout(status_frame)
        
        status_title = QLabel("Current Soil Status")
        status_title.setFont(QFont("Arial", 14, QFont.Bold))
        status_title.setStyleSheet(f"color: {COLORS['header']};")
        status_layout.addWidget(status_title)
        
        self.soil_status_label = QLabel("Waiting for data...")
        self.soil_status_label.setFont(QFont("Arial", 12))
        self.soil_status_label.setStyleSheet(f"color: {COLORS['text']};")
        status_layout.addWidget(self.soil_status_label)
        
        # Add NPK summary to status
        self.npk_summary_label = QLabel("NPK Status: Waiting for data...")
        self.npk_summary_label.setFont(QFont("Arial", 12))
        self.npk_summary_label.setStyleSheet(f"color: {COLORS['text']};")
        status_layout.addWidget(self.npk_summary_label)
        
        # Plant recommendations
        recommendations_frame = QFrame()
        recommendations_frame.setStyleSheet(f"background-color: {COLORS['card']}; border: 2px solid {COLORS['border']}; border-radius: 10px;")
        recommendations_layout = QVBoxLayout(recommendations_frame)
        
        # Create tab widget for different recommendation types
        recommendations_tabs = QTabWidget()
        recommendations_tabs.setStyleSheet(f"background-color: {COLORS['card']}; color: {COLORS['text']};")
        
        # General recommendations tab
        general_tab = QWidget()
        general_layout = QVBoxLayout(general_tab)
        
        general_title = QLabel("Fertility-Based Recommendations")
        general_title.setFont(QFont("Arial", 12, QFont.Bold))
        general_title.setStyleSheet(f"color: {COLORS['header']};")
        general_layout.addWidget(general_title)
        
        # Table for general plant recommendations
        self.plants_table = QTableWidget(0, 2)
        self.plants_table.setHorizontalHeaderLabels(["Plant", "Description"])
        self.plants_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.plants_table.setStyleSheet(f"background-color: {COLORS['card']}; color: {COLORS['text']};")
        general_layout.addWidget(self.plants_table)
        
        # NPK-specific recommendations tab
        npk_tab = QWidget()
        npk_layout = QVBoxLayout(npk_tab)
        
        npk_title = QLabel("NPK-Based Recommendations")
        npk_title.setFont(QFont("Arial", 12, QFont.Bold))
        npk_title.setStyleSheet(f"color: {COLORS['header']};")
        npk_layout.addWidget(npk_title)
        
        # Table for NPK-specific plant recommendations
        self.npk_plants_table = QTableWidget(0, 3)
        self.npk_plants_table.setHorizontalHeaderLabels(["Plant", "NPK Requirement", "Description"])
        self.npk_plants_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.npk_plants_table.setStyleSheet(f"background-color: {COLORS['card']}; color: {COLORS['text']};")
        npk_layout.addWidget(self.npk_plants_table)
        
        # Add tabs to tab widget
        recommendations_tabs.addTab(general_tab, "General")
        recommendations_tabs.addTab(npk_tab, "NPK-Specific")
        
        recommendations_layout.addWidget(recommendations_tabs)
        
        # Add frames to layout
        layout.addWidget(status_frame)
        layout.addWidget(recommendations_frame)
        
        return tab
    
    def update_dashboard(self, data):
        """Update dashboard with new data"""
        # Add data to history
        self.data_history.append(data)
        
        # Limit history size
        if len(self.data_history) > self.max_history:
            self.data_history = self.data_history[-self.max_history:]
        
        # Update current readings
        self.temp_label.setText(f"Temperature: {data.get('temperature', '--')} °C")
        self.moisture_label.setText(f"Moisture: {data.get('moisture', '--')} %")
        self.nitrogen_label.setText(f"Nitrogen (N): {data.get('nitrogen', '--')}")
        self.phosphorus_label.setText(f"Phosphorus (P): {data.get('phosphorus', '--')}")
        self.potassium_label.setText(f"Potassium (K): {data.get('potassium', '--')}")
        
        # Update fertility prediction
        fertility_class = data.get('fertility_class')
        fertility_label = data.get('fertility_label', '--')
        confidence = data.get('confidence', '--')
        
        # Set color based on fertility class
        if fertility_class == 0:
            color = COLORS['less_fertile']
        elif fertility_class == 1:
            color = COLORS['fertile']
        elif fertility_class == 2:
            color = COLORS['highly_fertile']
        else:
            color = COLORS['text']
        
        self.fertility_label.setText(f"Fertility: {fertility_label}")
        self.fertility_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.confidence_label.setText(f"Confidence: {confidence}%")
        
        # Update NPK predictions
        n_value = data.get('nitrogen', 0)
        p_value = data.get('phosphorus', 0)
        k_value = data.get('potassium', 0)
        
        # Determine NPK categories and recommendations
        # Nitrogen
        if n_value < 100:
            n_category = "low"
            n_recommendation = "Add nitrogen-rich fertilizers or compost."
            n_health = 30
        elif n_value < 200:
            n_category = "medium"
            n_recommendation = "Moderate nitrogen levels, monitor regularly."
            n_health = 60
        else:
            n_category = "high"
            n_recommendation = "Good nitrogen levels, maintain current practices."
            n_health = 90
        
        # Phosphorus
        if p_value < 5:
            p_category = "low"
            p_recommendation = "Add phosphorus-rich fertilizers or bone meal."
            p_health = 30
        elif p_value < 10:
            p_category = "medium"
            p_recommendation = "Moderate phosphorus levels, monitor regularly."
            p_health = 60
        else:
            p_category = "high"
            p_recommendation = "Good phosphorus levels, maintain current practices."
            p_health = 90
        
        # Potassium
        if k_value < 300:
            k_category = "low"
            k_recommendation = "Add potassium-rich fertilizers or wood ash."
            k_health = 30
        elif k_value < 500:
            k_category = "medium"
            k_recommendation = "Moderate potassium levels, monitor regularly."
            k_health = 60
        else:
            k_category = "high"
            k_recommendation = "Good potassium levels, maintain current practices."
            k_health = 90
        
        # Update UI with NPK status
        self.n_status_label.setText(f"Nitrogen (N): {n_value}")
        self.n_category_label.setText(f"Status: {n_category.capitalize()} ({n_health}%)")
        self.n_recommendation_label.setText(f"Recommendation: {n_recommendation}")
        self.n_status_label.setStyleSheet(f"color: {COLORS[n_category]}; font-weight: bold;")
        self.n_category_label.setStyleSheet(f"color: {COLORS[n_category]};")
        
        # Update nitrogen meter
        self.n_meter.setValue(n_health)
        self.n_meter.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid grey;
                border-radius: 5px;
                background-color: #f0f0f0;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS[n_category]};
                width: 20px;
            }}
        """)
        
        self.p_status_label.setText(f"Phosphorus (P): {p_value}")
        self.p_category_label.setText(f"Status: {p_category.capitalize()} ({p_health}%)")
        self.p_recommendation_label.setText(f"Recommendation: {p_recommendation}")
        self.p_status_label.setStyleSheet(f"color: {COLORS[p_category]}; font-weight: bold;")
        self.p_category_label.setStyleSheet(f"color: {COLORS[p_category]};")
        
        # Update phosphorus meter
        self.p_meter.setValue(p_health)
        self.p_meter.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid grey;
                border-radius: 5px;
                background-color: #f0f0f0;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS[p_category]};
                width: 20px;
            }}
        """)
        
        self.k_status_label.setText(f"Potassium (K): {k_value}")
        self.k_category_label.setText(f"Status: {k_category.capitalize()} ({k_health}%)")
        self.k_recommendation_label.setText(f"Recommendation: {k_recommendation}")
        self.k_status_label.setStyleSheet(f"color: {COLORS[k_category]}; font-weight: bold;")
        self.k_category_label.setStyleSheet(f"color: {COLORS[k_category]};")
        
        # Update potassium meter
        self.k_meter.setValue(k_health)
        self.k_meter.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid grey;
                border-radius: 5px;
                background-color: #f0f0f0;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS[k_category]};
                width: 20px;
            }}
        """)
        
        # Update history table
        self._update_history_table()
        
        # Update recommendations
        self._update_recommendations(fertility_class, data)
        
        # Store current data for LLM analysis
        self.current_data = data
        self.current_fertility_prediction = {
            'fertility_class': fertility_class,
            'fertility_label': fertility_label,
            'confidence': confidence
        }
        
        # Update status
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_label.setText(f"Last update: {timestamp}")
    
    def update_plots(self):
        """Update plot canvases with latest data"""
        if not self.data_history:
            return
        
        # Extract data for plotting
        timestamps = [datetime.fromisoformat(d.get('timestamp', datetime.now().isoformat())) 
                     for d in self.data_history]
        temperatures = [d.get('temperature', 0) for d in self.data_history]
        moistures = [d.get('moisture', 0) for d in self.data_history]
        nitrogens = [d.get('nitrogen', 0) for d in self.data_history]
        phosphoruses = [d.get('phosphorus', 0) for d in self.data_history]
        potassiums = [d.get('potassium', 0) for d in self.data_history]
        
        # Temperature and moisture plot
        self.temp_moisture_canvas.axes.clear()
        ax1 = self.temp_moisture_canvas.axes
        ax1.set_title('Temperature and Moisture')
        ax1.set_xlabel('Time')
        
        # Plot temperature
        color = 'tab:red'
        ax1.set_ylabel('Temperature (°C)', color=color)
        ax1.plot(timestamps, temperatures, color=color, marker='o', linestyle='-', label='Temperature')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create second y-axis for moisture
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Moisture (%)', color=color)
        ax2.plot(timestamps, moistures, color=color, marker='s', linestyle='-', label='Moisture')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Format x-axis to show time
        ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # NPK plot
        self.npk_canvas.axes.clear()
        ax = self.npk_canvas.axes
        ax.set_title('NPK Levels')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        
        # Plot NPK values
        ax.plot(timestamps, nitrogens, 'g-', marker='o', label='Nitrogen (N)')
        ax.plot(timestamps, phosphoruses, 'b-', marker='s', label='Phosphorus (P)')
        ax.plot(timestamps, potassiums, 'r-', marker='^', label='Potassium (K)')
        
        # Format x-axis to show time
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add legend
        ax.legend()
        
        # Redraw canvases
        self.temp_moisture_canvas.fig.tight_layout()
        self.temp_moisture_canvas.draw()
        
        self.npk_canvas.fig.tight_layout()
        self.npk_canvas.draw()
    
    def _update_history_table(self):
        """Update history table with latest data"""
        # Clear table
        self.history_table.setRowCount(0)
        
        # Add data to table (most recent first)
        for i, data in enumerate(reversed(self.data_history)):
            self.history_table.insertRow(i)
            
            # Format timestamp
            timestamp = datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat()))
            time_str = timestamp.strftime("%H:%M:%S")
            
            # Add data to table
            self.history_table.setItem(i, 0, QTableWidgetItem(time_str))
            self.history_table.setItem(i, 1, QTableWidgetItem(str(data.get('temperature', '--'))))
            self.history_table.setItem(i, 2, QTableWidgetItem(str(data.get('moisture', '--'))))
            self.history_table.setItem(i, 3, QTableWidgetItem(str(data.get('nitrogen', '--'))))
            self.history_table.setItem(i, 4, QTableWidgetItem(str(data.get('phosphorus', '--'))))
            self.history_table.setItem(i, 5, QTableWidgetItem(str(data.get('potassium', '--'))))
            
            # Determine NPK categories
            n_value = data.get('nitrogen', 0)
            p_value = data.get('phosphorus', 0)
            k_value = data.get('potassium', 0)
            
            # Nitrogen status
            if n_value < 100:
                n_category = "low"
            elif n_value < 200:
                n_category = "medium"
            else:
                n_category = "high"
            
            # Phosphorus status
            if p_value < 5:
                p_category = "low"
            elif p_value < 10:
                p_category = "medium"
            else:
                p_category = "high"
            
            # Potassium status
            if k_value < 300:
                k_category = "low"
            elif k_value < 500:
                k_category = "medium"
            else:
                k_category = "high"
            
            # Add NPK status with color
            n_status_item = QTableWidgetItem(n_category.capitalize())
            n_status_item.setForeground(QColor(COLORS[n_category]))
            self.history_table.setItem(i, 6, n_status_item)
            
            p_status_item = QTableWidgetItem(p_category.capitalize())
            p_status_item.setForeground(QColor(COLORS[p_category]))
            self.history_table.setItem(i, 7, p_status_item)
            
            k_status_item = QTableWidgetItem(k_category.capitalize())
            k_status_item.setForeground(QColor(COLORS[k_category]))
            self.history_table.setItem(i, 8, k_status_item)
            
            # Add fertility with color
            fertility_item = QTableWidgetItem(data.get('fertility_label', '--'))
            
            # Set color based on fertility class
            fertility_class = data.get('fertility_class')
            if fertility_class == 0:
                fertility_item.setForeground(QColor(COLORS['less_fertile']))
            elif fertility_class == 1:
                fertility_item.setForeground(QColor(COLORS['fertile']))
            elif fertility_class == 2:
                fertility_item.setForeground(QColor(COLORS['highly_fertile']))
            
            self.history_table.setItem(i, 9, fertility_item)
    
    def _update_recommendations(self, fertility_class, data):
        """Update plant recommendations based on soil fertility"""
        if fertility_class is None:
            return
        
        # Update soil status
        fertility_labels = ["Less Fertile", "Fertile", "Highly Fertile"]
        if fertility_class in [0, 1, 2]:
            status_text = f"Your soil is classified as <b>{fertility_labels[fertility_class]}</b> with "
            status_text += f"N: {data.get('nitrogen', '--')}, P: {data.get('phosphorus', '--')}, K: {data.get('potassium', '--')}."
            
            if fertility_class == 0:
                status_text += " Consider adding organic matter and fertilizers to improve soil fertility."
            elif fertility_class == 1:
                status_text += " Your soil has good fertility for many crops."
            elif fertility_class == 2:
                status_text += " Your soil is highly fertile and suitable for nutrient-demanding crops."
            
            self.soil_status_label.setText(status_text)
        
        # Get NPK values and determine categories
        n_value = data.get('nitrogen', 0)
        p_value = data.get('phosphorus', 0)
        k_value = data.get('potassium', 0)
        
        # Determine NPK categories
        if n_value < 100:
            n_category = "low"
        elif n_value < 200:
            n_category = "medium"
        else:
            n_category = "high"
        
        if p_value < 5:
            p_category = "low"
        elif p_value < 10:
            p_category = "medium"
        else:
            p_category = "high"
        
        if k_value < 300:
            k_category = "low"
        elif k_value < 500:
            k_category = "medium"
        else:
            k_category = "high"
        
        # Update NPK summary
        npk_summary = f"NPK Status: N: {n_category.capitalize()}, P: {p_category.capitalize()}, K: {k_category.capitalize()}"
        self.npk_summary_label.setText(npk_summary)
        
        # Update plant recommendations table
        self.plants_table.setRowCount(0)
        
        if fertility_class in PLANT_RECOMMENDATIONS:
            recommendations = PLANT_RECOMMENDATIONS[fertility_class]
            
            for i, plant in enumerate(recommendations):
                self.plants_table.insertRow(i)
                self.plants_table.setItem(i, 0, QTableWidgetItem(plant["name"]))
                self.plants_table.setItem(i, 1, QTableWidgetItem(plant["description"]))
        
        # Update NPK-specific recommendations
        self.npk_plants_table.setRowCount(0)  # Clear existing rows
        
        # Define NPK-specific plant recommendations
        npk_recommendations = [
            # Plants for low N
            ("Legumes (Beans, Peas)", "Low N, Medium P, Medium K", "Nitrogen-fixing plants that improve soil nitrogen") if n_category == "low" else None,
            ("Clover", "Low N, Low P, Medium K", "Cover crop that fixes nitrogen in soil") if n_category == "low" else None,
            
            # Plants for low P
            ("Potatoes", "Medium N, Low P, High K", "Grow well in low phosphorus conditions") if p_category == "low" else None,
            ("Sweet Potatoes", "Low N, Low P, Medium K", "Tolerant of low phosphorus soils") if p_category == "low" else None,
            
            # Plants for low K
            ("Lettuce", "Medium N, Medium P, Low K", "Can grow in potassium-deficient soils") if k_category == "low" else None,
            ("Parsley", "Medium N, Low P, Low K", "Tolerates lower potassium levels") if k_category == "low" else None,
            
            # Plants for high N
            ("Tomatoes", "High N, Medium P, Medium K", "Thrive in nitrogen-rich environments") if n_category == "high" else None,
            ("Leafy Greens", "High N, Medium P, Medium K", "Benefit from high nitrogen for leaf development") if n_category == "high" else None,
            
            # Plants for high P
            ("Carrots", "Low N, High P, Medium K", "Root development benefits from phosphorus") if p_category == "high" else None,
            ("Onions", "Medium N, High P, Medium K", "Bulb formation enhanced by phosphorus") if p_category == "high" else None,
            
            # Plants for high K
            ("Squash", "Medium N, Medium P, High K", "Fruit development benefits from potassium") if k_category == "high" else None,
            ("Peppers", "Medium N, Medium P, High K", "Potassium improves fruit quality") if k_category == "high" else None,
            
            # Balanced NPK plants
            ("Corn", "Medium N, Medium P, Medium K", "Requires balanced nutrients for optimal growth") if n_category == "medium" and p_category == "medium" and k_category == "medium" else None,
            ("Cucumber", "Medium N, Medium P, Medium K", "Grows best with balanced NPK levels") if n_category == "medium" and p_category == "medium" and k_category == "medium" else None,
        ]
        
        # Add valid NPK recommendations to table
        row = 0
        for recommendation in npk_recommendations:
            if recommendation is not None:
                plant, npk_req, description = recommendation
                self.npk_plants_table.insertRow(row)
                self.npk_plants_table.setItem(row, 0, QTableWidgetItem(plant))
                self.npk_plants_table.setItem(row, 1, QTableWidgetItem(npk_req))
                self.npk_plants_table.setItem(row, 2, QTableWidgetItem(description))
                row += 1
        
        # Store recommendations for LLM analysis
        self.current_recommendations = {
            'plant_recommendations': PLANT_RECOMMENDATIONS.get(fertility_class, []),
            'fertility_class': fertility_class,
            'fertility_label': fertility_labels[fertility_class] if fertility_class in [0, 1, 2] else 'Unknown'
        }
    
    def update_connection_status(self, connected, message):
        """Update connection status indicator"""
        if connected:
            self.connection_indicator.setText("✅ Connected")
            self.connection_indicator.setStyleSheet("color: white;")
        else:
            self.connection_indicator.setText("⚠️ Simulation Mode")
            self.connection_indicator.setStyleSheet("color: yellow;")
        
        self.status_label.setText(message)
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop data collection thread
        if hasattr(self, 'data_thread'):
            self.data_thread.stop()
        
        event.accept()
        
    def show_help(self):
        """Show help information about NPK readings"""
        help_text = """
        <h3>Understanding NPK Readings</h3>
        <p><b>Nitrogen (N):</b> Essential for leaf growth and green vegetation. Measured in mg/kg.</p>
        <ul>
            <li><b>Low:</b> &lt;100 mg/kg - Yellowing leaves, stunted growth</li>
            <li><b>Medium:</b> 100-200 mg/kg - Adequate for most plants</li>
            <li><b>High:</b> &gt;200 mg/kg - Excellent for leafy vegetables</li>
        </ul>
        
        <p><b>Phosphorus (P):</b> Critical for root development and flowering. Measured in mg/kg.</p>
        <ul>
            <li><b>Low:</b> &lt;5 mg/kg - Poor root development, delayed maturity</li>
            <li><b>Medium:</b> 5-10 mg/kg - Adequate for most plants</li>
            <li><b>High:</b> &gt;10 mg/kg - Excellent for root crops and flowering plants</li>
        </ul>
        
        <p><b>Potassium (K):</b> Improves overall plant health and disease resistance. Measured in mg/kg.</p>
        <ul>
            <li><b>Low:</b> &lt;300 mg/kg - Weak stems, susceptibility to disease</li>
            <li><b>Medium:</b> 300-500 mg/kg - Adequate for most plants</li>
            <li><b>High:</b> &gt;500 mg/kg - Excellent for fruit development and stress resistance</li>
        </ul>
        """
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("NPK Readings Help")
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(help_text)
        msg_box.setStyleSheet(f"background-color: {COLORS['background']}; color: {COLORS['text']};")
        msg_box.exec_()

    def _create_llm_analysis_tab(self):
        """Create the LLM Analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # API Key input section
        api_key_frame = QFrame()
        api_key_frame.setStyleSheet("background-color: white; border-radius: 10px;")
        api_key_layout = QHBoxLayout(api_key_frame)
        
        api_key_label = QLabel("OpenAI API Key:")
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Enter your OpenAI API Key or set OPENAI_API_KEY environment variable")
        self.api_key_input.setText(os.environ.get("OPENAI_API_KEY", ""))
        
        api_key_layout.addWidget(api_key_label)
        api_key_layout.addWidget(self.api_key_input)
        
        # Analysis buttons
        button_frame = QFrame()
        button_frame.setStyleSheet("background-color: white; border-radius: 10px;")
        button_layout = QHBoxLayout(button_frame)
        
        self.analyze_button = QPushButton("Analyze Soil Health")
        self.analyze_button.clicked.connect(self.perform_llm_analysis)
        self.analyze_button.setStyleSheet(f"background-color: {COLORS['primary']}; color: white; padding: 10px;")
        
        self.stream_button = QPushButton("Stream Analysis")
        self.stream_button.clicked.connect(self.perform_streaming_analysis)
        self.stream_button.setStyleSheet(f"background-color: {COLORS['accent']}; color: white; padding: 10px;")
        
        button_layout.addWidget(self.analyze_button)
        button_layout.addWidget(self.stream_button)
        
        # Analysis output
        output_frame = QFrame()
        output_frame.setStyleSheet("background-color: white; border-radius: 10px;")
        output_layout = QVBoxLayout(output_frame)
        
        output_title = QLabel("Soil Health Analysis:")
        output_title.setFont(QFont("Arial", 14, QFont.Bold))
        
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setMinimumHeight(400)
        
        output_layout.addWidget(output_title)
        output_layout.addWidget(self.analysis_text)
        
        # Add all widgets to layout
        layout.addWidget(api_key_frame)
        layout.addWidget(button_frame)
        layout.addWidget(output_frame)
        
        return tab
    
    def perform_llm_analysis(self):
        """Perform LLM analysis on current soil data"""
        # Check if we have data to analyze
        if not hasattr(self, 'current_data') or not self.current_data:
            QMessageBox.warning(self, "No Data", "No soil data available for analysis. Please collect data first.")
            return
        
        # Get API key
        api_key = self.api_key_input.text() or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            QMessageBox.warning(self, "API Key Required", "Please enter your OpenAI API Key or set the OPENAI_API_KEY environment variable.")
            return
        
        # Initialize LLM analysis with API key
        self.llm_analysis = LLMSoilAnalysis(api_key=api_key)
        
        # Show loading message
        self.analysis_text.setText("Analyzing soil health data... Please wait.")
        QApplication.processEvents()
        
        # Perform analysis
        result = self.llm_analysis.get_soil_analysis(
            self.current_data,
            self.current_fertility_prediction,
            self.current_recommendations
        )
        
        # Display results
        if "error" in result:
            self.analysis_text.setText(f"Error: {result['error']}")
        else:
            self.analysis_text.setText(result["analysis"])
    
    def perform_streaming_analysis(self):
        """Perform streaming LLM analysis on current soil data"""
        # Check if we have data to analyze
        if not hasattr(self, 'current_data') or not self.current_data:
            QMessageBox.warning(self, "No Data", "No soil data available for analysis. Please collect data first.")
            return
        
        # Get API key
        api_key = self.api_key_input.text() or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            QMessageBox.warning(self, "API Key Required", "Please enter your OpenAI API Key or set the OPENAI_API_KEY environment variable.")
            return
        
        # Initialize LLM analysis with API key
        self.llm_analysis = LLMSoilAnalysis(api_key=api_key)
        
        # Clear previous analysis
        self.analysis_text.setText("")
        self.current_llm_analysis = ""
        
        # Define callback for streaming response
        def update_text(chunk):
            self.current_llm_analysis += chunk
            self.analysis_text.setText(self.current_llm_analysis)
            self.analysis_text.verticalScrollBar().setValue(
                self.analysis_text.verticalScrollBar().maximum()
            )
            QApplication.processEvents()
        
        # Perform streaming analysis
        self.llm_analysis.get_streaming_analysis(
            self.current_data,
            self.current_fertility_prediction,
            update_text,
            self.current_recommendations
        )

# Main application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show dashboard
    dashboard = SoilHealthDashboard()
    dashboard.show()
    
    sys.exit(app.exec_())