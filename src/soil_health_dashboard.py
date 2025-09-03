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
                             QLineEdit, QTextEdit, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor, QPalette, QPixmap

# Import matplotlib for plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import our soil health interface
from soil_health_interface import SoilHealthPredictor, ESP32Interface
from llm_soil_analysis import LLMSoilAnalysis

# Define color constants
COLORS = {
    'primary': '#2E7D32',  # Dark green
    'secondary': '#81C784',  # Light green
    'accent': '#FF8F00',  # Amber
    'background': '#F5F5F5',  # Light gray
    'text': '#212121',  # Dark gray
    'warning': '#F44336',  # Red
    'success': '#4CAF50',  # Green
    'info': '#2196F3',  # Blue
    'less_fertile': '#F44336',  # Red
    'fertile': '#FFC107',  # Yellow
    'highly_fertile': '#4CAF50',  # Green
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
        
        # Initialize LLM Soil Analysis
        self.llm_analysis = LLMSoilAnalysis()
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
        
        # Status indicator
        self.connection_indicator = QLabel("⚠️ Not Connected")
        self.connection_indicator.setFont(QFont("Arial", 12))
        self.connection_indicator.setStyleSheet("color: white;")
        
        # Add to layout
        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(self.connection_indicator)
        
        return header
    
    def _create_dashboard_tab(self):
        tab = QWidget()
        layout = QGridLayout(tab)
        
        # Current readings section
        readings_group = QFrame()
        readings_group.setStyleSheet("background-color: white; border-radius: 10px;")
        readings_layout = QVBoxLayout(readings_group)
        
        readings_title = QLabel("Current Readings")
        readings_title.setFont(QFont("Arial", 14, QFont.Bold))
        readings_layout.addWidget(readings_title)
        
        # Create labels for sensor readings
        self.temp_label = QLabel("Temperature: --")
        self.moisture_label = QLabel("Moisture: --")
        self.nitrogen_label = QLabel("Nitrogen (N): --")
        self.phosphorus_label = QLabel("Phosphorus (P): --")
        self.potassium_label = QLabel("Potassium (K): --")
        
        # Add labels to layout
        readings_layout.addWidget(self.temp_label)
        readings_layout.addWidget(self.moisture_label)
        readings_layout.addWidget(self.nitrogen_label)
        readings_layout.addWidget(self.phosphorus_label)
        readings_layout.addWidget(self.potassium_label)
        readings_layout.addStretch()
        
        # Soil fertility prediction section
        fertility_group = QFrame()
        fertility_group.setStyleSheet("background-color: white; border-radius: 10px;")
        fertility_layout = QVBoxLayout(fertility_group)
        
        fertility_title = QLabel("Soil Fertility Prediction")
        fertility_title.setFont(QFont("Arial", 14, QFont.Bold))
        fertility_layout.addWidget(fertility_title)
        
        self.fertility_label = QLabel("Fertility: --")
        self.fertility_label.setFont(QFont("Arial", 16, QFont.Bold))
        
        self.confidence_label = QLabel("Confidence: --")
        
        fertility_layout.addWidget(self.fertility_label)
        fertility_layout.addWidget(self.confidence_label)
        fertility_layout.addStretch()
        
        # Plots section
        plots_group = QFrame()
        plots_group.setStyleSheet("background-color: white; border-radius: 10px;")
        plots_layout = QVBoxLayout(plots_group)
        
        plots_title = QLabel("Sensor Data Trends")
        plots_title.setFont(QFont("Arial", 14, QFont.Bold))
        plots_layout.addWidget(plots_title)
        
        # Create plot canvases
        self.temp_moisture_canvas = MplCanvas(self, width=5, height=3, dpi=100)
        self.npk_canvas = MplCanvas(self, width=5, height=3, dpi=100)
        
        plots_layout.addWidget(self.temp_moisture_canvas)
        plots_layout.addWidget(self.npk_canvas)
        
        # Add widgets to grid layout
        layout.addWidget(readings_group, 0, 0, 1, 1)
        layout.addWidget(fertility_group, 0, 1, 1, 1)
        layout.addWidget(plots_group, 1, 0, 1, 2)
        
        return tab
    
    def _create_history_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create table for history data
        self.history_table = QTableWidget(0, 7)
        self.history_table.setHorizontalHeaderLabels(["Time", "Temp (°C)", "Moisture (%)", 
                                                     "N", "P", "K", "Fertility"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        layout.addWidget(self.history_table)
        
        return tab
    
    def _create_recommendations_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Current soil status
        status_frame = QFrame()
        status_frame.setStyleSheet("background-color: white; border-radius: 10px;")
        status_layout = QVBoxLayout(status_frame)
        
        status_title = QLabel("Current Soil Status")
        status_title.setFont(QFont("Arial", 14, QFont.Bold))
        status_layout.addWidget(status_title)
        
        self.soil_status_label = QLabel("Waiting for data...")
        self.soil_status_label.setFont(QFont("Arial", 12))
        status_layout.addWidget(self.soil_status_label)
        
        # Plant recommendations
        recommendations_frame = QFrame()
        recommendations_frame.setStyleSheet("background-color: white; border-radius: 10px;")
        recommendations_layout = QVBoxLayout(recommendations_frame)
        
        recommendations_title = QLabel("Recommended Plants")
        recommendations_title.setFont(QFont("Arial", 14, QFont.Bold))
        recommendations_layout.addWidget(recommendations_title)
        
        # Table for plant recommendations
        self.plants_table = QTableWidget(0, 2)
        self.plants_table.setHorizontalHeaderLabels(["Plant", "Description"])
        self.plants_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        recommendations_layout.addWidget(self.plants_table)
        
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
            
            self.history_table.setItem(i, 6, fertility_item)
    
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
        
        # Update plant recommendations table
        self.plants_table.setRowCount(0)
        
        if fertility_class in PLANT_RECOMMENDATIONS:
            recommendations = PLANT_RECOMMENDATIONS[fertility_class]
            
            for i, plant in enumerate(recommendations):
                self.plants_table.insertRow(i)
                self.plants_table.setItem(i, 0, QTableWidgetItem(plant["name"]))
                self.plants_table.setItem(i, 1, QTableWidgetItem(plant["description"]))
        
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