#!/usr/bin/env python3
"""
Soil Quality and Fertility Prediction System
Main entry point for the application
"""

import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the Streamlit app
from src.soil_health_streamlit import main

if __name__ == "__main__":
    # Run the Streamlit app
    main()