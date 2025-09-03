import unittest
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Import the modules to test
from utils.soil_health_interface import SoilHealthPredictor

class TestSoilHealthPredictor(unittest.TestCase):
    """Test cases for the SoilHealthPredictor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.predictor = SoilHealthPredictor()
    
    def test_initialization(self):
        """Test that the predictor initializes correctly"""
        self.assertIsNotNone(self.predictor)
    
    def test_fallback_model(self):
        """Test that the fallback model works"""
        # This is a simple test to ensure the fallback model can make predictions
        # In a real test, you would use mock data and verify the output
        self.assertTrue(hasattr(self.predictor, 'model'))

if __name__ == '__main__':
    unittest.main()