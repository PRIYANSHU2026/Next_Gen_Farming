import pandas as pd
import numpy as np
import json
import os
import csv

class PlantRecommendationSystem:
    def __init__(self, crop_data_path=None):
        # Load crop data or use default data if path not provided
        if crop_data_path and os.path.exists(crop_data_path):
            self.crop_data = self._load_crop_data(crop_data_path)
        else:
            self.crop_data = self._get_default_crop_data()
        
        # Define optimal soil parameter ranges for different fertility levels
        self.fertility_ranges = {
            0: {  # Less Fertile
                'N': (0, 180),
                'P': (0, 10),
                'K': (0, 400),
                'ph': (5.5, 8.0),
                'ec': (0.0, 0.5),
                'oc': (0.0, 0.8),
            },
            1: {  # Fertile
                'N': (180, 250),
                'P': (10, 20),
                'K': (400, 600),
                'ph': (6.0, 7.5),
                'ec': (0.5, 1.0),
                'oc': (0.8, 1.5),
            },
            2: {  # Highly Fertile
                'N': (250, 500),
                'P': (20, 50),
                'K': (600, 1000),
                'ph': (6.5, 7.2),
                'ec': (1.0, 2.0),
                'oc': (1.5, 3.0),
            }
        }
    
    def _load_crop_data(self, file_path):
        """Load crop data from CSV or JSON file"""
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            # Convert DataFrame to the format expected by the recommendation system
            crops_list = []
            for _, row in df.iterrows():
                crop_dict = {
                    "name": row["crop"],
                    "N_min": row["n_min"],
                    "N_max": row["n_max"],
                    "P_min": row["p_min"],
                    "P_max": row["p_max"],
                    "K_min": row["k_min"],
                    "K_max": row["k_max"],
                    "ph_min": row["ph_min"],
                    "ph_max": row["ph_max"],
                    "temperature_min": row["temperature_min"],
                    "temperature_max": row["temperature_max"],
                    "description": row["description"],
                    "growing_season": "Year-round",  # Default value
                    "water_requirement": "Medium",    # Default value
                    "fertilizer_requirement": "Medium"  # Default value
                }
                crops_list.append(crop_dict)
            return {"crops": crops_list}
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            print(f"Unsupported file format: {file_path}")
            return self._get_default_crop_data()
    
    def _get_default_crop_data(self):
        """Return default crop data with optimal growing conditions"""
        return {
            "crops": [
                {
                    "name": "Rice",
                    "N_min": 120,
                    "N_max": 200,
                    "P_min": 10,
                    "P_max": 25,
                    "K_min": 100,
                    "K_max": 200,
                    "ph_min": 5.5,
                    "ph_max": 6.5,
                    "rainfall_min": 1200,
                    "rainfall_max": 2000,
                    "temperature_min": 22,
                    "temperature_max": 32,
                    "description": "Staple food crop that grows well in wet conditions.",
                    "growing_season": "Kharif",
                    "water_requirement": "High",
                    "fertilizer_requirement": "Medium"
                },
                {
                    "name": "Wheat",
                    "N_min": 100,
                    "N_max": 150,
                    "P_min": 20,
                    "P_max": 40,
                    "K_min": 80,
                    "K_max": 120,
                    "ph_min": 6.0,
                    "ph_max": 7.5,
                    "rainfall_min": 450,
                    "rainfall_max": 650,
                    "temperature_min": 15,
                    "temperature_max": 25,
                    "description": "Important cereal crop for bread and other food products.",
                    "growing_season": "Rabi",
                    "water_requirement": "Medium",
                    "fertilizer_requirement": "Medium"
                },
                {
                    "name": "Maize (Corn)",
                    "N_min": 150,
                    "N_max": 250,
                    "P_min": 15,
                    "P_max": 30,
                    "K_min": 100,
                    "K_max": 180,
                    "ph_min": 5.5,
                    "ph_max": 7.5,
                    "rainfall_min": 500,
                    "rainfall_max": 800,
                    "temperature_min": 20,
                    "temperature_max": 30,
                    "description": "Versatile crop used for food, feed, and industrial products.",
                    "growing_season": "Kharif/Rabi",
                    "water_requirement": "Medium",
                    "fertilizer_requirement": "High"
                },
                {
                    "name": "Cotton",
                    "N_min": 120,
                    "N_max": 200,
                    "P_min": 20,
                    "P_max": 40,
                    "K_min": 120,
                    "K_max": 200,
                    "ph_min": 6.0,
                    "ph_max": 8.0,
                    "rainfall_min": 600,
                    "rainfall_max": 1200,
                    "temperature_min": 20,
                    "temperature_max": 35,
                    "description": "Important fiber crop for textile industry.",
                    "growing_season": "Kharif",
                    "water_requirement": "Medium",
                    "fertilizer_requirement": "High"
                },
                {
                    "name": "Sugarcane",
                    "N_min": 200,
                    "N_max": 300,
                    "P_min": 20,
                    "P_max": 40,
                    "K_min": 150,
                    "K_max": 250,
                    "ph_min": 6.0,
                    "ph_max": 7.5,
                    "rainfall_min": 1200,
                    "rainfall_max": 2000,
                    "temperature_min": 20,
                    "temperature_max": 35,
                    "description": "Perennial crop grown for sugar production.",
                    "growing_season": "Year-round",
                    "water_requirement": "High",
                    "fertilizer_requirement": "High"
                },
                {
                    "name": "Tomato",
                    "N_min": 100,
                    "N_max": 180,
                    "P_min": 15,
                    "P_max": 30,
                    "K_min": 120,
                    "K_max": 200,
                    "ph_min": 6.0,
                    "ph_max": 7.0,
                    "rainfall_min": 400,
                    "rainfall_max": 600,
                    "temperature_min": 20,
                    "temperature_max": 30,
                    "description": "Popular vegetable crop with high market value.",
                    "growing_season": "Year-round",
                    "water_requirement": "Medium",
                    "fertilizer_requirement": "Medium"
                },
                {
                    "name": "Potato",
                    "N_min": 120,
                    "N_max": 200,
                    "P_min": 25,
                    "P_max": 45,
                    "K_min": 150,
                    "K_max": 250,
                    "ph_min": 5.0,
                    "ph_max": 6.5,
                    "rainfall_min": 500,
                    "rainfall_max": 700,
                    "temperature_min": 15,
                    "temperature_max": 25,
                    "description": "Important tuber crop with high carbohydrate content.",
                    "growing_season": "Rabi",
                    "water_requirement": "Medium",
                    "fertilizer_requirement": "High"
                },
                {
                    "name": "Soybean",
                    "N_min": 50,
                    "N_max": 100,
                    "P_min": 20,
                    "P_max": 40,
                    "K_min": 80,
                    "K_max": 150,
                    "ph_min": 6.0,
                    "ph_max": 7.5,
                    "rainfall_min": 600,
                    "rainfall_max": 1000,
                    "temperature_min": 20,
                    "temperature_max": 30,
                    "description": "Legume crop that fixes nitrogen in soil and has high protein content.",
                    "growing_season": "Kharif",
                    "water_requirement": "Medium",
                    "fertilizer_requirement": "Low"
                },
                {
                    "name": "Groundnut (Peanut)",
                    "N_min": 40,
                    "N_max": 80,
                    "P_min": 20,
                    "P_max": 40,
                    "K_min": 75,
                    "K_max": 125,
                    "ph_min": 6.0,
                    "ph_max": 7.5,
                    "rainfall_min": 500,
                    "rainfall_max": 1000,
                    "temperature_min": 20,
                    "temperature_max": 30,
                    "description": "Important oilseed and food crop that grows well in sandy loam soils.",
                    "growing_season": "Kharif",
                    "water_requirement": "Medium",
                    "fertilizer_requirement": "Low"
                },
                {
                    "name": "Sunflower",
                    "N_min": 80,
                    "N_max": 120,
                    "P_min": 20,
                    "P_max": 40,
                    "K_min": 80,
                    "K_max": 120,
                    "ph_min": 6.0,
                    "ph_max": 7.5,
                    "rainfall_min": 400,
                    "rainfall_max": 700,
                    "temperature_min": 20,
                    "temperature_max": 30,
                    "description": "Oilseed crop that is drought-tolerant and can grow in various soil types.",
                    "growing_season": "Rabi/Zaid",
                    "water_requirement": "Low",
                    "fertilizer_requirement": "Medium"
                },
                {
                    "name": "Chickpea (Gram)",
                    "N_min": 30,
                    "N_max": 60,
                    "P_min": 20,
                    "P_max": 40,
                    "K_min": 40,
                    "K_max": 80,
                    "ph_min": 6.0,
                    "ph_max": 8.0,
                    "rainfall_min": 400,
                    "rainfall_max": 600,
                    "temperature_min": 15,
                    "temperature_max": 25,
                    "description": "Pulse crop that is drought-tolerant and fixes nitrogen in soil.",
                    "growing_season": "Rabi",
                    "water_requirement": "Low",
                    "fertilizer_requirement": "Low"
                },
                {
                    "name": "Mustard",
                    "N_min": 60,
                    "N_max": 100,
                    "P_min": 20,
                    "P_max": 40,
                    "K_min": 40,
                    "K_max": 80,
                    "ph_min": 6.0,
                    "ph_max": 7.5,
                    "rainfall_min": 400,
                    "rainfall_max": 600,
                    "temperature_min": 15,
                    "temperature_max": 25,
                    "description": "Oilseed crop that grows well in cooler temperatures.",
                    "growing_season": "Rabi",
                    "water_requirement": "Low",
                    "fertilizer_requirement": "Medium"
                },
                {
                    "name": "Onion",
                    "N_min": 80,
                    "N_max": 120,
                    "P_min": 30,
                    "P_max": 50,
                    "K_min": 80,
                    "K_max": 150,
                    "ph_min": 6.0,
                    "ph_max": 7.0,
                    "rainfall_min": 350,
                    "rainfall_max": 550,
                    "temperature_min": 15,
                    "temperature_max": 25,
                    "description": "Important vegetable crop with high market demand.",
                    "growing_season": "Rabi",
                    "water_requirement": "Medium",
                    "fertilizer_requirement": "Medium"
                },
                {
                    "name": "Banana",
                    "N_min": 200,
                    "N_max": 300,
                    "P_min": 20,
                    "P_max": 40,
                    "K_min": 300,
                    "K_max": 500,
                    "ph_min": 6.0,
                    "ph_max": 7.5,
                    "rainfall_min": 1200,
                    "rainfall_max": 2000,
                    "temperature_min": 20,
                    "temperature_max": 35,
                    "description": "Perennial fruit crop with high potassium requirement.",
                    "growing_season": "Year-round",
                    "water_requirement": "High",
                    "fertilizer_requirement": "High"
                },
                {
                    "name": "Mango",
                    "N_min": 100,
                    "N_max": 200,
                    "P_min": 20,
                    "P_max": 40,
                    "K_min": 150,
                    "K_max": 300,
                    "ph_min": 5.5,
                    "ph_max": 7.5,
                    "rainfall_min": 800,
                    "rainfall_max": 1500,
                    "temperature_min": 24,
                    "temperature_max": 35,
                    "description": "Popular fruit crop that requires well-drained soil.",
                    "growing_season": "Perennial",
                    "water_requirement": "Medium",
                    "fertilizer_requirement": "Medium"
                },
                {
                    "name": "Beans",
                    "N_min": 40,
                    "N_max": 80,
                    "P_min": 20,
                    "P_max": 40,
                    "K_min": 60,
                    "K_max": 100,
                    "ph_min": 6.0,
                    "ph_max": 7.5,
                    "rainfall_min": 400,
                    "rainfall_max": 700,
                    "temperature_min": 18,
                    "temperature_max": 30,
                    "description": "Legume crop that fixes nitrogen in soil and has high protein content.",
                    "growing_season": "Kharif/Rabi",
                    "water_requirement": "Medium",
                    "fertilizer_requirement": "Low"
                }
            ]
        }
    
    def get_recommendations(self, soil_data, fertility_class, top_n=5):
        """Get crop recommendations based on soil data and fertility class"""
        if fertility_class not in [0, 1, 2]:
            return {
                "error": "Invalid fertility class. Must be 0 (Less Fertile), 1 (Fertile), or 2 (Highly Fertile)."
            }
        
        # Extract soil parameters
        N = soil_data.get('nitrogen', soil_data.get('N', 0))
        P = soil_data.get('phosphorus', soil_data.get('P', 0))
        K = soil_data.get('potassium', soil_data.get('K', 0))
        ph = soil_data.get('ph', 7.0)
        temperature = soil_data.get('temperature', 25)
        
        # Calculate suitability scores for each crop
        crop_scores = []
        
        for crop in self.crop_data["crops"]:
            # Calculate score based on how well soil parameters match crop requirements
            n_score = self._calculate_parameter_score(N, crop["N_min"], crop["N_max"])
            p_score = self._calculate_parameter_score(P, crop["P_min"], crop["P_max"])
            k_score = self._calculate_parameter_score(K, crop["K_min"], crop["K_max"])
            ph_score = self._calculate_parameter_score(ph, crop["ph_min"], crop["ph_max"])
            temp_score = self._calculate_parameter_score(temperature, crop["temperature_min"], crop["temperature_max"])
            
            # Calculate overall score (weighted average)
            overall_score = (n_score * 0.25 + p_score * 0.2 + k_score * 0.2 + 
                             ph_score * 0.2 + temp_score * 0.15) * 100
            
            # Adjust score based on fertility class
            if fertility_class == 0:  # Less Fertile
                # Favor crops that can grow in less fertile conditions
                if crop["fertilizer_requirement"] == "Low":
                    overall_score *= 1.2
            elif fertility_class == 2:  # Highly Fertile
                # Favor crops that benefit from highly fertile soil
                if crop["fertilizer_requirement"] == "High":
                    overall_score *= 1.2
            
            crop_scores.append({
                "name": crop["name"],
                "score": round(overall_score, 2),
                "description": crop["description"],
                "growing_season": crop["growing_season"],
                "water_requirement": crop["water_requirement"],
                "fertilizer_requirement": crop["fertilizer_requirement"],
                "n_requirement": f"{crop['N_min']} - {crop['N_max']}",
                "p_requirement": f"{crop['P_min']} - {crop['P_max']}",
                "k_requirement": f"{crop['K_min']} - {crop['K_max']}",
                "ph_requirement": f"{crop['ph_min']} - {crop['ph_max']}",
                "temperature_requirement": f"{crop['temperature_min']} - {crop['temperature_max']} °C"
            })
        
        # Sort crops by score (descending)
        crop_scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Get top N recommendations
        top_recommendations = crop_scores[:top_n]
        
        # Add soil improvement recommendations based on fertility class
        improvement_recommendations = self._get_soil_improvement_recommendations(soil_data, fertility_class)
        
        return {
            "crop_recommendations": top_recommendations,
            "soil_improvement": improvement_recommendations
        }
    
    def _calculate_parameter_score(self, value, min_val, max_val):
        """Calculate score (0-1) based on how well a parameter fits within optimal range"""
        if value < min_val:
            # Below optimal range
            return max(0, 1 - (min_val - value) / min_val)
        elif value > max_val:
            # Above optimal range
            return max(0, 1 - (value - max_val) / max_val)
        else:
            # Within optimal range
            return 1.0
    
    def _get_soil_improvement_recommendations(self, soil_data, fertility_class):
        """Get recommendations for improving soil based on current parameters"""
        recommendations = []
        
        # Extract soil parameters
        N = soil_data.get('nitrogen', soil_data.get('N', 0))
        P = soil_data.get('phosphorus', soil_data.get('P', 0))
        K = soil_data.get('potassium', soil_data.get('K', 0))
        ph = soil_data.get('ph', 7.0)
        
        # Check if parameters are within optimal range for the fertility class
        if fertility_class in self.fertility_ranges:
            ranges = self.fertility_ranges[fertility_class]
            
            # Check Nitrogen
            if N < ranges['N'][0]:
                recommendations.append({
                    "parameter": "Nitrogen (N)",
                    "status": "Low",
                    "recommendation": "Add nitrogen-rich fertilizers like urea or ammonium sulfate. Consider planting legumes as cover crops to fix nitrogen."
                })
            elif N > ranges['N'][1]:
                recommendations.append({
                    "parameter": "Nitrogen (N)",
                    "status": "High",
                    "recommendation": "Reduce nitrogen fertilization. Plant nitrogen-consuming crops like corn or leafy greens."
                })
            
            # Check Phosphorus
            if P < ranges['P'][0]:
                recommendations.append({
                    "parameter": "Phosphorus (P)",
                    "status": "Low",
                    "recommendation": "Add phosphorus-rich fertilizers like superphosphate or bone meal. Incorporate organic matter like compost."
                })
            elif P > ranges['P'][1]:
                recommendations.append({
                    "parameter": "Phosphorus (P)",
                    "status": "High",
                    "recommendation": "Avoid phosphorus fertilizers. Plant phosphorus-consuming crops. Consider soil testing for phosphorus runoff risk."
                })
            
            # Check Potassium
            if K < ranges['K'][0]:
                recommendations.append({
                    "parameter": "Potassium (K)",
                    "status": "Low",
                    "recommendation": "Add potassium-rich fertilizers like potassium chloride or wood ash. Incorporate compost or manure."
                })
            elif K > ranges['K'][1]:
                recommendations.append({
                    "parameter": "Potassium (K)",
                    "status": "High",
                    "recommendation": "Reduce potassium fertilization. Plant potassium-consuming crops like tomatoes or potatoes."
                })
            
            # Check pH
            if ph < ranges['ph'][0]:
                recommendations.append({
                    "parameter": "pH",
                    "status": "Acidic",
                    "recommendation": "Apply agricultural lime to raise pH. Avoid acidifying fertilizers."
                })
            elif ph > ranges['ph'][1]:
                recommendations.append({
                    "parameter": "pH",
                    "status": "Alkaline",
                    "recommendation": "Apply sulfur or acidifying organic matter like pine needles to lower pH."
                })
        
        # General recommendations based on fertility class
        if fertility_class == 0:  # Less Fertile
            recommendations.append({
                "parameter": "Overall Fertility",
                "status": "Low",
                "recommendation": "Incorporate organic matter like compost or well-rotted manure. Consider cover cropping and crop rotation to build soil health. Apply balanced fertilizers based on soil test results."
            })
        elif fertility_class == 1:  # Fertile
            recommendations.append({
                "parameter": "Overall Fertility",
                "status": "Good",
                "recommendation": "Maintain current fertility with regular additions of organic matter. Follow crop rotation practices. Apply fertilizers based on crop needs and soil test results."
            })
        elif fertility_class == 2:  # Highly Fertile
            recommendations.append({
                "parameter": "Overall Fertility",
                "status": "High",
                "recommendation": "Focus on maintaining soil structure and preventing nutrient runoff. Consider planting nutrient-demanding crops. Monitor for potential imbalances in micronutrients."
            })
        
        return recommendations

# Example usage
if __name__ == "__main__":
    # Initialize recommendation system with CSV data
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop_recommendations.csv")
    recommender = PlantRecommendationSystem(crop_data_path=csv_path)
    
    # Example soil data
    soil_data = {
        "nitrogen": 150,
        "phosphorus": 10,
        "potassium": 200,
        "ph": 6.5,
        "temperature": 25,
        "moisture": 60
    }
    
    # Get recommendations for fertility class 1 (Fertile)
    recommendations = recommender.get_recommendations(soil_data, fertility_class=1)
    
    # Print recommendations
    print("\n🌱 Crop Recommendations:")
    for i, crop in enumerate(recommendations["crop_recommendations"]):
        print(f"{i+1}. {crop['name']} (Score: {crop['score']}%)")
        print(f"   Description: {crop['description']}")
        print(f"   Growing Season: {crop['growing_season']}")
        print(f"   Requirements: Water - {crop['water_requirement']}, Fertilizer - {crop['fertilizer_requirement']}")
        print()
    
    print("\n🔍 Soil Improvement Recommendations:")
    for rec in recommendations["soil_improvement"]:
        print(f"Parameter: {rec['parameter']} (Status: {rec['status']})")
        print(f"Recommendation: {rec['recommendation']}")
        print()