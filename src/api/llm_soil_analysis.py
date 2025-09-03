import os
import json
import requests
import time
from datetime import datetime

class LLMSoilAnalysis:
    def __init__(self, api_key=None):
        """Initialize the LLM Soil Analysis module
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided, will look for OPENAI_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = "gpt-4o"
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def get_soil_analysis(self, soil_data, fertility_prediction, crop_recommendations=None):
        """Get detailed soil analysis and recommendations using LLM
        
        Args:
            soil_data (dict): Dictionary containing soil sensor data
            fertility_prediction (dict): Dictionary containing fertility prediction results
            crop_recommendations (dict, optional): Dictionary containing crop recommendations
            
        Returns:
            dict: Dictionary containing LLM analysis results
        """
        if not self.api_key:
            return {
                "error": "No API key provided. Set OPENAI_API_KEY environment variable or provide api_key parameter."
            }
        
        # Create prompt for LLM
        prompt = self._create_prompt(soil_data, fertility_prediction, crop_recommendations)
        
        try:
            # Call OpenAI API
            response = self._call_openai_api(prompt)
            
            # Parse response
            analysis = self._parse_response(response)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis
            }
        except Exception as e:
            return {
                "error": str(e)
            }
    
    def _create_prompt(self, soil_data, fertility_prediction, crop_recommendations):
        """Create prompt for LLM based on soil data and predictions"""
        # Format soil data
        soil_data_str = json.dumps(soil_data, indent=2)
        
        # Format fertility prediction
        fertility_str = json.dumps(fertility_prediction, indent=2)
        
        # Format crop recommendations if provided
        crop_rec_str = ""
        if crop_recommendations:
            crop_rec_str = f"\n\nCrop Recommendations:\n{json.dumps(crop_recommendations, indent=2)}"
        
        # Create prompt
        prompt = f"""You are an expert agricultural scientist and soil analyst. Analyze the following soil data and provide detailed insights and recommendations.

Soil Data:
{soil_data_str}

Fertility Prediction:
{fertility_str}{crop_rec_str}

Based on this information, please provide:

1. A detailed analysis of the soil health and quality
2. Specific insights about the NPK levels and what they mean for plant growth
3. Recommendations for improving soil health if needed
4. Suggestions for optimal crops based on these soil conditions
5. Sustainable farming practices that would work well with this soil profile

Format your response in clear sections with markdown formatting. Be specific and practical in your recommendations."""
        
        return prompt
    
    def _call_openai_api(self, prompt):
        """Call OpenAI API with the given prompt"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert agricultural scientist and soil analyst."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1500
        }
        
        # Make API call
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API call failed: {str(e)}")
    
    def _parse_response(self, response):
        """Parse OpenAI API response"""
        try:
            # Extract content from response
            content = response['choices'][0]['message']['content']
            return content
        except (KeyError, IndexError) as e:
            raise Exception(f"Failed to parse API response: {str(e)}")
    
    def get_streaming_analysis(self, soil_data, fertility_prediction, callback, crop_recommendations=None):
        """Get soil analysis with streaming response
        
        Args:
            soil_data (dict): Dictionary containing soil sensor data
            fertility_prediction (dict): Dictionary containing fertility prediction results
            callback (function): Callback function to receive streaming chunks of text
            crop_recommendations (dict, optional): Dictionary containing crop recommendations
            
        Returns:
            str: Complete analysis text
        """
        if not self.api_key:
            callback("Error: No API key provided. Set OPENAI_API_KEY environment variable or provide api_key parameter.")
            return ""
        
        # Create prompt for LLM
        prompt = self._create_prompt(soil_data, fertility_prediction, crop_recommendations)
        
        # Create payload for streaming request
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert agricultural scientist and soil analyst."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1500,
            "stream": True
        }
        
        # Make streaming API call
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                stream=True
            )
            response.raise_for_status()
            
            # Process streaming response
            complete_response = ""
            for line in response.iter_lines():
                if line:
                    # Remove 'data: ' prefix and skip empty lines
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        line_json = line_text[6:]
                        if line_json.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(line_json)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                content = chunk['choices'][0].get('delta', {}).get('content', '')
                                if content:
                                    complete_response += content
                                    callback(content)
                        except json.JSONDecodeError:
                            pass
            
            return complete_response
        except requests.exceptions.RequestException as e:
            error_msg = f"API call failed: {str(e)}"
            callback(error_msg)
            return error_msg

# Example usage
if __name__ == "__main__":
    # Set your OpenAI API key here or as an environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # Initialize LLM Soil Analysis
    llm_analysis = LLMSoilAnalysis(api_key=api_key)
    
    # Example soil data
    soil_data = {
        "temperature": 25.3,
        "moisture": 62.7,
        "nitrogen": 180.5,
        "phosphorus": 15.2,
        "potassium": 320.8,
        "ph": 6.8
    }
    
    # Example fertility prediction
    fertility_prediction = {
        "fertility_class": 1,
        "fertility_label": "Fertile",
        "confidence": 87.5
    }
    
    # Example crop recommendations
    crop_recommendations = {
        "crop_recommendations": [
            {
                "name": "Tomato",
                "score": 92.5,
                "description": "Popular vegetable crop with high market value."
            },
            {
                "name": "Maize",
                "score": 88.3,
                "description": "Versatile crop used for food, feed, and industrial products."
            },
            {
                "name": "Wheat",
                "score": 85.1,
                "description": "Important cereal crop for bread and other food products."
            }
        ]
    }
    
    # Define callback function for streaming response
    def print_chunk(chunk):
        print(chunk, end="", flush=True)
    
    # Get streaming analysis
    print("\n🔍 Soil Analysis (Streaming):\n")
    llm_analysis.get_streaming_analysis(soil_data, fertility_prediction, print_chunk, crop_recommendations)
    
    # Get complete analysis
    print("\n\n🔍 Soil Analysis (Complete):\n")
    result = llm_analysis.get_soil_analysis(soil_data, fertility_prediction, crop_recommendations)
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(result["analysis"])