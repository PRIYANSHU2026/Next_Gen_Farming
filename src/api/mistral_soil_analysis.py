import requests
import json
import time

class MistralSoilAnalysis:
    """
    A class to integrate with Mistral AI for detailed soil health analysis and recommendations.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the Mistral AI integration.
        
        Args:
            api_key (str, optional): Mistral AI API key. If not provided, it must be set later.
        """
        self.api_key = api_key
        self.api_url = "https://api.mistral.ai/v1/chat/completions"
        self.client = self  # Use self as client for compatibility
    
    def set_api_key(self, api_key):
        """
        Set the Mistral AI API key.
        
        Args:
            api_key (str): The API key for Mistral AI.
        """
        self.api_key = api_key
    
    def create_prompt(self, soil_data, fertility_prediction, recommendations=None):
        """
        Create a prompt for Mistral AI based on soil data and fertility prediction.
        
        Args:
            soil_data (dict): Dictionary containing soil sensor data.
            fertility_prediction (dict): Dictionary containing fertility prediction results.
            recommendations (dict, optional): Dictionary containing crop recommendations.
            
        Returns:
            list: A list of messages for the Mistral AI API.
        """
        # Extract soil data
        temperature = soil_data.get('temperature', 'N/A')
        moisture = soil_data.get('moisture', 'N/A')
        nitrogen = soil_data.get('nitrogen', 'N/A')
        phosphorus = soil_data.get('phosphorus', 'N/A')
        potassium = soil_data.get('potassium', 'N/A')
        ph = soil_data.get('ph', 'N/A')
        
        # Extract fertility prediction
        fertility_label = fertility_prediction.get('fertility_label', 'Unknown')
        confidence = fertility_prediction.get('confidence', 0)
        
        # Create system message
        system_message = {
            "role": "system",
            "content": """You are an expert agricultural scientist specializing in soil health analysis. 
            Your task is to provide detailed analysis and actionable recommendations based on soil sensor data. 
            Your analysis should be comprehensive yet easy to understand for farmers. 
            Include specific recommendations for improving soil health and optimizing crop growth."""
        }
        
        # Create user message with soil data
        user_message_content = f"""Please analyze the following soil data and provide detailed insights and recommendations:

## Soil Sensor Data:
- Temperature: {temperature} °C
- Moisture: {moisture} %
- Nitrogen (N): {nitrogen} mg/kg
- Phosphorus (P): {phosphorus} mg/kg
- Potassium (K): {potassium} mg/kg
- pH: {ph}

## Soil Fertility Prediction:
- Predicted Fertility Class: {fertility_label}
- Prediction Confidence: {confidence}%
"""
        
        # Add crop recommendations if available
        if recommendations and 'crop_recommendations' in recommendations:
            user_message_content += "\n\n## Current Crop Recommendations:\n"
            for crop in recommendations['crop_recommendations']:
                user_message_content += f"- {crop['name']}\n"
        
        # Add soil improvement suggestions if available
        if recommendations and 'soil_improvement' in recommendations:
            user_message_content += "\n\n## Current Soil Improvement Suggestions:\n"
            for suggestion in recommendations['soil_improvement']:
                user_message_content += f"- {suggestion}\n"
        
        user_message_content += "\n\nPlease provide:\n1. A detailed analysis of the soil health based on these parameters\n2. Specific recommendations for improving soil quality\n3. Suitable crops for this soil profile (confirm or suggest alternatives to the current recommendations)\n4. Long-term soil management strategies"
        
        user_message = {
            "role": "user",
            "content": user_message_content
        }
        
        return [system_message, user_message]
    
    def analyze_soil(self, soil_data, fertility_prediction, recommendations=None, stream=False):
        """
        Analyze soil health using Mistral AI.
        
        Args:
            soil_data (dict): Dictionary containing soil sensor data.
            fertility_prediction (dict): Dictionary containing fertility prediction results.
            recommendations (dict, optional): Dictionary containing crop recommendations.
            stream (bool, optional): Whether to stream the response. Defaults to False.
            
        Returns:
            dict: The response from Mistral AI, or an error message.
        """
        if not self.api_key:
            return {"error": "API key not set. Please set the API key first."}
        
        # Create messages for the API
        messages = self.create_prompt(soil_data, fertility_prediction, recommendations)
        
        # Prepare the request payload
        payload = {
            "model": "mistral-medium",  # Using Mistral Medium model
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048,
            "stream": stream
        }
        
        # Set up headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            # Make the API request
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload)
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                if stream:
                    # Return the response object for streaming
                    return response
                else:
                    # Parse and return the response
                    result = response.json()
                    return self._parse_response(result)
            else:
                # Return error information
                return {
                    "error": f"API request failed with status code {response.status_code}",
                    "details": response.text
                }
        
        except Exception as e:
            # Return exception information
            return {"error": f"An error occurred: {str(e)}"}
    
    def stream_analysis(self, soil_data, fertility_prediction, recommendations=None):
        """
        Stream soil health analysis from Mistral AI.
        
        Args:
            soil_data (dict): Dictionary containing soil sensor data.
            fertility_prediction (dict): Dictionary containing fertility prediction results.
            recommendations (dict, optional): Dictionary containing crop recommendations.
            
        Yields:
            str: Chunks of the response as they are received.
        """
        response = self.analyze_soil(soil_data, fertility_prediction, recommendations, stream=True)
        
        if isinstance(response, dict) and "error" in response:
            yield json.dumps(response)
            return
        
        # Process the streaming response
        for line in response.iter_lines():
            if line:
                # Remove the "data: " prefix if present
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    line_text = line_text[6:]
                
                # Skip empty lines or [DONE] marker
                if line_text.strip() == '' or line_text.strip() == '[DONE]':
                    continue
                
                try:
                    # Parse the JSON chunk
                    chunk = json.loads(line_text)
                    
                    # Extract the content if available
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        content = delta.get('content', '')
                        if content:
                            yield content
                except json.JSONDecodeError:
                    # If not valid JSON, yield the raw line
                    yield line_text
    
    def _parse_response(self, response):
        """
        Parse the response from Mistral AI.
        
        Args:
            response (dict): The response from the Mistral AI API.
            
        Returns:
            dict: The parsed response with analysis content.
        """
        try:
            # Extract the content from the response
            if 'choices' in response and len(response['choices']) > 0:
                content = response['choices'][0]['message']['content']
                return {"analysis": content}
            else:
                return {"error": "No content found in the response", "raw_response": response}
        
        except Exception as e:
            return {"error": f"Failed to parse response: {str(e)}", "raw_response": response}

    def chat(self, model, messages, temperature=0.7, max_tokens=1024):
        """
        Custom implementation of chat functionality for Mistral AI.
        
        Args:
            model (str): The model to use for chat completion.
            messages (list): List of message dictionaries with role and content.
            temperature (float): Temperature parameter for response generation.
            max_tokens (int): Maximum number of tokens to generate.
            
        Returns:
            dict: A response object with choices containing the generated message.
        """
        if not self.api_key:
            raise ValueError("API key not set. Please set the API key first.")
        
        # Prepare the request payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Set up headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            # Make the API request
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload)
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                
                # Create a response object with the same structure expected by the application
                class ChatResponse:
                    class Choice:
                        class Message:
                            def __init__(self, content):
                                self.content = content
                        
                        def __init__(self, message_content):
                            self.message = self.Message(message_content)
                    
                    def __init__(self, choices):
                        self.choices = [self.Choice(choice["message"]["content"]) 
                                      for choice in choices]
                
                return ChatResponse(result["choices"])
            else:
                # Raise an exception with error details
                response.raise_for_status()
        
        except Exception as e:
            raise Exception(f"An error occurred during chat: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Initialize the Mistral AI integration
    mistral = MistralSoilAnalysis(api_key="your_api_key_here")
    
    # Example soil data
    soil_data = {
        'temperature': 25.5,
        'moisture': 65.2,
        'nitrogen': 150.0,
        'phosphorus': 20.5,
        'potassium': 300.0,
        'ph': 6.8
    }
    
    # Example fertility prediction
    fertility_prediction = {
        'fertility_class': 2,
        'fertility_label': 'Highly Fertile',
        'confidence': 92.5
    }
    
    # Example recommendations
    recommendations = {
        'crop_recommendations': [
            {'name': 'Wheat', 'description': 'Good for this soil profile'},
            {'name': 'Rice', 'description': 'Suitable for this region'}
        ],
        'soil_improvement': [
            'Add organic matter to improve soil structure',
            'Consider adding lime to adjust pH'
        ]
    }
    
    # Analyze soil health
    result = mistral.analyze_soil(soil_data, fertility_prediction, recommendations)
    
    # Print the result
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print("Soil Health Analysis:")
        print(result['analysis'])