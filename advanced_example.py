#!/usr/bin/env python3
"""
Advanced example of using Gemini API with code execution capabilities.
This script demonstrates file processing and data visualization.
"""

import os
import sys
import base64
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from google import genai
from google.genai import types

def setup_api():
    """Set up the Gemini API client with proper authentication."""
    # Load API key from environment variable or .env file
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please set your API key using:")
        print("$env:GEMINI_API_KEY='your_api_key_here'")
        print("Or create a .env file with GEMINI_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # Initialize the Gemini API client
    genai.configure(api_key=api_key)
    return genai.Client()

def create_sample_data():
    """Create a sample CSV file for demonstration purposes."""
    # Create a sample dataset
    np.random.seed(42)
    data = {
        'Date': pd.date_range(start='2023-01-01', periods=30, freq='D'),
        'Temperature': np.random.normal(20, 5, 30),
        'Humidity': np.random.normal(60, 10, 30),
        'Precipitation': np.random.exponential(1, 30),
        'WindSpeed': np.random.normal(15, 7, 30)
    }
    
    df = pd.DataFrame(data)
    
    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    print(f"Created sample weather data file: {temp_file.name}")
    return temp_file.name

def encode_file_to_base64(file_path):
    """Encode a file to base64 for sending to Gemini API."""
    with open(file_path, 'rb') as file:
        file_content = file.read()
    return base64.b64encode(file_content).decode('utf-8')

def run_file_analysis_example(client, file_path):
    """Run a code execution example that analyzes a file."""
    try:
        # Read and encode the file
        file_content = encode_file_to_base64(file_path)
        file_name = os.path.basename(file_path)
        
        # Create a prompt that asks Gemini to analyze the file
        prompt = f"""
        I'm providing a CSV file with weather data. Please:
        1. Load and explore the dataset
        2. Create a visualization showing the relationship between temperature and humidity
        3. Calculate the correlation between different weather metrics
        4. Identify any interesting patterns or anomalies in the data
        
        The file is named {file_name}.
        """
        
        # Configure the file input
        file_part = types.Part.from_data(
            mime_type="text/csv",
            data=file_content,
            file_name=file_name
        )
        
        # Configure the model to use code execution with file input
        response = client.models.generate_content(
            model="gemini-2.0-pro",  # Using pro model for more complex analysis
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_text(prompt),
                        file_part
                    ],
                    role="user"
                )
            ],
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    code_execution=types.ToolCodeExecution()
                )]
            )
        )
        
        # Print the response
        print("\nGemini Response:")
        print("=" * 50)
        print(response.text)
        print("=" * 50)
        
        return response
    
    except Exception as e:
        print(f"Error: {e}")
        return None

def run_custom_prompt_example(client):
    """Run a code execution example with a custom user prompt."""
    try:
        # Get user input
        print("\nEnter your own prompt for Gemini with code execution:")
        print("Example: 'Create a fractal tree visualization using matplotlib'")
        user_prompt = input("> ")
        
        if not user_prompt.strip():
            print("Empty prompt. Using default prompt.")
            user_prompt = "Create a fractal tree visualization using matplotlib with at least 3 levels of recursion."
        
        # Configure the model to use code execution
        response = client.models.generate_content(
            model="gemini-2.0-pro",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    code_execution=types.ToolCodeExecution()
                )]
            )
        )
        
        # Print the response
        print("\nGemini Response:")
        print("=" * 50)
        print(response.text)
        print("=" * 50)
        
        return response
    
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    """Main function to run the example."""
    client = setup_api()
    
    print("Gemini Advanced Code Execution Demo")
    print("=" * 50)
    print("1. Analyze a sample weather dataset (CSV file)")
    print("2. Run a custom prompt with code execution")
    print("=" * 50)
    
    try:
        choice = int(input("\nSelect an option (1-2): ") or "1")
        
        if choice == 1:
            # Create a sample data file and run analysis
            file_path = create_sample_data()
            run_file_analysis_example(client, file_path)
            
            # Clean up the temporary file
            try:
                os.unlink(file_path)
                print(f"Deleted temporary file: {file_path}")
            except Exception as e:
                print(f"Warning: Could not delete temporary file: {e}")
                
        elif choice == 2:
            # Run with custom user prompt
            run_custom_prompt_example(client)
            
        else:
            print("Invalid choice. Exiting.")
            
    except ValueError:
        print("Invalid input. Exiting.")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        
    print("\nDemo completed.")

if __name__ == "__main__":
    main() 