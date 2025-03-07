#!/usr/bin/env python3
"""
Basic example of using Gemini API with code execution capabilities.
This script demonstrates how to enable code execution for Gemini models.
"""

import os
import sys
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

def run_code_execution_example(client, prompt):
    """Run a code execution example with the given prompt."""
    try:
        # Configure the model to use code execution
        response = client.models.generate_content(
            model="gemini-2.0-flash",  # You can also use gemini-2.0-pro for more complex tasks
            contents=prompt,
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
        
        # You can also access the code execution parts specifically
        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'code_execution_results'):
                            print("\nCode Execution Results:")
                            print(part.code_execution_results)
        
        return response
    
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    """Main function to run the example."""
    client = setup_api()
    
    # Example prompts that demonstrate code execution
    examples = [
        """
        Calculate the sum of the first 20 prime numbers.
        Show the code and the result.
        """,
        
        """
        Create a visualization of a sine wave using matplotlib.
        Use different colors for the positive and negative parts of the wave.
        """,
        
        """
        Using pandas, create a small dataset of 5 students with columns for name, 
        age, and test scores. Then calculate the average score and identify the 
        student with the highest score.
        """
    ]
    
    # Run the first example by default
    selected_example = 0
    
    print("Gemini Code Execution Demo")
    print("Available examples:")
    for i, example in enumerate(examples):
        print(f"{i+1}. {example.strip().split('.')[0]}.")
    
    try:
        choice = int(input("\nSelect an example (1-3) or press Enter for default: ") or str(selected_example + 1))
        if 1 <= choice <= len(examples):
            selected_example = choice - 1
        else:
            print(f"Invalid choice. Using example 1.")
            selected_example = 0
    except ValueError:
        print(f"Invalid input. Using example 1.")
    
    print(f"\nRunning example {selected_example + 1}:")
    print(f"Prompt: {examples[selected_example].strip()}")
    
    run_code_execution_example(client, examples[selected_example])

if __name__ == "__main__":
    main() 