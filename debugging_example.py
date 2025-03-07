#!/usr/bin/env python3
"""
Debugging example using Gemini API with code execution capabilities.
This script demonstrates how Gemini can help debug problematic code.
"""

import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Sample buggy code examples for debugging
BUGGY_CODE_EXAMPLES = [
    {
        "title": "List Indexing Error",
        "code": """
def get_third_element(my_list):
    # This function should return the third element of a list
    return my_list[3]

# Test the function
test_list = [10, 20, 30, 40, 50]
print(f"The third element is: {get_third_element(test_list)}")

# Another test with a shorter list
short_list = [5, 10]
print(f"The third element is: {get_third_element(short_list)}")
        """
    },
    {
        "title": "Infinite Recursion",
        "code": """
def calculate_factorial(n):
    # This function should calculate the factorial of n
    if n <= 0:
        return 1
    else:
        # There's a bug in this recursive call
        return n * calculate_factorial(n)

# Test the function
try:
    result = calculate_factorial(5)
    print(f"Factorial of 5 is: {result}")
except Exception as e:
    print(f"Error occurred: {e}")
        """
    },
    {
        "title": "Dictionary Key Error",
        "code": """
def get_user_info(user_id, users_dict):
    # This function should return user info from a dictionary
    return {
        "name": users_dict[user_id]["name"],
        "email": users_dict[user_id]["email"],
        "age": users_dict[user_id]["age"]
    }

# Test data
users = {
    1: {"name": "Alice", "email": "alice@example.com"},
    2: {"name": "Bob", "email": "bob@example.com", "age": 30},
    3: {"name": "Charlie", "email": "charlie@example.com", "age": 25}
}

# Test the function
print(get_user_info(1, users))
print(get_user_info(2, users))
        """
    },
    {
        "title": "Logic Error in Sorting Algorithm",
        "code": """
def bubble_sort(arr):
    # This function should sort an array using bubble sort
    n = len(arr)
    
    # Traverse through all array elements
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n-i-1):
            # Traverse the array from 0 to n-i-1
            # Swap if the element found is greater than the next element
            if arr[j] < arr[j+1]:  # Bug: should be '>' for ascending order
                arr[j], arr[j+1] = arr[j+1], arr[j]
    
    return arr

# Test the function
test_array = [64, 34, 25, 12, 22, 11, 90]
sorted_array = bubble_sort(test_array)
print(f"Sorted array: {sorted_array}")
        """
    }
]

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

def debug_code_with_gemini(client, buggy_code, title):
    """Use Gemini to debug the provided code."""
    try:
        # Create a prompt that asks Gemini to debug the code
        prompt = f"""
        I have a Python code snippet with a bug. The code is titled "{title}".
        Please:
        1. Identify the bug(s) in the code
        2. Explain why the bug occurs
        3. Fix the code
        4. Test your fixed solution using code execution
        5. Explain what you changed and why it works now
        
        Here's the buggy code:
        ```python
        {buggy_code}
        ```
        """
        
        # Configure the model to use code execution
        response = client.models.generate_content(
            model="gemini-2.0-pro",  # Using pro model for better debugging
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    code_execution=types.ToolCodeExecution()
                )]
            )
        )
        
        # Print the response
        print("\nGemini's Debugging Analysis:")
        print("=" * 70)
        print(response.text)
        print("=" * 70)
        
        return response
    
    except Exception as e:
        print(f"Error: {e}")
        return None

def debug_custom_code(client):
    """Debug custom code provided by the user."""
    print("\nEnter your buggy code below (type 'END' on a new line when finished):")
    print("Example: def add(a, b): return a - b  # Bug: should be addition, not subtraction")
    
    code_lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        code_lines.append(line)
    
    buggy_code = "\n".join(code_lines)
    
    if not buggy_code.strip():
        print("No code provided. Using a default example.")
        buggy_code = """
def find_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total  # Bug: should return total / len(numbers)

# Test
test_numbers = [10, 20, 30, 40, 50]
print(f"Average: {find_average(test_numbers)}")
        """
    
    title = input("Enter a title for your code (or press Enter for default): ") or "Custom Code Debugging"
    
    return debug_code_with_gemini(client, buggy_code, title)

def main():
    """Main function to run the debugging examples."""
    client = setup_api()
    
    print("Gemini Code Debugging Demo")
    print("=" * 50)
    print("Select a buggy code example to debug:")
    
    for i, example in enumerate(BUGGY_CODE_EXAMPLES):
        print(f"{i+1}. {example['title']}")
    
    print(f"{len(BUGGY_CODE_EXAMPLES)+1}. Enter your own buggy code")
    print("=" * 50)
    
    try:
        choice = int(input("\nSelect an option: "))
        
        if 1 <= choice <= len(BUGGY_CODE_EXAMPLES):
            # Debug a predefined example
            example = BUGGY_CODE_EXAMPLES[choice-1]
            print(f"\nDebugging: {example['title']}")
            print("\nBuggy Code:")
            print("-" * 50)
            print(example['code'])
            print("-" * 50)
            
            debug_code_with_gemini(client, example['code'], example['title'])
            
        elif choice == len(BUGGY_CODE_EXAMPLES) + 1:
            # Debug custom code
            debug_custom_code(client)
            
        else:
            print("Invalid choice. Exiting.")
            
    except ValueError:
        print("Invalid input. Exiting.")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        
    print("\nDebugging demo completed.")

if __name__ == "__main__":
    main() 