#!/usr/bin/env python3
"""
Launcher script for Gemini Code Execution Demo.
This script provides a simple interface to run any of the example scripts.
"""

import os
import sys
import subprocess
import platform

# Define the available examples
EXAMPLES = [
    {
        "name": "Setup Environment",
        "script": "setup_env.py",
        "description": "Set up your environment for using Gemini API"
    },
    {
        "name": "Basic Examples",
        "script": "basic_example.py",
        "description": "Simple examples of Gemini code execution"
    },
    {
        "name": "Advanced Examples",
        "script": "advanced_example.py",
        "description": "Advanced examples with file processing and data visualization"
    },
    {
        "name": "Debugging Examples",
        "script": "debugging_example.py",
        "description": "Examples of using Gemini to debug problematic code"
    },
    {
        "name": "Epic Experiment",
        "script": "epic_experiment.py",
        "description": "Comprehensive experiment combining multiple libraries and analyses"
    }
]

def clear_screen():
    """Clear the terminal screen based on the operating system."""
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

def check_files():
    """Check if all example files exist."""
    missing_files = []
    for example in EXAMPLES:
        if not os.path.exists(example["script"]):
            missing_files.append(example["script"])
    
    if missing_files:
        print("Warning: The following files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nSome examples may not work correctly.")
        input("Press Enter to continue...")

def run_script(script_path):
    """Run a Python script."""
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
    except FileNotFoundError:
        print(f"Error: Script file '{script_path}' not found.")
    
    input("\nPress Enter to return to the main menu...")

def show_menu():
    """Display the main menu and handle user input."""
    while True:
        clear_screen()
        print("=" * 60)
        print("             GEMINI CODE EXECUTION DEMO")
        print("=" * 60)
        print("Select an example to run:")
        print()
        
        for i, example in enumerate(EXAMPLES):
            print(f"{i+1}. {example['name']}")
            print(f"   {example['description']}")
            print()
        
        print("0. Exit")
        print("=" * 60)
        
        try:
            choice = input(f"Enter your choice (0-{len(EXAMPLES)}): ")
            
            if choice == "0":
                clear_screen()
                print("Thank you for trying the Gemini Code Execution Demo!")
                print("Visit https://ai.google.dev/ for more information about Gemini API.")
                break
            
            choice = int(choice)
            if 1 <= choice <= len(EXAMPLES):
                example = EXAMPLES[choice-1]
                clear_screen()
                print(f"Running: {example['name']}")
                print(f"Description: {example['description']}")
                print("=" * 60)
                run_script(example["script"])
            else:
                print("Invalid choice. Please try again.")
                input("Press Enter to continue...")
        
        except ValueError:
            print("Invalid input. Please enter a number.")
            input("Press Enter to continue...")
        
        except KeyboardInterrupt:
            clear_screen()
            print("\nExiting the demo. Goodbye!")
            break

def check_python_version():
    """Check if the Python version is compatible."""
    required_version = (3, 7)
    current_version = sys.version_info
    
    if current_version < required_version:
        print(f"Error: Python {required_version[0]}.{required_version[1]} or higher is required.")
        print(f"Current version: {current_version[0]}.{current_version[1]}.{current_version[2]}")
        return False
    
    return True

def main():
    """Main function to run the launcher."""
    # Check Python version
    if not check_python_version():
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Check if all example files exist
    check_files()
    
    # Show the main menu
    show_menu()

if __name__ == "__main__":
    main() 