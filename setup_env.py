#!/usr/bin/env python3
"""
Environment setup utility for Gemini Code Execution Demo.
This script helps users set up their environment for using Gemini API.
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if the Python version is compatible."""
    required_version = (3, 7)
    current_version = sys.version_info
    
    if current_version < required_version:
        print(f"Error: Python {required_version[0]}.{required_version[1]} or higher is required.")
        print(f"Current version: {current_version[0]}.{current_version[1]}.{current_version[2]}")
        return False
    
    print(f"✓ Python version {current_version[0]}.{current_version[1]}.{current_version[2]} is compatible.")
    return True

def install_requirements():
    """Install required packages from requirements.txt."""
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print(f"Error: {requirements_file} not found.")
        return False
    
    print(f"Installing required packages from {requirements_file}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("✓ All required packages installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return False

def setup_api_key():
    """Help the user set up their Gemini API key."""
    print("\nSetting up Gemini API key:")
    print("1. If you don't have an API key, get one from https://ai.google.dev/")
    print("2. You can set the API key in one of the following ways:")
    
    # Check if .env file exists
    if os.path.exists(".env"):
        print("   - .env file found. You can add your API key to this file.")
    else:
        print("   - Create a .env file with: GEMINI_API_KEY=your_api_key_here")
    
    # Show OS-specific environment variable instructions
    system = platform.system()
    if system == "Windows":
        print("   - Set environment variable in PowerShell: $env:GEMINI_API_KEY='your_api_key_here'")
        print("   - Set environment variable in CMD: set GEMINI_API_KEY=your_api_key_here")
    elif system in ["Linux", "Darwin"]:  # Darwin is macOS
        print("   - Set environment variable in terminal: export GEMINI_API_KEY='your_api_key_here'")
    
    # Ask if the user wants to create/update .env file
    create_env = input("\nDo you want to create/update .env file with your API key? (y/n): ").lower()
    if create_env == 'y':
        api_key = input("Enter your Gemini API key: ").strip()
        if api_key:
            try:
                with open(".env", "w") as f:
                    f.write(f"GEMINI_API_KEY={api_key}\n")
                print("✓ API key saved to .env file.")
                return True
            except Exception as e:
                print(f"Error saving API key: {e}")
                return False
        else:
            print("No API key provided. Skipping .env file creation.")
    
    return True

def verify_setup():
    """Verify that the setup is working by importing key modules."""
    try:
        print("\nVerifying setup by importing key modules...")
        
        # Try importing required modules
        import google.generativeai
        import numpy
        import pandas
        import matplotlib
        
        print("✓ All required modules imported successfully.")
        
        # Check if API key is set
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        
        if api_key:
            print("✓ Gemini API key is set in environment.")
            # Don't print the actual key for security reasons
        else:
            print("⚠ Gemini API key not found in environment.")
            print("  You'll need to set it before running the examples.")
        
        return True
    
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Please run the setup again or install missing packages manually.")
        return False

def main():
    """Main function to run the setup process."""
    print("=" * 50)
    print("Gemini Code Execution Demo - Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("⚠ There were issues installing requirements.")
        proceed = input("Do you want to continue with setup? (y/n): ").lower()
        if proceed != 'y':
            sys.exit(1)
    
    # Setup API key
    setup_api_key()
    
    # Verify setup
    verify_setup()
    
    print("\nSetup completed!")
    print("You can now run the examples:")
    print("1. python basic_example.py - Basic code execution examples")
    print("2. python advanced_example.py - Advanced examples with file processing")
    print("\nHappy coding with Gemini!")

if __name__ == "__main__":
    main() 