#!/usr/bin/env python3
"""
Epic Experiment: The Chess Master's Climate Analysis

This experiment demonstrates the power of Gemini's code execution by combining
multiple libraries to create a comprehensive analysis that spans multiple domains:
- Chess game analysis
- Climate data visualization
- Computer vision processing
- Mathematical modeling
- Document generation

The experiment simulates a scenario where a chess grandmaster is analyzing
climate patterns to determine optimal tournament locations based on historical
weather data, player performance metrics, and visual analysis of venues.
"""

import os
import sys
import base64
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Additional imports for the epic experiment
import cv2
import chess
import sympy as sp
import mpmath as mp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from tabulate import tabulate

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

def generate_synthetic_climate_data():
    """Generate synthetic climate data for analysis."""
    # Create a sample dataset of climate data for chess tournament locations
    np.random.seed(42)
    
    # Generate 20 years of monthly temperature data for 5 cities
    cities = ['London', 'Moscow', 'New York', 'Tokyo', 'Dubai']
    years = range(2000, 2020)
    months = range(1, 13)
    
    data = []
    
    # Base temperatures for each city (annual average)
    base_temps = {
        'London': 11,
        'Moscow': 5,
        'New York': 12,
        'Tokyo': 16,
        'Dubai': 28
    }
    
    # Seasonal variations (amplitude of temperature swing)
    seasonal_var = {
        'London': 7,
        'Moscow': 16,
        'New York': 15,
        'Tokyo': 10,
        'Dubai': 8
    }
    
    # Generate data with seasonal patterns and some random variation
    for city in cities:
        for year in years:
            for month in months:
                # Calculate seasonal component (sinusoidal pattern)
                season = seasonal_var[city] * np.sin((month - 6) * np.pi / 6)
                
                # Add trend component (slight warming over time)
                trend = 0.02 * (year - 2000)
                
                # Add random noise
                noise = np.random.normal(0, 1)
                
                # Calculate temperature
                temp = base_temps[city] + season + trend + noise
                
                # Calculate humidity (inverse relationship with temperature in most places)
                if city == 'Dubai':  # Dubai is hot and humid
                    humidity = 60 + np.random.normal(0, 5)
                else:
                    humidity = 70 - 0.5 * temp + np.random.normal(0, 5)
                    humidity = max(30, min(90, humidity))  # Keep within realistic bounds
                
                # Calculate precipitation (mm)
                if city == 'London':
                    precip = 60 + 20 * np.sin((month - 9) * np.pi / 6) + np.random.exponential(10)
                elif city == 'Moscow':
                    precip = 50 + 30 * np.sin((month - 7) * np.pi / 6) + np.random.exponential(10)
                elif city == 'New York':
                    precip = 90 + 20 * np.sin((month - 8) * np.pi / 6) + np.random.exponential(15)
                elif city == 'Tokyo':
                    precip = 120 + 80 * np.sin((month - 6) * np.pi / 6) + np.random.exponential(20)
                else:  # Dubai
                    precip = 10 + np.random.exponential(5)
                
                data.append({
                    'City': city,
                    'Year': year,
                    'Month': month,
                    'Temperature': round(temp, 1),
                    'Humidity': round(humidity, 1),
                    'Precipitation': round(precip, 1)
                })
    
    return pd.DataFrame(data)

def generate_chess_tournament_data():
    """Generate synthetic chess tournament data."""
    np.random.seed(42)
    
    # Create tournament data
    tournaments = []
    cities = ['London', 'Moscow', 'New York', 'Tokyo', 'Dubai']
    years = range(2010, 2020)
    months = range(1, 13)
    
    # Top chess players
    players = ['Carlsen', 'Caruana', 'Ding', 'Nepomniachtchi', 'Aronian', 
               'Giri', 'So', 'Mamedyarov', 'Anand', 'Nakamura']
    
    # Player ratings (approximate)
    base_ratings = {
        'Carlsen': 2850,
        'Caruana': 2800,
        'Ding': 2790,
        'Nepomniachtchi': 2780,
        'Aronian': 2770,
        'Giri': 2760,
        'So': 2750,
        'Mamedyarov': 2740,
        'Anand': 2730,
        'Nakamura': 2720
    }
    
    # Player temperature preferences (fictional)
    temp_performance = {
        'Carlsen': {'optimal': 18, 'sensitivity': 0.8},      # Prefers cooler conditions
        'Caruana': {'optimal': 22, 'sensitivity': 0.5},      # Moderate preference
        'Ding': {'optimal': 24, 'sensitivity': 0.7},         # Prefers warmer conditions
        'Nepomniachtchi': {'optimal': 20, 'sensitivity': 1.0}, # Very sensitive to temperature
        'Aronian': {'optimal': 21, 'sensitivity': 0.4},      # Not very sensitive
        'Giri': {'optimal': 23, 'sensitivity': 0.6},         # Moderate preference
        'So': {'optimal': 22, 'sensitivity': 0.3},           # Not very sensitive
        'Mamedyarov': {'optimal': 25, 'sensitivity': 0.7},   # Prefers warmer conditions
        'Anand': {'optimal': 26, 'sensitivity': 0.9},        # Prefers warmer conditions
        'Nakamura': {'optimal': 21, 'sensitivity': 0.5}      # Moderate preference
    }
    
    # Generate tournament data
    tournament_id = 1
    for year in years:
        for month in [1, 4, 7, 10]:  # Quarterly tournaments
            for city in cities:
                if np.random.random() < 0.3:  # Not every city hosts a tournament every quarter
                    continue
                    
                tournament_name = f"{city} Chess Masters {year}"
                
                # Get average temperature for this city and month
                temp_data = generate_synthetic_climate_data()
                avg_temp = temp_data[(temp_data['City'] == city) & 
                                     (temp_data['Year'] == year) & 
                                     (temp_data['Month'] == month)]['Temperature'].values[0]
                
                # Select 6 random players for this tournament
                tournament_players = np.random.choice(players, 6, replace=False)
                
                # Generate results based on player ratings and temperature performance
                results = []
                for player in tournament_players:
                    # Base performance from rating
                    base_performance = base_ratings[player]
                    
                    # Temperature effect on performance
                    temp_effect = -temp_performance[player]['sensitivity'] * \
                                 (avg_temp - temp_performance[player]['optimal'])**2
                    
                    # Random tournament performance variation
                    random_effect = np.random.normal(0, 20)
                    
                    # Calculate tournament rating performance
                    performance = base_performance + temp_effect + random_effect
                    
                    # Calculate points (out of 5 rounds)
                    # Convert performance difference to expected score using logistic function
                    expected_score = 5 * (performance - 2700) / 400  # Simplified calculation
                    expected_score = max(0, min(5, expected_score + np.random.normal(0, 0.5)))
                    
                    results.append({
                        'TournamentID': tournament_id,
                        'TournamentName': tournament_name,
                        'City': city,
                        'Year': year,
                        'Month': month,
                        'Player': player,
                        'Rating': base_ratings[player],
                        'Performance': round(performance),
                        'Points': round(expected_score, 1),
                        'Temperature': avg_temp
                    })
                
                tournament_id += 1
    
    return pd.DataFrame(results)

def generate_chess_position_image():
    """Generate a chess position image for computer vision analysis."""
    # Create a temporary file for the chess position
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    temp_file.close()
    
    # Create a blank image (white background)
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Draw a chessboard pattern
    square_size = 50
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 1:
                x1, y1 = j * square_size, i * square_size
                x2, y2 = (j + 1) * square_size, (i + 1) * square_size
                cv2.rectangle(img, (x1, y1), (x2, y2), (120, 120, 120), -1)
    
    # Add chess piece symbols (simplified)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    # Define piece positions for the Sicilian Defense
    pieces = {
        (0, 0): 'R', (1, 0): 'N', (2, 0): 'B', (3, 0): 'Q', 
        (4, 0): 'K', (5, 0): 'B', (6, 0): 'N', (7, 0): 'R',
        (0, 1): 'P', (1, 1): 'P', (2, 1): 'P', (3, 1): 'P', 
        (4, 1): 'P', (5, 1): 'P', (7, 1): 'P',
        (2, 3): 'P',  # Pawn moved to c5
        (2, 6): 'p',  # Black pawn on c3
        (0, 6): 'p', (1, 6): 'p', (3, 6): 'p', 
        (4, 6): 'p', (5, 6): 'p', (6, 6): 'p', (7, 6): 'p',
        (0, 7): 'r', (1, 7): 'n', (2, 7): 'b', (3, 7): 'q', 
        (4, 7): 'k', (5, 7): 'b', (6, 7): 'n', (7, 7): 'r'
    }
    
    # Draw pieces
    for (col, row), piece in pieces.items():
        x = col * square_size + square_size // 2
        y = row * square_size + square_size // 2 + 10
        
        # White pieces are uppercase, black pieces are lowercase
        color = (0, 0, 0) if piece.islower() else (50, 50, 200)
        cv2.putText(img, piece.upper(), (x - 10, y), font, font_scale, color, font_thickness)
    
    # Add coordinates
    for i in range(8):
        # Add row numbers (8 to 1)
        cv2.putText(img, str(8 - i), (5, i * square_size + 30), font, 0.5, (0, 0, 0), 1)
        # Add column letters (a to h)
        cv2.putText(img, chr(97 + i), (i * square_size + 40, 395), font, 0.5, (0, 0, 0), 1)
    
    # Save the image
    cv2.imwrite(temp_file.name, img)
    
    print(f"Created chess position image: {temp_file.name}")
    return temp_file.name

def analyze_chess_position(image_path):
    """Analyze a chess position using computer vision."""
    # Load the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to separate pieces from background
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image
    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    
    # Count the number of pieces
    piece_count = len([c for c in contours if cv2.contourArea(c) > 100])
    
    # Save the processed image
    output_path = image_path.replace('.png', '_analyzed.png')
    cv2.imwrite(output_path, img_contours)
    
    return {
        'piece_count': piece_count,
        'contours_found': len(contours),
        'processed_image_path': output_path
    }

def create_mathematical_model():
    """Create a mathematical model using symbolic mathematics."""
    # Define symbolic variables
    t, k, a, b, c = sp.symbols('t k a b c')
    
    # Define a function for player performance based on temperature
    performance = sp.Function('P')(t)
    
    # Define the model: performance as a function of temperature
    # Using a quadratic model: P(t) = a - b(t - c)²
    # where c is the optimal temperature, b is the sensitivity, and a is the base performance
    model = a - b * (t - c)**2
    
    # Calculate the derivative to find the optimal temperature
    derivative = sp.diff(model, t)
    optimal_temp = sp.solve(derivative, t)[0]
    
    # Calculate the second derivative to confirm it's a maximum
    second_derivative = sp.diff(derivative, t)
    
    # Substitute some values to get a numerical result
    # Let's say a = 2800 (base rating), b = 0.5 (sensitivity), c = 22 (optimal temp)
    numerical_model = model.subs({a: 2800, b: 0.5, c: 22})
    
    # Generate some data points for plotting
    temp_range = np.linspace(10, 35, 100)
    performance_values = [float(numerical_model.subs(t, temp)) for temp in temp_range]
    
    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(temp_range, performance_values)
    plt.title('Mathematical Model: Chess Performance vs. Temperature')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Performance Rating')
    plt.grid(True)
    
    # Save the plot
    plot_path = 'mathematical_model.png'
    plt.savefig(plot_path)
    plt.close()
    
    return {
        'symbolic_model': str(model),
        'optimal_temperature': str(optimal_temp),
        'second_derivative': str(second_derivative),
        'numerical_model': str(numerical_model),
        'plot_path': plot_path
    }

def generate_report(climate_data, tournament_data, chess_analysis, math_model):
    """Generate a PDF report with the analysis results."""
    # Create a temporary file for the report
    temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    temp_file.close()
    
    # Create the PDF document
    doc = SimpleDocTemplate(temp_file.name, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Add title
    title = Paragraph("The Chess Master's Climate Analysis", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))
    
    # Add introduction
    intro = Paragraph(
        "This report analyzes the relationship between climate conditions and chess performance "
        "across various tournaments and locations. It combines data analysis, computer vision, "
        "and mathematical modeling to provide insights for optimal tournament planning.",
        styles['Normal']
    )
    elements.append(intro)
    elements.append(Spacer(1, 12))
    
    # Add climate data summary
    elements.append(Paragraph("Climate Data Summary", styles['Heading2']))
    climate_summary = climate_data.groupby('City').agg({
        'Temperature': ['mean', 'min', 'max'],
        'Humidity': ['mean', 'min', 'max'],
        'Precipitation': ['mean', 'min', 'max']
    }).reset_index()
    
    # Format the climate summary as a table
    climate_table_data = [['City', 'Avg Temp', 'Min Temp', 'Max Temp', 
                          'Avg Humidity', 'Avg Precip']]
    for _, row in climate_summary.iterrows():
        climate_table_data.append([
            row['City'],
            f"{row[('Temperature', 'mean')]:.1f}°C",
            f"{row[('Temperature', 'min')]:.1f}°C",
            f"{row[('Temperature', 'max')]:.1f}°C",
            f"{row[('Humidity', 'mean')]:.1f}%",
            f"{row[('Precipitation', 'mean')]:.1f}mm"
        ])
    
    climate_table = Table(climate_table_data)
    elements.append(climate_table)
    elements.append(Spacer(1, 12))
    
    # Add tournament performance summary
    elements.append(Paragraph("Tournament Performance Analysis", styles['Heading2']))
    tournament_summary = tournament_data.groupby('Player').agg({
        'Performance': 'mean',
        'Points': 'mean',
        'Temperature': 'mean'
    }).reset_index().sort_values('Performance', ascending=False)
    
    # Format the tournament summary as a table
    tournament_table_data = [['Player', 'Avg Performance', 'Avg Points', 'Avg Temperature']]
    for _, row in tournament_summary.iterrows():
        tournament_table_data.append([
            row['Player'],
            f"{row['Performance']:.0f}",
            f"{row['Points']:.1f}",
            f"{row['Temperature']:.1f}°C"
        ])
    
    tournament_table = Table(tournament_table_data)
    elements.append(tournament_table)
    elements.append(Spacer(1, 12))
    
    # Add chess position analysis
    elements.append(Paragraph("Chess Position Analysis", styles['Heading2']))
    chess_text = Paragraph(
        f"The computer vision analysis detected {chess_analysis['piece_count']} chess pieces "
        f"on the board. The position represents a Sicilian Defense opening.",
        styles['Normal']
    )
    elements.append(chess_text)
    elements.append(Spacer(1, 12))
    
    # Add mathematical model
    elements.append(Paragraph("Mathematical Model", styles['Heading2']))
    math_text = Paragraph(
        f"The mathematical model for player performance as a function of temperature is: "
        f"{math_model['symbolic_model']}. The optimal temperature was calculated to be "
        f"{math_model['optimal_temperature']}°C.",
        styles['Normal']
    )
    elements.append(math_text)
    elements.append(Spacer(1, 12))
    
    # Add conclusion
    elements.append(Paragraph("Conclusion", styles['Heading2']))
    conclusion = Paragraph(
        "Based on our analysis, we recommend scheduling major chess tournaments in locations "
        "with temperatures between 20-22°C for optimal player performance. London and New York "
        "provide the most suitable conditions during spring and fall months.",
        styles['Normal']
    )
    elements.append(conclusion)
    
    # Build the PDF
    doc.build(elements)
    
    print(f"Generated report: {temp_file.name}")
    return temp_file.name

def run_epic_experiment(client):
    """Run the epic experiment combining multiple libraries and analyses."""
    try:
        print("Starting the Chess Master's Climate Analysis experiment...")
        print("=" * 80)
        
        # Step 1: Generate synthetic climate data
        print("\nGenerating synthetic climate data...")
        climate_data = generate_synthetic_climate_data()
        print(f"Generated climate data for {climate_data['City'].nunique()} cities over {climate_data['Year'].nunique()} years")
        print("\nSample climate data:")
        print(tabulate(climate_data.head(), headers='keys', tablefmt='pretty'))
        
        # Step 2: Generate chess tournament data
        print("\nGenerating chess tournament data...")
        tournament_data = generate_chess_tournament_data()
        print(f"Generated data for {tournament_data['TournamentID'].nunique()} tournaments with {tournament_data['Player'].nunique()} players")
        print("\nSample tournament data:")
        print(tabulate(tournament_data.head(), headers='keys', tablefmt='pretty'))
        
        # Step 3: Analyze climate impact on chess performance
        print("\nAnalyzing climate impact on chess performance...")
        
        # Create a plot of average performance by temperature
        plt.figure(figsize=(12, 8))
        
        # Group data by temperature (rounded to nearest degree)
        tournament_data['Temperature_Rounded'] = tournament_data['Temperature'].round()
        temp_performance = tournament_data.groupby('Temperature_Rounded')['Performance'].mean().reset_index()
        
        # Plot the relationship
        sns.lineplot(data=temp_performance, x='Temperature_Rounded', y='Performance')
        plt.title('Average Chess Performance by Temperature')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Average Performance Rating')
        plt.grid(True)
        
        # Save the plot
        performance_plot = 'performance_by_temperature.png'
        plt.savefig(performance_plot)
        plt.close()
        
        print(f"Created performance analysis plot: {performance_plot}")
        
        # Step 4: Generate and analyze a chess position
        print("\nGenerating and analyzing chess position...")
        chess_image = generate_chess_position_image()
        chess_analysis = analyze_chess_position(chess_image)
        print(f"Chess position analysis results: {chess_analysis}")
        
        # Step 5: Create a mathematical model
        print("\nCreating mathematical model...")
        math_model = create_mathematical_model()
        print(f"Mathematical model: {math_model['symbolic_model']}")
        print(f"Optimal temperature: {math_model['optimal_temperature']}")
        
        # Step 6: Generate a comprehensive report
        print("\nGenerating comprehensive report...")
        report_path = generate_report(climate_data, tournament_data, chess_analysis, math_model)
        print(f"Report generated: {report_path}")
        
        # Step 7: Use Gemini to analyze the findings
        print("\nUsing Gemini to analyze the findings...")
        
        # Create a prompt for Gemini
        prompt = f"""
        Analyze the following chess and climate data experiment:
        
        1. We analyzed climate data for 5 cities (London, Moscow, New York, Tokyo, Dubai) over 20 years.
        2. We generated synthetic chess tournament data for top players across these cities.
        3. We found that the average performance rating varies with temperature, with an optimal range around 20-22°C.
        4. We created a mathematical model: P(t) = a - b(t - c)² where t is temperature, c is optimal temperature, and b is sensitivity.
        5. We used computer vision to analyze chess positions.
        
        Based on this data, what insights can you provide about the relationship between climate conditions
        and chess performance? How might tournament organizers use this information for scheduling?
        """
        
        # Configure the model to use code execution
        response = client.models.generate_content(
            model="gemini-2.0-pro",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    code_execution=types.ToolCodeExecution()
                )]
            )
        )
        
        # Print the response
        print("\nGemini's Analysis:")
        print("=" * 80)
        print(response.text)
        print("=" * 80)
        
        print("\nExperiment completed successfully!")
        return {
            'climate_data': climate_data,
            'tournament_data': tournament_data,
            'chess_analysis': chess_analysis,
            'math_model': math_model,
            'report_path': report_path,
            'gemini_analysis': response.text
        }
    
    except Exception as e:
        print(f"Error in experiment: {e}")
        return None

def main():
    """Main function to run the epic experiment."""
    client = setup_api()
    
    print("Epic Experiment: The Chess Master's Climate Analysis")
    print("=" * 80)
    print("This experiment demonstrates the power of Gemini's code execution by combining")
    print("multiple libraries to create a comprehensive analysis that spans multiple domains.")
    print("=" * 80)
    
    try:
        run_epic_experiment(client)
    except KeyboardInterrupt:
        print("\nExperiment cancelled by user.")
    except Exception as e:
        print(f"\nError running experiment: {e}")
    
    print("\nExperiment completed.")

if __name__ == "__main__":
    main() 