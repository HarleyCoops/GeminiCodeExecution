# Gemini Code Execution Demo

This repository demonstrates how to use Gemini 2.0's code execution capabilities via the Gemini API. Code execution allows Gemini models to run Python code in a sandbox environment, enabling calculations, data analysis, and visualizations on the fly.

## What is Gemini Code Execution?

Code execution gives Gemini models access to a Python sandbox, allowing the models to:
- Run Python code and learn from the results
- Perform calculations and data analysis
- Create visualizations using Matplotlib
- Process and analyze user-uploaded files
- Debug code

The code execution environment includes popular libraries like NumPy, Pandas, and Matplotlib, making it powerful for data science and analytical tasks.

## Prerequisites

- Python 3.7+
- A Gemini API key (get one from [Google AI Studio](https://ai.google.dev/))
- Required Python packages (see `requirements.txt`)

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/gemini-code-execution-demo.git
   cd gemini-code-execution-demo
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API key:
   ```bash
   # On Windows PowerShell
   $env:GEMINI_API_KEY="your_api_key_here"
   
   # Alternatively, you can set it in your code
   ```

## How Code Execution Works

When code execution is enabled:

1. The Gemini model generates Python code based on your prompt
2. The code is executed in a secure sandbox environment
3. The execution results are returned to the model
4. The model uses these results to provide a more informed response

The code execution sandbox has the following limitations:
- Execution time limit: 30 seconds per execution
- Maximum 5 executions without re-prompting
- Limited to the pre-installed Python libraries
- No internet access from within the sandbox

## Basic Usage

Here's a simple example of using code execution with the Gemini API:

```python
from google import genai
from google.genai import types

# Initialize the client
client = genai.Client(api_key="YOUR_GEMINI_API_KEY")

# Create a prompt that requires code execution
prompt = """
What is the sum of the first 50 prime numbers?
Generate and run code for the calculation.
"""

# Generate content with code execution enabled
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=prompt,
    config=types.GenerateContentConfig(
        tools=[types.Tool(
            code_execution=types.ToolCodeExecution
        )]
    )
)

# Print the response
print(response.text)
```

## Example Use Cases

This repository includes examples for:

1. **Basic Calculations**: Solving mathematical problems
2. **Data Analysis**: Analyzing datasets with Pandas
3. **Visualizations**: Creating charts and graphs with Matplotlib
4. **File Processing**: Working with uploaded files
5. **Debugging**: Fixing code issues

## Available Libraries

The code execution environment includes the following pre-installed libraries:

- **Data Analysis & Processing**
  - NumPy - Numerical computing
  - Pandas - Data manipulation and analysis
  - Tabulate - Pretty-print tabular data
  
- **Machine Learning & Statistics**
  - Scikit-learn (sklearn) - Machine learning
  - StatsModels - Statistical models and tests
  
- **Visualization**
  - Matplotlib - Comprehensive visualization
  - Seaborn - Statistical data visualization
  - Altair - Declarative statistical visualization
  
- **Scientific Computing**
  - SciPy - Scientific computing
  - SymPy - Symbolic mathematics
  - MPMath - Arbitrary precision arithmetic
  
- **Computer Vision & Image Processing**
  - OpenCV (cv2) - Computer vision
  
- **Document Processing**
  - PDFMiner - PDF document parsing
  - ReportLab - PDF generation
  - StripRTF - RTF document parsing
  
- **Specialized Libraries**
  - Chess - Chess game representation and move generation

Note: You cannot install additional libraries in the sandbox environment. All code must use only these pre-installed libraries.

## Advanced Features

- **File I/O**: Process user-uploaded files
- **Graph Output**: Generate and display visualizations
- **Multimodal Capabilities**: Combine with other Gemini features

## Epic Experiment: The Chess Master's Climate Analysis

This repository includes an epic experiment that demonstrates the full power of Gemini's code execution capabilities by combining multiple libraries to create a comprehensive analysis spanning multiple domains:

### What the Experiment Does

The experiment simulates a scenario where a chess grandmaster is analyzing climate patterns to determine optimal tournament locations based on historical weather data, player performance metrics, and visual analysis of venues. It:

1. **Generates synthetic climate data** for five cities over 20 years
2. **Creates chess tournament data** with performance metrics for top players
3. **Analyzes the relationship** between temperature and chess performance
4. **Uses computer vision** to analyze chess positions
5. **Creates a mathematical model** using symbolic mathematics
6. **Generates a comprehensive PDF report** with visualizations and findings
7. **Uses Gemini** to analyze the results and provide insights

### Libraries Showcased

The experiment demonstrates the use of multiple libraries available in the Gemini code execution environment:

- **Data Analysis**: NumPy, Pandas, Tabulate
- **Visualization**: Matplotlib, Seaborn, Altair
- **Machine Learning**: Scikit-learn
- **Computer Vision**: OpenCV (cv2)
- **Mathematical Modeling**: SymPy, MPMath
- **Statistical Analysis**: StatsModels
- **Document Generation**: ReportLab
- **Chess Analysis**: Chess

### Running the Experiment

To run the epic experiment:

```bash
python epic_experiment.py
```

This will execute the full analysis pipeline and generate various outputs including data visualizations and a PDF report.

## Resources

- [Gemini API Documentation](https://ai.google.dev/docs/gemini_api)
- [Google Developers Blog: Gemini 2.0 Deep Dive](https://developers.googleblog.com/en/gemini-20-deep-dive-code-execution/)
- [Gemini API Developer Forum](https://ai.google.dev/community)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 