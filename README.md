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

The code execution environment includes these pre-installed libraries:

- NumPy
- Pandas
- Matplotlib
- SciPy
- Scikit-learn
- And more (see the full list in the API documentation)

## Advanced Features

- **File I/O**: Process user-uploaded files
- **Graph Output**: Generate and display visualizations
- **Multimodal Capabilities**: Combine with other Gemini features

## Resources

- [Gemini API Documentation](https://ai.google.dev/docs/gemini_api)
- [Google Developers Blog: Gemini 2.0 Deep Dive](https://developers.googleblog.com/en/gemini-20-deep-dive-code-execution/)
- [Gemini API Developer Forum](https://ai.google.dev/community)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 