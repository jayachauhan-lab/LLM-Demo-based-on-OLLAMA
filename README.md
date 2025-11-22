# LLM-Demo-based-on-OLLAMA
This project demonstrates how to use a local Large Language Model (LLM), running via Ollama, to perform natural-language question answering and reasoning over a structured 100-item Acme office-supplies dataset stored in a CSV file.
The demo loads the dataset, converts it into a prompt-friendly table, and passes it to an LLM (LLaMA-3 by default). The model answers questions about product details, pricing, stock levels, and computes reorder quantities and costs.
No cloud API calls are used — everything runs locally, with no token fees or external dependencies.

------------------
## Features
- Local LLM inference using Ollama (no Internet required)
- Uses LLaMA-3 or any Ollama model you choose
- Loads a 100-item Acme office supply dataset from CSV
- Natural language queries about the dataset
-----------
## LLM performs:
- Product identification
- Highest-price/stock comparison
- Arithmetic (reorder qty & cost)
- Product detail summarization
- Automatic verification against ground-truth CSV values
- Robust Ollama binary & model checks
- Includes two complete example interactions with follow-ups
------------
**Project Structure**

├── llm_demo.py                     # Main LLM demo script

├── products.csv  # 100-row dataset (Acme office supplies)

└── README.md                   # Project documentation

-------------
 ## Requirements
1. Python 3.8+

Install Python if not already installed.

2. Ollama

Download from:
```bash
https://ollama.com/download
```

3. LLM model (default: llama3)

The script automatically checks for the model and pulls it if missing.

You may override the model using environment variables:
```bash
set OLLAMA_MODEL=llama3:8b
set OLLAMA_BIN=C:\path\to\ollama.exe
```
## Installation

Clone this repository:

git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>


Install any Python dependencies (none beyond the standard library are required).

Ensure products.csv is in the same directory as llm_demo.py.

## How to Run

Run the script:
```bash
python llm_demo.py
```

The script will:

Validate the Ollama binary
Pull the model if not present
Load the 100-item CSV dataset
Run Example 1 (highest-price product + follow-up)
Run Example 2 (product detail lookup + follow-up)
Compare LLM responses with computed ground truth

## Dataset

The dataset includes 100 real office supplies, all in your original naming style:

Acme Notebook
Acme Pen
Acme Stapler
Acme Marker
...
(100 total items)

Each record contains:
id — p1 through p100
name — item name (prefixed with Acme)
price — random realistic price (0.89–149.99)
stock — random stock amount (10–1000)

Example:

id,name,price,stock

p1,Acme Notebook,12.99,450

p2,Acme Pen,1.49,900
...

## Example Interactions
Example 1: Highest Price Query

User Prompt:
“Which product has the highest price, and what is the name and id?”

LLM Response:
Returns the correct Acme item based on CSV values.

Verification:
Script computes the correct answer and compares it to LLM output.

Follow-Up:

Compute reorder qty + reorder cost for the highest-stock product.

Example 2: Product Details Lookup

User Prompt:
“Provide a concise summary of product id p2.”

Follow-Up:
Compute reorder qty and cost for p2.
