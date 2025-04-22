# InformBench

InformBench is a benchmark for evaluating the factuality and regulatory compliance of Large Language Models (LLMs) in clinical research document generation. The framework specifically focuses on informed consent form (ICF) generation and validation, using clinical trial protocols as source documents.

## Features

- PDF parsing and extraction of clinical document content
- Informed consent form (ICF) generation from clinical trial protocols
- Factuality evaluation of generated content against source protocols
- Regulatory compliance checking of generated informed consent forms
- RAG-based agent for context-aware document generation
- Multi-model support for generation and evaluation

## Environment Setup

### Prerequisites

- Python 3.12+
- pip or pipenv

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ryanwangzf/InformBench.git
   cd InformBench
   ```

2. Set up the environment with pipenv (recommended):
   ```bash
   pipenv install
   pipenv shell
   ```

   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables for LLM access:
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` with your Azure OpenAI API credentials:
     ```
     AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
     AZURE_OPENAI_API_KEY="your-api-key"
     AZURE_OPENAI_GPT4O_DEPLOYMENT="gpt-4o"
     AZURE_OPENAI_GPT4O_MINI_DEPLOYMENT="gpt-4o-mini"
     AZURE_OPENAI_O3_MINI_DEPLOYMENT="o3-mini"
     AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-small"
     ```


## Jupyter Notebook Examples

The repository includes several Jupyter notebooks demonstrating key functionality:

### 1. Schedule of Assessment (SOA) Extraction
**File:** `test_soa_extraction.ipynb`

This notebook demonstrates the extraction of Schedule of Assessment tables from clinical trial protocols. It shows how to parse structured information from complex PDF documents and convert it into a usable format for downstream tasks.

### 2. Procedure and Risk Extraction
**File:** `test_procedure_risk_extraction.ipynb`

Extract procedures and their associated risks from clinical trial protocols. This notebook shows how to identify and pair procedures with potential risks for accurate representation in informed consent documents.

### 3. Factuality Evaluation
**File:** `test_fact_eval.ipynb`

Evaluate the factual accuracy of generated content against source documents. The notebook demonstrates InformBench's factuality scoring framework, which checks whether generated content correctly represents information from the source protocol.

### 4. Compliance Evaluation
**File:** `test_compliance_eval.ipynb`

Check the regulatory compliance of generated informed consent forms. This notebook shows how to evaluate whether generated documents meet legal and ethical requirements for informed consent.

### 5. InformGen Agent
**File:** `test_informgen_agent.ipynb`

Demonstrates the InformGen agent, which generates informed consent form sections from clinical trial protocols. The notebook shows how to configure the agent, provide source documents, and generate compliant ICF content.

### 6. RAG Agent
**File:** `test_rag_agent.ipynb`

Shows the implementation of a Retrieval-Augmented Generation (RAG) agent for document generation. This notebook demonstrates how to configure vector databases, create retrievers, and generate context-aware content.

## Running the Examples

To run any notebook example:

1. Ensure your environment is activated:
   ```bash
   pipenv shell
   ```

2. Start Jupyter:
   ```bash
   jupyter notebook
   ```

3. Navigate to the desired notebook and run the cells