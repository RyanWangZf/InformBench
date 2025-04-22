"""
This file contains the code for parsing the risk section from the protocol.
"""

import os
import pdb
import json
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from benchmark.llm import call_llm_json_output

# Prompt for extracting procedure-risk pairs from protocol content
PROCEDURE_RISK_EXTRACTION_PROMPT = """
Your task is to identify procedure-risk pairs from the following protocol content.

For each procedure mentioned in the text, identify:
1. The name of the procedure
2. Any associated risks, side effects, or adverse events linked to that procedure

Return ONLY procedure-risk pairs that are explicitly mentioned in the text. Do not infer or assume risks not directly stated.

Protocol Content:
{content}

Return the response in the following JSON format:
```json
{{
  "procedure_risk_pairs": [
    {{
      "procedure": "Name of procedure 1",
      "risks": ["Risk 1", "Risk 2", ...]
    }},
    {{
      "procedure": "Name of procedure 2",
      "risks": ["Risk 1", "Risk 2", ...]
    }},
    ...
  ]
}}
```

If no procedure-risk pairs are found, return an empty list for "procedure_risk_pairs".
"""

def extract_procedure_risk_pairs(document: Document, llm: str) -> Tuple[Dict[str, List[str]], int]:
    """
    Extract procedure-risk pairs from a single document.
    
    Args:
        document: A Document object containing content and metadata
        llm: The LLM model to use for extraction
        
    Returns:
        A tuple containing:
        - Dictionary mapping procedures to risks
        - Page number the information was extracted from
    """
    page_number = document.metadata.get("page_number", -1)
    
    # Extract procedure-risk pairs using LLM
    response = call_llm_json_output(
        PROCEDURE_RISK_EXTRACTION_PROMPT,
        inputs={"content": document.page_content},
        llm=llm,
        max_completion_tokens=1024
    )
    
    # Parse the response
    try:
        result = json.loads(response)
        pairs = result.get("procedure_risk_pairs", [])
        
        # Convert to dictionary format
        procedure_risk_dict = {}
        for pair in pairs:
            procedure = pair.get("procedure", "")
            risks = pair.get("risks", [])
            
            if procedure and risks:
                procedure_risk_dict[procedure] = risks
                
        return procedure_risk_dict, page_number
    
    except json.JSONDecodeError:
        # Handle parsing errors
        print(f"Error parsing LLM response on page {page_number}")
        return {}, page_number

def extract_procedure_risk_pairs_from_documents(
    documents: List[Document], 
    llm: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Extract procedure-risk pairs from a list of documents.
    
    Args:
        documents: List of Document objects containing protocol content
        llm: The LLM model to use for extraction
        
    Returns:
        Dictionary containing:
        - procedure_risk_pairs: Dictionary mapping procedures to risks
        - sources: Dictionary mapping procedures to page numbers
    """
    all_procedure_risk_pairs = {}
    procedure_sources = {}
    
    print(f"Extracting procedure-risk pairs from {len(documents)} documents...")
    
    for doc in documents:
        # Extract pairs from this document
        pairs, page_number = extract_procedure_risk_pairs(doc, llm)
        
        # Add to the global collections
        for procedure, risks in pairs.items():
            # If procedure already exists, extend the risks list
            if procedure in all_procedure_risk_pairs:
                existing_risks = set(all_procedure_risk_pairs[procedure])
                for risk in risks:
                    if risk not in existing_risks:
                        all_procedure_risk_pairs[procedure].append(risk)
                        
                # Update the source page numbers
                procedure_sources[procedure].append(page_number)
            else:
                # New procedure
                all_procedure_risk_pairs[procedure] = risks
                procedure_sources[procedure] = [page_number]
    
    return {
        "procedure_risk_pairs": all_procedure_risk_pairs,
        "sources": procedure_sources
    }

def save_procedure_risk_pairs(
    extraction_results: Dict[str, Any],
    output_path: str = "./procedure_risk_pairs.json"
):
    """
    Save the extracted procedure-risk pairs to a JSON file.
    
    Args:
        extraction_results: Results from extract_procedure_risk_pairs_from_documents
        output_path: Path to save the JSON file
    """
    with open(output_path, "w") as f:
        json.dump(extraction_results, f, indent=2)
    
    print(f"Saved procedure-risk pairs to {output_path}")
    
    # Also print a summary
    procedure_count = len(extraction_results["procedure_risk_pairs"])
    risk_count = sum(len(risks) for risks in extraction_results["procedure_risk_pairs"].values())
    
    print(f"Summary: Found {procedure_count} procedures with {risk_count} associated risks")
    
    # Print the top 5 procedures by number of risks
    print("\nTop procedures by number of risks:")
    sorted_procedures = sorted(
        extraction_results["procedure_risk_pairs"].items(), 
        key=lambda x: len(x[1]), 
        reverse=True
    )
    
    for procedure, risks in sorted_procedures[:5]:
        source_pages = extraction_results["sources"].get(procedure, [])
        print(f"- {procedure}: {len(risks)} risks (from pages {source_pages})")

def extract_and_save_procedure_risk_pairs(
    documents: List[Document],
    output_path: str = "./procedure_risk_pairs.json",
    llm: str = "gpt-4o-mini"
):
    """
    Extract procedure-risk pairs from documents and save the results.
    
    Args:
        documents: List of Document objects containing protocol content
        output_path: Path to save the results
        llm: The LLM model to use for extraction
    """
    # Extract the pairs
    results = extract_procedure_risk_pairs_from_documents(documents, llm)
    
    # Save to file
    save_procedure_risk_pairs(results, output_path)
    
    return results

