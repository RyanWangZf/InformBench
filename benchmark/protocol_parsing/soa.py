"""
This file contains the code for parsing the SOA from the protocol.
"""

import os
import pdb
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from benchmark.llm import call_llm_json_output

# Prompt for identifying documents that contain SOA tables
SOA_IDENTIFICATION_PROMPT = """
Your task is to determine if the following protocol content contains a Schedule of Assessments (SOA) or Schedule of Activities table.

SOA tables typically:
1. Show what procedures/assessments happen at each visit or timepoint
2. Have visit names/timepoints as column headers (like "Screening", "Baseline", "Week 4", etc.)
3. Have procedures/assessments as row labels (like "Informed Consent", "Physical Exam", "Blood Collection", etc.)
4. Often use "X" or checkmarks to indicate when a procedure occurs

Protocol Content:
{content}

Return your response in the following JSON format:
```json
{{
  "contains_soa_table": true/false,
  "confidence": "high"/"medium"/"low",
  "rationale": "Brief explanation of why you think this contains or doesn't contain an SOA table"
}}
```
"""

# Prompt for extracting SOA table structure
SOA_EXTRACTION_PROMPT = """
Your task is to extract and structure the Schedule of Assessments (SOA) or Schedule of Activities table from the following protocol content.

Protocol Content:
{content}

For this task:
1. Identify the table headers (timepoints/visits)
2. Identify the row labels (procedures/assessments)
3. Determine which procedures happen at which timepoints

Return the structured table in the following JSON format:
```json
{{
  "timepoints": ["Screening", "Baseline", "Week 4", ...],
  "procedures": [
    {{
      "name": "Physical Examination",
      "schedule": ["X", "X", "", "X", ...] 
    }},
    {{
      "name": "Blood Collection",
      "schedule": ["X", "X", "X", "", ...]
    }},
    ...
  ]
}}
```

Use "X" to indicate when a procedure occurs at a timepoint, and "" (empty string) when it doesn't.
The "schedule" arrays must have the same length as the "timepoints" array, with each position corresponding to the timepoint at the same index.
"""

def identify_soa_table(document: Document, llm: str) -> Tuple[bool, int, str]:
    """
    Determine if a document contains an SOA table.
    
    Args:
        document: A Document object containing content and metadata
        llm: The LLM model to use for extraction
        
    Returns:
        A tuple containing:
        - Boolean indicating if document contains SOA table
        - Page number
        - Confidence level (high/medium/low)
    """
    page_number = document.metadata.get("page_number", -1)
    
    # Ask LLM if the document contains an SOA table
    response = call_llm_json_output(
        SOA_IDENTIFICATION_PROMPT,
        inputs={"content": document.page_content},
        llm=llm,
        max_completion_tokens=512
    )
    
    try:
        result = json.loads(response)
        contains_soa = result.get("contains_soa_table", False)
        confidence = result.get("confidence", "low")
        
        return contains_soa, page_number, confidence
    except json.JSONDecodeError:
        print(f"Error parsing LLM response on page {page_number}")
        return False, page_number, "low"

def extract_soa_table(document: Document, llm: str) -> Tuple[Dict[str, Any], int]:
    """
    Extract SOA table structure from a document.
    
    Args:
        document: A Document object containing content and metadata
        llm: The LLM model to use for extraction
        
    Returns:
        A tuple containing:
        - Dictionary with the structured SOA table
        - Page number the information was extracted from
    """
    page_number = document.metadata.get("page_number", -1)
    
    # Extract SOA table using LLM
    response = call_llm_json_output(
        SOA_EXTRACTION_PROMPT,
        inputs={"content": document.page_content},
        llm=llm,
        max_completion_tokens=1024
    )
    
    try:
        result = json.loads(response)
        return result, page_number
    except json.JSONDecodeError:
        print(f"Error parsing LLM response on page {page_number}")
        return {}, page_number

def convert_to_dataframe(soa_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Convert the extracted SOA data to a pandas DataFrame.
    
    Args:
        soa_data: Dictionary containing timepoints and procedures
        
    Returns:
        pandas DataFrame with procedures as index and timepoints as columns
    """
    if not soa_data or "timepoints" not in soa_data or "procedures" not in soa_data:
        return None
    
    timepoints = soa_data.get("timepoints", [])
    procedures = soa_data.get("procedures", [])
    
    if not timepoints or not procedures:
        return None
    
    # Create a dictionary for DataFrame construction
    data = {}
    for i, timepoint in enumerate(timepoints):
        data[timepoint] = []
        for procedure in procedures:
            schedule = procedure.get("schedule", [])
            # Make sure we have a value for this timepoint
            value = schedule[i] if i < len(schedule) else ""
            data[timepoint].append(value)
    
    # Create the DataFrame
    df = pd.DataFrame(data, index=[p.get("name", f"Procedure {i+1}") for i, p in enumerate(procedures)])
    
    return df

def merge_soa_tables(tables: List[Tuple[Dict[str, Any], int]]) -> Tuple[Dict[str, Any], List[int]]:
    """
    Merge multiple SOA tables into one comprehensive table.
    
    Args:
        tables: List of (table_data, page_number) tuples
        
    Returns:
        Tuple containing:
        - Merged table data
        - List of source page numbers
    """
    if not tables:
        return {}, []
    
    # Start with the first table
    merged_data = tables[0][0]
    page_numbers = [tables[0][1]]
    
    # Return early if only one table
    if len(tables) == 1:
        return merged_data, page_numbers
    
    # Process additional tables
    for table_data, page_number in tables[1:]:
        page_numbers.append(page_number)
        
        # Skip if table is empty
        if not table_data or "timepoints" not in table_data or "procedures" not in table_data:
            continue
        
        # Merge timepoints (keeping order)
        current_timepoints = merged_data.get("timepoints", [])
        new_timepoints = table_data.get("timepoints", [])
        
        # Find timepoints that aren't in the merged table yet
        unique_new_timepoints = [tp for tp in new_timepoints if tp not in current_timepoints]
        
        # If we have new timepoints, we need to extend all existing procedures
        if unique_new_timepoints:
            merged_data["timepoints"] = current_timepoints + unique_new_timepoints
            
            # Extend the schedule for each existing procedure
            for procedure in merged_data.get("procedures", []):
                procedure["schedule"] = procedure.get("schedule", []) + [""] * len(unique_new_timepoints)
        
        # Add new procedures
        current_procedure_names = [p.get("name") for p in merged_data.get("procedures", [])]
        
        for new_procedure in table_data.get("procedures", []):
            new_name = new_procedure.get("name")
            
            # Skip if procedure already exists
            if new_name in current_procedure_names:
                continue
            
            # Create a full schedule for the new procedure
            full_schedule = []
            new_schedule = new_procedure.get("schedule", [])
            
            # Map new schedule to merged timepoints
            for tp in merged_data["timepoints"]:
                if tp in new_timepoints:
                    idx = new_timepoints.index(tp)
                    value = new_schedule[idx] if idx < len(new_schedule) else ""
                    full_schedule.append(value)
                else:
                    full_schedule.append("")
            
            # Add procedure to merged data
            new_procedure["schedule"] = full_schedule
            merged_data.setdefault("procedures", []).append(new_procedure)
    
    return merged_data, page_numbers

def extract_soa_from_documents(documents: List[Document], llm: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Extract SOA tables from a list of documents.
    
    Args:
        documents: List of Document objects containing protocol content
        llm: The LLM model to use for extraction
        
    Returns:
        Dictionary containing:
        - soa_table: The structured SOA table data
        - dataframe: The SOA table as a pandas DataFrame
        - source_pages: List of page numbers where SOA tables were found
    """
    soa_tables = []
    
    print(f"Searching for SOA tables in {len(documents)} documents...")
    
    # First, identify documents that contain SOA tables
    for doc in documents:
        contains_soa, page_number, confidence = identify_soa_table(doc, llm)
        
        if contains_soa and confidence in ["high", "medium"]:
            print(f"Found SOA table on page {page_number} (confidence: {confidence})")
            
            # Extract the table structure
            table_data, _ = extract_soa_table(doc, llm)
            
            if table_data and "timepoints" in table_data and "procedures" in table_data:
                soa_tables.append((table_data, page_number))
    
    # Merge all found tables
    print(f"Found {len(soa_tables)} SOA tables, merging...")
    merged_table, source_pages = merge_soa_tables(soa_tables)
    
    # Convert to DataFrame
    df = convert_to_dataframe(merged_table)
    
    return {
        "soa_table": merged_table,
        "dataframe": df,
        "source_pages": source_pages
    }

def save_soa_table(extraction_results: Dict[str, Any], output_path: str = "./soa_table.json"):
    """
    Save the extracted SOA table to files.
    
    Args:
        extraction_results: Results from extract_soa_from_documents
        output_path: Path to save the JSON file
    """
    # Save the JSON structure
    with open(output_path, "w") as f:
        # Create a version that can be serialized to JSON
        serializable_results = {
            "soa_table": extraction_results["soa_table"],
            "source_pages": extraction_results["source_pages"]
        }
        json.dump(serializable_results, f, indent=2)
    
    # Save the DataFrame as CSV if available
    df = extraction_results.get("dataframe")
    if df is not None:
        csv_path = output_path.replace(".json", ".csv")
        df.to_csv(csv_path)
        print(f"Saved SOA table as CSV to {csv_path}")
    
    print(f"Saved SOA table as JSON to {output_path}")
    
    # Print a summary
    source_pages = extraction_results.get("source_pages", [])
    procedures_count = len(extraction_results.get("soa_table", {}).get("procedures", []))
    timepoints_count = len(extraction_results.get("soa_table", {}).get("timepoints", []))
    
    print(f"\nSummary: Found SOA table with {procedures_count} procedures across {timepoints_count} timepoints")
    print(f"Source pages: {source_pages}")
    
    # Print the DataFrame
    if df is not None:
        print("\nSOA Table Preview:")
        print(df.head(10))

def extract_and_save_soa_table(
    documents: List[Document],
    output_path: str = "./soa_table.json",
    llm: str = "gpt-4o-mini"
):
    """
    Extract SOA table from documents and save the results.
    
    Args:
        documents: List of Document objects containing protocol content
        output_path: Path to save the results
        llm: The LLM model to use for extraction
        
    Returns:
        The extraction results
    """
    # Extract the SOA table
    results = extract_soa_from_documents(documents, llm)
    
    # Save to file
    save_soa_table(results, output_path)
    
    return results
