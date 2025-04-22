"""
Load the protocol and ICF data.

Loaded data schema:
    - 'protocol': A dictionary containing the protocol data.
    - 'icf': A dictionary containing the ICF data.

'protocol' List[Document]: each element is a document with the content of the section in the protocol. with the header 1, header 2, page number, etc of the section.
'icf' List[Document]: each element is a document with the content of the section in the ICF. with the section title, section content.
"""
from typing import Dict, Any, List
import pandas as pd
import os
from langchain_core.documents import Document

def load_informbench_benchmark_data(
    data_path: str,
    target_nctids: List[str] = None,
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    Load the protocol and ICF data.

    Args:
        data_path: The path to the data.
        target_nctids: The NCTIDs to load. If None, all data will be loaded.
        debug: Whether to load the data in debug mode. If debug, will only return 10 trials' data.
            And for each protocol document, only keep the first 20 chunks to save cost.

    Returns:
        A list of dictionaries containing the protocol and ICF data. Each dictionary is a trial's data.
            Each dictionary has the following keys:
                - 'protocol': A list of documents with the content of the section in the protocol.
                - 'icf': A list of documents with the content of the section in the ICF.
    """
    # Load the ICF data
    icf_path = os.path.join(data_path, "icf_section_by_section_cleaned.csv")
    icf_df = pd.read_csv(icf_path)
    
    # Load the protocol data
    protocol_path = os.path.join(data_path, "protocol_section_by_section_cleaned.csv")
    protocol_df = pd.read_csv(protocol_path)
    
    # Filter by target_nctids if provided
    if target_nctids:
        icf_df = icf_df[icf_df['NCTID'].isin(target_nctids)]
        protocol_df = protocol_df[protocol_df['NCTID'].isin(target_nctids)]
    
    # Get unique NCTIDs
    all_nctids = set(icf_df['NCTID'].unique()).union(set(protocol_df['NCTID'].unique()))
    
    # If debug mode, limit to 10 trials
    if debug:
        all_nctids = list(all_nctids)[:10]
        icf_df = icf_df[icf_df['NCTID'].isin(all_nctids)]
        protocol_df = protocol_df[protocol_df['NCTID'].isin(all_nctids)]
    
    # Group dataframes by NCTID for faster access
    icf_grouped = dict(list(icf_df.groupby('NCTID')))
    protocol_grouped = dict(list(protocol_df.groupby('NCTID')))
    
    result = []
    # Process each trial
    for nctid in all_nctids:
        trial_data = {"nctid": nctid}
        
        # Process ICF data
        icf_docs = []
        if nctid in icf_grouped:
            icf_trial_df = icf_grouped[nctid]
            for _, row in icf_trial_df.iterrows():
                metadata = {
                    "nctid": nctid,
                    "standard_section": row["Standard Section"],
                    "subsections": row["SubSections"]
                }
                doc = Document(page_content=row["Content"], metadata=metadata)
                icf_docs.append(doc)
        trial_data["icf"] = icf_docs
        
        # Process protocol data
        protocol_docs = []
        if nctid in protocol_grouped:
            protocol_trial_df = protocol_grouped[nctid]
            for _, row in protocol_trial_df.iterrows():
                metadata = {
                    "nctid": nctid,
                    "header_1": row["Header_1"],
                    "header_2": row["Header_2"]
                }
                doc = Document(page_content=row["Content"], metadata=metadata)
                protocol_docs.append(doc)
                if debug and len(protocol_docs) >= 20: # only keep the first 20 chunks to save cost
                    break
        trial_data["protocol"] = protocol_docs
        
        result.append(trial_data)
    
    return result

