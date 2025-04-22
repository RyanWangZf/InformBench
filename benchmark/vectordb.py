# create a chroma db based on the input protocol data
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.schema import Document
from typing import List, Dict, Any
import os
from langchain.tools import Tool

def create_vector_db(docs: List[Document]):
    """
    Create a vector database from a list of documents.
    """
    # Initialize OpenAI embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY")
    )
    
    # Create Chroma DB from documents
    vectordb = Chroma.from_documents(
        docs, 
        embeddings,
        collection_name="protocol_docs"
    )
    
    return vectordb

def create_custom_retriever_tool(vectordb):
    """
    Create a custom retriever tool for protocol documents.
    
    Args:
        vectordb: A vector database containing the documents to search.
        
    Returns:
        A Tool that retrieves documents.
    """
    def retrieve_documents(query: str, k: int = 10, filter_header_1: str = None, filter_header_2: str = None) -> List[Document]:
        """
        Retrieve documents from the vector database based on query and optional filters.
        
        Args:
            query: The search query
            k: Number of documents to retrieve (default: 10)
            filter_header_1: Filter by first level header
            filter_header_2: Filter by second level header
            
        Returns:
            List of retrieved documents
        """
        # Build metadata filter if any filters are provided
        metadata_filter = {}
        if filter_header_1:
            metadata_filter["header_1"] = filter_header_1
        if filter_header_2:
            metadata_filter["header_2"] = filter_header_2
            
        # Use similarity search
        filter_dict = metadata_filter if metadata_filter else None
        docs = vectordb.similarity_search(
            query=query,
            k=k,
            filter=filter_dict
        )
        
        # Return formatted documents
        return docs
    
    # Extract available headers from the documents in the vectordb for tool description
    available_header_1 = set()
    available_header_2 = set()
    
    try:
        for doc in vectordb.get()["documents"]:
            metadata = doc.metadata if hasattr(doc, "metadata") else None
            if metadata:
                if "header_1" in metadata:
                    available_header_1.add(metadata["header_1"])
                if "header_2" in metadata:
                    available_header_2.add(metadata["header_2"])
    except Exception as e:
        print(f"Warning: Could not extract headers from vector database: {e}")
    
    # Format available headers for the description
    header_1_list = ", ".join(f'"{h}"' for h in sorted(available_header_1) if h)
    header_2_list = ", ".join(f'"{h}"' for h in sorted(available_header_2) if h)
    
    # Create the tool
    return Tool(
        name="protocol_retriever",
        func=retrieve_documents,
        description=f"""Tool for retrieving relevant protocol documents.
        
        Available header 1 in the protocol: {header_1_list}
        Available header 2 in the protocol: {header_2_list}
        
        This tool takes the following arguments:
        - query (required): The search query to find relevant protocol information
        - filter_header_1 (optional): Filter by a specific first level header
        - filter_header_2 (optional): Filter by a specific second level header
        """
    )