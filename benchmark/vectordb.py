# create a chroma db based on the input protocol data
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from langchain.schema import Document
from typing import List
import os


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

def create_retriever_tool_node(vectordb):
    """
    Create a retriever tool for a given protocol data.

    Args:
        vectordb: A vector database containing the documents to search.

    Returns:
        A retriever tool.
    """
    # Create retriever from the vector database
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    
    # Create and return a retriever tool
    return create_retriever_tool(
        retriever=retriever,
        name="protocol_retriever",
        description="Retrieves information from clinical trial protocol documents based on the query",
        response_format="content_and_artifact",
    )


def create_retriever_tool_node_with_metadata_filter(vectordb):
    """
    Create a retriever tool that supports metadata filtering for protocol headers.
    
    Args:
        vectordb: A vector database containing the documents to search.
        
    Returns:
        A retriever tool that supports metadata filtering.
    """
    # Get the collection to extract metadata information
    collection = vectordb._collection
    
    # Create a metadata-aware retriever
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 10,
            "filter": None  # This will be populated at query time
        }
    )
    
    # Create a custom retriever that handles metadata filtering
    class MetadataFilterRetriever:
        def __init__(self, base_retriever):
            self.base_retriever = base_retriever
            
        def invoke(self, query_dict):
            """
            Invoke the retriever with optional metadata filters.
            
            Args:
                query_dict: A dictionary containing:
                    - query: The search query
                    - k: Number of results (optional)
                    - filter_header_1: Filter for header_1 (optional)
                    - filter_header_2: Filter for header_2 (optional)
            
            Returns:
                List of retrieved documents
            """
            query = query_dict.get("query", "")
            k = query_dict.get("k", 10)
            
            # Extract metadata filters
            metadata_filter = {}
            
            # Check for header filters
            if "filter_header_1" in query_dict and query_dict["filter_header_1"]:
                filter_header_1 = query_dict["filter_header_1"]
                # Handle both single string and list of strings
                if isinstance(filter_header_1, list):
                    metadata_filter["header_1"] = {"$in": filter_header_1}
                else:
                    metadata_filter["header_1"] = filter_header_1
                
            if "filter_header_2" in query_dict and query_dict["filter_header_2"]:
                filter_header_2 = query_dict["filter_header_2"]
                # Handle both single string and list of strings
                if isinstance(filter_header_2, list):
                    metadata_filter["header_2"] = {"$in": filter_header_2}
                else:
                    metadata_filter["header_2"] = filter_header_2
            
            # Set filter if any metadata filters are specified
            filter_dict = metadata_filter if metadata_filter else None
            
            # Update search kwargs with filters
            self.base_retriever.search_kwargs["filter"] = filter_dict
            self.base_retriever.search_kwargs["k"] = k
            
            # Perform retrieval
            return self.base_retriever.invoke(query)
    
    # Create the custom retriever
    metadata_retriever = MetadataFilterRetriever(retriever)
    
    # Get available headers from the documents in the vectordb
    # This could be cached for efficiency
    available_header_1 = set()
    available_header_2 = set()
    for doc in vectordb.get()["documents"]:
        metadata = doc.metadata if hasattr(doc, "metadata") else None
        if metadata:
            if "header_1" in metadata:
                available_header_1.add(metadata["header_1"])
            if "header_2" in metadata:
                available_header_2.add(metadata["header_2"])
    
    # Format available headers for the description
    header_1_list = ", ".join(f'"{h}"' for h in sorted(available_header_1) if h)
    header_2_list = ", ".join(f'"{h}"' for h in sorted(available_header_2) if h)
    
    # Create comprehensive tool description with filter instructions
    description = f"""Retrieves information from clinical trial protocol documents based on the query.
    
    You can filter by document headers to narrow down your search:
    
    Available header 1 in the protocol: {header_1_list}
    Available header 2 in the protocol: {header_2_list}
    
    To use filtering, include the filter parameters in your query:
    - filter_header_1: Select documents by the first level header
    - filter_header_2: Select documents by the second level header
    
    Example usage:
    {{
        "query": "", \\ the query to search for
        "k": 5, \\ the number of results to return
        "filter_header_1": ["", ""] \\ the list of first level headers to filter by
        "filter_header_2": ["", ""] \\ the list of second level headers to filter by
    }}
    """
    
    # Create and return the retriever tool
    return create_retriever_tool(
        retriever=metadata_retriever,
        name="protocol_metadata_retriever",
        description=description,
        response_format="content_and_artifact",
    )

