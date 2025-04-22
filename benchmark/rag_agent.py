import pdb
from typing import Dict, Any
from pydantic import BaseModel
from typing import List, Annotated, Sequence, TypeVar, cast
from langgraph.graph import Graph, StateGraph, END
from langgraph.graph.message import add_messages, BaseMessage
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langgraph.utils.runnable import RunnableCallable
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode
import asyncio
from functools import partial

from .base_agent import BaseAgent, AgentState, cut_off_tokens
from .vectordb import create_retriever_tool_node

# Define prompts outside the class to be reusable
KEYWORD_GENERATION_PROMPT = PromptTemplate.from_template(
    """You are an expert assistant helping to generate keywords for ICF section content creation. 
    Given the name of an ICF section, generate a list of relevant keywords that can be used 
    for semantic search to find related information in clinical trial protocols.
    
    ICF Section: {icf_section}
    
    Generate 5-10 specific, relevant keywords or phrases that would help in searching for 
    information related to this section in a clinical trial protocol. Focus on medical terminology
    and concepts that would appear in protocol documents.
    
    Keywords:"""
)

CONTENT_GENERATION_PROMPT = PromptTemplate.from_template(
    """You are an expert medical writer tasked with creating content for an Informed Consent Form (ICF).
    Your job is to write clear, accurate, and patient-friendly content for the specified ICF section.
    
    ICF Section: {icf_section}
    
    Below are relevant sections from the clinical trial protocol that contain information needed for this ICF section:
    
    {protocol_sections}
    
    Using ONLY the information from these protocol sections, create content for the ICF section.
    The content should:
    1. Be written at an 8th grade reading level
    2. Explain complex medical concepts in simple terms
    3. Be comprehensive but concise
    4. Follow typical ICF formatting and structure for this type of section
    5. Focus only on information relevant to the specified section
    
    Write the complete ICF section content:"""
)

class RAGAgent(BaseAgent):
    """
    RAGAgent is a class that uses a RAG pipeline to generate content.
    """

    name = "baseline_rag_agent"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent_graph = None
        self.retriever_tool = None
        
    def create_agent_graph(self) -> Graph:
        """
        Create a graph of the agent's components.
        """
        # Define the workflow
        workflow = StateGraph(AgentState)
        
        # Define node functions
        def extract_sections(state: AgentState) -> AgentState:
            """Extract target sections from the state."""
            messages = state["messages"]
            
            # Get the user query
            user_query = [m for m in messages if isinstance(m, HumanMessage)][-1].content
            
            # Parse the user query to extract the target ICF sections
            if "target_icf_sections" in state:
                target_sections = state["target_icf_sections"]
            else:
                # If not explicitly provided, try to parse from the query
                # Default to a single section if parsing fails
                target_sections = ["ICF Section"]
            
            return {
                "messages": messages,
                "target_icf_sections": target_sections
            }
        
        def generate_keywords(state: AgentState) -> AgentState:
            """Generate keywords for each target section."""
            messages = state["messages"]
            target_sections = state["target_icf_sections"]
            
            # Store section-keywords mapping
            section_keywords = {}
            
            # Generate keywords for each section
            for section in target_sections:
                # Format prompt with section name
                formatted_prompt = KEYWORD_GENERATION_PROMPT.format(icf_section=section)
                
                # Generate keywords
                response = self.llm.invoke(formatted_prompt)
                
                # Extract keywords
                keywords = response.content if hasattr(response, "content") else str(response)
                
                # Store keywords for this section
                section_keywords[section] = keywords
            
            # Update state with generated keywords
            return {
                "messages": messages,
                "section_keywords": section_keywords,
                "target_icf_sections": target_sections
            }
        
        def retrieve_documents(state: AgentState) -> AgentState:
            """Retrieve relevant documents using the keywords."""
            messages = state["messages"]
            section_keywords = state["section_keywords"]
            target_sections = state["target_icf_sections"]
            
            if not self.retriever_tool:
                raise ValueError("Retriever tool is not set")
            
            # Store section-documents mapping
            retrieved_documents = {}
            
            # Retrieve documents for each section
            for section in target_sections:
                keywords = section_keywords[section]
                
                # Use retriever tool to get documents
                retrieved_docs = self.retriever_tool.invoke({"query": keywords, "k": 10})
                
                # TODO: now the retrieved_docs is still a concapted str, need to be fixed
                
                # Store retrieved documents for this section
                retrieved_documents[section] = retrieved_docs
            
            # Update state with retrieved documents
            return {
                "messages": messages,
                "section_keywords": section_keywords,
                "retrieved_documents": retrieved_documents,
                "target_icf_sections": target_sections
            }
        
        def generate_content(state: AgentState) -> AgentState:
            """Generate ICF content based on the retrieved documents."""
            messages = state["messages"]
            retrieved_documents = state["retrieved_documents"]
            target_sections = state["target_icf_sections"]
            
            # Store section-content mapping
            generated_content = {}
            
            # Generate content for each section
            for section in target_sections:
                documents = retrieved_documents[section]
                
                # Format documents as text
                if isinstance(documents, list):
                    protocol_sections = "\n\n".join([f"DOCUMENT {i+1}:\n{doc.page_content}" 
                                             for i, doc in enumerate(documents)])
                else:
                    protocol_sections = documents # just a string
                
                # Truncate if too long to fit in context window
                protocol_sections = cut_off_tokens(protocol_sections, 6000)
                
                # Format prompt with section name and documents
                formatted_prompt = CONTENT_GENERATION_PROMPT.format(
                    icf_section=section,
                    protocol_sections=protocol_sections
                )
                
                # Generate content
                response = self.llm.invoke(formatted_prompt)
                
                # Extract content
                content = response.content if hasattr(response, "content") else str(response)
                
                # Store content for this section
                generated_content[section] = content
            
            # Create final response message
            result = "Generated ICF Sections:\n\n"
            for section in target_sections:
                result += f"## {section}\n\n{generated_content[section]}\n\n"
            
            final_message = AIMessage(content=result)
            
            # Update state with generated content and add final message
            return {
                "messages": messages + [final_message],
                "generated_content": generated_content
            }
        
        # Add nodes to the graph
        workflow.add_node("extract_sections", extract_sections)
        workflow.add_node("generate_keywords", generate_keywords)
        workflow.add_node("retrieve_documents", retrieve_documents)
        workflow.add_node("generate_content", generate_content)
        
        # Define edges
        workflow.add_edge("extract_sections", "generate_keywords")
        workflow.add_edge("generate_keywords", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "generate_content")
        workflow.add_edge("generate_content", END)
        
        # Set entry point
        workflow.set_entry_point("extract_sections")
        
        # Compile the graph
        return workflow.compile()
    
    def generate(self, **kwargs) -> Dict[str, Any]:
        """
        Generate ICF content using the RAG pipeline.
        
        Args:
            input_query (str): The user query to process
            retriever_tool: The retriever tool to use for document retrieval
            target_icf_sections (List[str]): List of ICF sections to generate
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, Any]: The result from the agent graph containing the generated content
        """
        # Extract required parameters
        input_query = kwargs.pop("input_query", None)
        if input_query is None:
            return {"error": "input_query is required"}
        
        self.retriever_tool = kwargs.pop("retriever_tool", None)
        if self.retriever_tool is None:
            return {"error": "retriever_tool is required"}
        
        target_icf_sections = kwargs.pop("target_icf_sections", [])
        if not target_icf_sections:
            return {"error": "target_icf_sections is required and cannot be empty"}
        
        # Create agent graph if not already created
        if self.agent_graph is None:
            self.agent_graph = self.create_agent_graph()
        
        try:
            # Prepare the initial state for the graph
            initial_state = {
                "messages": [HumanMessage(content=input_query)],
                "target_icf_sections": target_icf_sections
            }
            
            # Run the agent graph
            result = self.agent_graph.invoke(initial_state)
            return result
            
        except Exception as e:
            print(f"Error generating content: {e}")
            raise e
    
    