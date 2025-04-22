from benchmark.base_agent import BaseAgent
import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage
import pandas as pd
import logging

class InformGenAgent(BaseAgent):
    """
    Agent for generating informed consent form (ICF) sections based on templates.
    """
    
    def __init__(self, template_path=None, **kwargs):
        """
        Initialize the agent with a template.
        
        Args:
            template_path: Path to the template file (JSON)
            **kwargs: Additional arguments for the base agent
        """
        super().__init__(**kwargs)
        self.template = None
        
        # Load template if provided
        if template_path:
            self.load_template(template_path)
        else:
            # Default to Stanford template
            default_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "benchmark/templates/stanford.json"
            )
            if os.path.exists(default_path):
                self.load_template(default_path)
    
    def load_template(self, template_path: str) -> None:
        """
        Load a template from a file.
        
        Args:
            template_path: Path to the template file (JSON)
        """
        with open(template_path, 'r') as f:
            self.template = json.load(f)
            
        # Validate template structure
        if not self.template or "sections" not in self.template:
            raise ValueError("Invalid template format: must contain 'sections' key")
    
    def generate(self, **kwargs) -> Dict[str, Any]:
        """
        Generate ICF sections based on the template.
        
        Args:
            input_query: The general query
            target_sections: List of section names to generate
            protocol_docs: Protocol documents for reference
            soa_table: Optional Schedule of Assessment table
            procedure_risk_pairs: Optional procedure to risk mapping
            retriever_tool: Retriever tool for finding relevant protocol information
            **kwargs: Additional arguments
            
        Returns:
            Dict with generated content for each section
        """
        # Check if template is loaded
        if not self.template:
            return {"error": "No template loaded"}
        
        # Extract required parameters
        input_query = kwargs.pop("input_query", None)
        if input_query is None:
            return {"error": "input_query is required"}
        
        target_sections = kwargs.pop("target_sections", [])
        if not target_sections:
            # Default to all sections in the template if none specified
            target_sections = [section["target_section"] for section in self.template["sections"]]
        
        protocol_docs = kwargs.pop("protocol_docs", [])
        retriever_tool = kwargs.pop("retriever_tool", None)
        
        # Optional inputs for specific sections
        soa_table = kwargs.pop("soa_table", None)
        procedure_risk_pairs = kwargs.pop("procedure_risk_pairs", None)
        
        # Dictionary to store generated content
        generated_content = {}
        all_messages = []
        
        # Store warnings for sections not found in template
        warnings = []
        
        # Process each requested section
        for section_name in target_sections:
            # Find the section in the template
            section_template = None
            for section in self.template["sections"]:
                if section["target_section"] == section_name:
                    section_template = section
                    break
            
            # Handle case where section is not found in template
            if not section_template:
                warning_msg = f"Section '{section_name}' not found in template. Falling back to default RAG generation."
                warnings.append(warning_msg)
                logging.warning(warning_msg)
                
                # Fall back to default RAG generation
                section_content = self._generate_default_rag_content(
                    section_name,
                    protocol_docs,
                    retriever_tool
                )
                
                generated_content[section_name] = section_content
                all_messages.append(f"## {section_name}\n\n{section_content['content']}\n\n")
                continue
            
            # Check if this is a constant section (pre-defined content)
            if section_template.get("constant_section_flag", False) and "constant_text" in section_template:
                generated_content[section_name] = {
                    "content": section_template["constant_text"],
                    "source": "template_constant"
                }
                all_messages.append(f"## {section_name}\n\n{section_template['constant_text']}\n\n")
                continue
            
            # Determine if section needs SOA table or procedure-risk pairs
            use_soa = section_template.get("procedure_section_flag", False) and soa_table is not None
            use_risks = section_template.get("risk_section_flag", False) and procedure_risk_pairs is not None
            
            # Generate content for this section
            section_content = self._generate_section_content(
                section_name, 
                section_template, 
                protocol_docs,
                retriever_tool,
                soa_table if use_soa else None,
                procedure_risk_pairs if use_risks else None
            )
            
            # Store the generated content
            generated_content[section_name] = section_content
            
            # Prepare message for response
            message = f"## {section_name}\n\n{section_content['content']}\n\n"
            all_messages.append(message)
        
        # Create a combined result
        result_message = "Generated ICF Sections:\n\n" + "\n".join(all_messages)
        final_message = AIMessage(content=result_message)
        
        result = {
            "messages": [HumanMessage(content=input_query), final_message],
            "generated_sections": generated_content
        }
        
        # Add warnings if any sections were not found in template
        if warnings:
            result["warnings"] = warnings
        
        return result
    
    def _generate_default_rag_content(
        self,
        section_name: str,
        protocol_docs: List[Any],
        retriever_tool: Any
    ) -> Dict[str, Any]:
        """
        Generate content for a section not found in the template using a simple RAG approach.
        
        Args:
            section_name: Name of the section to generate
            protocol_docs: Protocol documents
            retriever_tool: Tool for retrieving relevant information
            
        Returns:
            Dict with generated content and metadata
        """
        # Basic prompt for the section
        prompt = (
            f"Generate the '{section_name}' section for an Informed Consent Form (ICF).\n\n"
            f"This section was not found in the template, so create content based on the "
            f"provided protocol information. Make sure to:\n"
            f"- Write in clear, simple language at an 8th grade reading level\n"
            f"- Format the content appropriately for an ICF document\n"
            f"- Be comprehensive but concise\n"
            f"- Explain any medical terms you use\n"
        )
        
        # Try to retrieve relevant information using the section name as query
        retrieved_docs = []
        if retriever_tool:
            try:
                retrieved_docs = retriever_tool.run(section_name)
                
                # Convert retrieved docs to text
                retrieved_text = "\n\n".join([
                    f"DOCUMENT {i+1}:\n{doc.page_content}" 
                    for i, doc in enumerate(retrieved_docs)
                ])
                
                # Add retrieval results to prompt
                prompt += f"\n\nRELEVANT PROTOCOL INFORMATION:\n{retrieved_text}"
            except Exception as e:
                prompt += f"\n\nError retrieving documents: {str(e)}"
        
        # Generate content using the LLM
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        
        # Return the generated content with metadata
        return {
            "content": content,
            "source": "default_rag",
            "prompt": prompt,
            "retrieved_docs_count": len(retrieved_docs) if retrieved_docs else 0,
            "fallback": True
        }
    
    def _generate_section_content(
        self, 
        section_name: str, 
        section_template: Dict[str, Any], 
        protocol_docs: List[Any],
        retriever_tool: Any,
        soa_table: Optional[pd.DataFrame] = None,
        procedure_risk_pairs: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate content for a specific section.
        
        Args:
            section_name: Name of the section
            section_template: Template for the section
            protocol_docs: Protocol documents
            retriever_tool: Tool for retrieving relevant information
            soa_table: Optional Schedule of Assessment table
            procedure_risk_pairs: Optional procedure to risk mapping
            
        Returns:
            Dict with generated content and metadata
        """
        # Prepare the prompt based on the template
        prompt = self._prepare_section_prompt(section_name, section_template, soa_table, procedure_risk_pairs)
        
        # Retrieve relevant protocol information if retriever_tool is available
        retrieved_docs = []
        if retriever_tool and "retrieval_query" in section_template:
            query = section_template["retrieval_query"]
            try:
                retrieved_docs = retriever_tool.run(query)
                
                # Convert retrieved docs to text
                retrieved_text = "\n\n".join([
                    f"DOCUMENT {i+1}:\n{doc.page_content}" 
                    for i, doc in enumerate(retrieved_docs)
                ])
            except Exception as e:
                retrieved_text = f"Error retrieving documents: {str(e)}"
        else:
            retrieved_text = "No retriever tool available or no retrieval query specified."
        
        # Add retrieval results to prompt if available
        if retrieved_docs:
            prompt += f"\n\nRELEVANT PROTOCOL INFORMATION:\n{retrieved_text}"
        
        # Add SOA table information if provided and section needs it
        if soa_table is not None:
            try:
                soa_text = str(soa_table)
                prompt += f"\n\nSCHEDULE OF ASSESSMENT:\n{soa_text}"
            except Exception as e:
                prompt += f"\n\nError processing SOA table: {str(e)}"
        
        # Add procedure-risk pairs if provided and section needs it
        if procedure_risk_pairs is not None:
            try:
                risk_text = "\n".join([
                    f"Procedure: {proc}, Risks: {', '.join(risks)}"
                    for proc, risks in procedure_risk_pairs.items()
                ])
                prompt += f"\n\nPROCEDURE-RISK MAPPING:\n{risk_text}"
            except Exception as e:
                prompt += f"\n\nError processing procedure-risk pairs: {str(e)}"
        
        # Generate content using the LLM
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        
        # Return the generated content with metadata
        return {
            "content": content,
            "source": "generated",
            "prompt": prompt,
            "retrieved_docs_count": len(retrieved_docs) if retrieved_docs else 0
        }
    
    def _prepare_section_prompt(
        self, 
        section_name: str, 
        section_template: Dict[str, Any],
        soa_table: Optional[pd.DataFrame] = None,
        procedure_risk_pairs: Optional[Dict[str, List[str]]] = None
    ) -> str:
        """
        Prepare a prompt for generating a section.
        
        Args:
            section_name: Name of the section
            section_template: Template for the section
            soa_table: Optional Schedule of Assessment table
            procedure_risk_pairs: Optional procedure to risk mapping
            
        Returns:
            Prompt string
        """
        # Start with the basic prompt
        prompt = f"Generate the '{section_name}' section for an Informed Consent Form (ICF).\n\n"
        
        # Add guidance from the template
        if "content_guidance" in section_template:
            prompt += f"GUIDANCE:\n{section_template['content_guidance']}\n\n"
        elif "fallback_template_str" in section_template:
            prompt += f"TEMPLATE:\n{section_template['fallback_template_str']}\n\n"
        
        # Add validation rules if available
        if "validation_rules" in section_template:
            rules_text = "\n".join([
                f"- {rule['rule_name']}: {rule['rule_description']}"
                for rule in section_template["validation_rules"]
            ])
            prompt += f"RULES TO FOLLOW:\n{rules_text}\n\n"
        
        # Add key facts if available
        if "key_facts" in section_template:
            facts_text = "\n".join([
                f"- {fact['fact_name']} ({fact['fact_type']}): {fact['fact_description']}"
                for fact in section_template["key_facts"]
            ])
            prompt += f"KEY FACTS TO INCLUDE:\n{facts_text}\n\n"
        
        # Special instructions for sections with SOA table
        if soa_table is not None:
            prompt += "A Schedule of Assessment (SOA) table is provided. Use this information to accurately describe the study procedures and timeline.\n\n"
        
        # Special instructions for sections with procedure-risk pairs
        if procedure_risk_pairs is not None:
            prompt += "Procedure-to-risk mappings are provided. Ensure these specific risks are addressed in relation to each procedure.\n\n"
        
        # Add final instructions
        prompt += (
            "Write in clear, simple language at an 8th grade reading level. "
            "Avoid medical jargon or complex terminology without explanation. "
            "When writing your outputs, please cite the retrieved protocol information with the reference numbers in your output at the end of the specific sentences, in the format [citation:x]. If a sentence comes from multiple contexts, please list all applicable citations, like [citation:3][citation:5]. Other than code and specific names and citations, your answer must be written in the same language as the question."
            "Avoid using bullet points and headers too many times which may look like GPT-generated content."
        )
        
        return prompt
        
        