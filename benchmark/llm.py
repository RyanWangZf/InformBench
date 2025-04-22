import json
from typing import Dict, Any
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
import os

def call_llm_json_output(
    prompt_template: str,
    inputs: Dict[str, str],
    llm: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_completion_tokens: int = 4000,
) -> str:
    """
    Call a language model with a prompt template and parse the output as JSON.
    
    Args:
        prompt_template: The template string to format with input variables
        input_variables: Dictionary of variables to substitute into the template
        model: Model name to use if llm is not provided
        temperature: Temperature for generation (lower = more deterministic)
        max_tokens: Maximum tokens to generate
        
    Returns:
        Parsed JSON response as a dictionary
    """
    llm = AzureChatOpenAI(
        azure_deployment=llm,
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version="2024-12-01-preview",
        temperature=temperature,
        model_kwargs={"max_completion_tokens": max_completion_tokens}
    )
    
    # Add JSON formatting instructions to the prompt
    json_instructions = """
    Your response should be a valid JSON object. 
    Format your entire response as a JSON object with appropriate keys and values.
    Do not include any text outside of the JSON object.
    """
    
    # Combine the original prompt with JSON instructions
    full_prompt = f"{prompt_template}\n\n{json_instructions}"
    
    # Create prompt template
    prompt = PromptTemplate(
        template=full_prompt,
        input_variables=list(inputs.keys())
    )
    
    # Format the prompt with input variables
    formatted_prompt = prompt.format(**inputs)
    
    # Call the LLM
    response = llm.invoke(formatted_prompt)
    
    # Extract the content string from the response
    if hasattr(response, 'content'):
        content = response.content
    else:
        content = str(response)
    
    return content