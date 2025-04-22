import json
from typing import Dict, Any, Union, Optional
from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
import os

def call_llm_json_output(
    prompt_template: str,
    input_variables: Dict[str, str],
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 4000,
) -> Dict[str, Any]:
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
        azure_deployment=model,
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version="2024-12-01-preview",
        temperature=temperature,
        max_tokens=max_tokens
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
        input_variables=list(input_variables.keys())
    )
    
    # Format the prompt with input variables
    formatted_prompt = prompt.format(**input_variables)
    
    # Call the LLM
    response = llm.invoke(formatted_prompt)
    
    # Extract the content string from the response
    if hasattr(response, 'content'):
        content = response.content
    else:
        content = str(response)
    
    # Parse JSON response
    try:
        # Try to parse the entire response as JSON
        parsed_response = json.loads(content)
        return parsed_response
    except json.JSONDecodeError:
        # If that fails, try to extract JSON from within the text
        try:
            # Look for JSON-like structures
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("Could not extract JSON from response")
        except Exception as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nOriginal response: {content}") 