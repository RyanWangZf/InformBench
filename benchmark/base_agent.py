import os
import json
from langchain_core.callbacks import FileCallbackHandler, StdOutCallbackHandler
from typing import Dict, Any, Callable, Literal, List
from langchain_core.language_models.base import BaseLanguageModel
from langchain_anthropic import ChatAnthropic
from langchain_together import Together
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain.schema import Document
import tiktoken

def cut_off_tokens(text: str, max_tokens: int, encoding_name: str = "gpt-4o"):
    encoding = tiktoken.encoding_for_model(encoding_name)
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        # cut off the last max_tokens tokens
        return encoding.decode(tokens[-max_tokens:])
    return text

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]
    target_icf_sections: List[str]
    section_keywords: List[str]
    retrieved_documents: Dict[str, List[Document]]
    generated_content: Dict[str, str]

class BaseAgent():
    
    def __init__(
        self,
        api_type: Literal["azure"],
        api_key: str,
        model_name: Literal["gpt-4o", "gpt-4o-mini", "o3-mini"] = None,
        endpoint: str=None,
        max_completion_tokens=5000,
        **kwargs
    ):  
        # get endpoint using model type
        self.endpoint = endpoint
        self.api_key = api_key

        # load model config
        self.model_name = model_name
        
        self.api_type = api_type
        
        self.max_completion_tokens = max_completion_tokens
        
        # get the model            
        self.llm = self.get_model(
            api=self.api_type,
            model_name=self.model_name,
            api_key=self.api_key,
            endpoint=self.endpoint,
            **kwargs
        )

    def get_model(
            self,
            api: str,
            api_key: str,
            model_name: str,
            endpoint: str = None,
            **kwargs
    ) -> BaseLanguageModel:
        """
        Get the appropriate language model based on the API type
        
        Args:
            api: The API provider ('together', 'anthropic', 'openai', 'google')
            api_key: The API key for the provider
            model: The model name
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            A language model instance
        """
        if (model_name not in ["o3-mini", "o3-preview"]):
            # remove max_completion_tokens from kwargs since it's not supported
            # by all models
            if "max_completion_tokens" in kwargs:
                del kwargs["max_completion_tokens"]
        
        llm = None
        if (api == "together"):
            llm = Together(
                model=model_name,
                together_api_key=api_key,
                **kwargs
            )
        elif (api == "anthropic"):
            llm = ChatAnthropic(
                model=model_name,
                api_key=api_key,
                **kwargs
            )
        elif (api == "openai"):
            llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                **kwargs
            )
        elif (api == "google"): 
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                **kwargs
            )
        elif (api == "azure"):
            llm = AzureChatOpenAI(
                azure_endpoint=endpoint,
                azure_deployment=model_name,
                api_key=api_key,
                api_version="2024-12-01-preview",
                **kwargs
            )
        else:
            raise ValueError(f"Invalid API: {api}")
        return llm

    def generate(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("This method should be implemented by the subclass")