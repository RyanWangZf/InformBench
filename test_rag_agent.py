import os
import pdb
from benchmark_datasets.data_loader import load_informbench_benchmark_data
from benchmark.rag_agent import RAGAgent
from benchmark.vectordb import create_vector_db, create_retriever_tool_node
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from dotenv import load_dotenv

# load the environment variables for calling azure openai
load_dotenv()

# Configuration
DATA_PATH = "./benchmark_datasets/data"
TARGET_NCTID = "NCT02788201"
TARGET_SECTIONS = ["Purpose of Research", "Duration of Study Involvement"]

def main():
    # 1. Load the data
    print("Loading data...")
    data = load_informbench_benchmark_data(
        data_path=DATA_PATH,
        target_nctids=[TARGET_NCTID],
        debug=True
    )
    
    # 2. Find the target trial
    target_trial = None
    for trial in data:
        if trial["nctid"] == TARGET_NCTID:
            target_trial = trial
            break
    
    if target_trial is None:
        print(f"Trial {TARGET_NCTID} not found in the data")
        return
    
    # 3. Extract protocol documents
    protocol_docs = target_trial["protocol"]
    
    # 4. Create vector database and retriever tool
    print("Creating vector database...")
    vectordb = create_vector_db(protocol_docs)
    retriever_tool = create_retriever_tool_node(vectordb)
    
    # 5. Initialize the RAG agent
    print("Initializing RAG agent...")
    rag_agent = RAGAgent(
        api_type="azure",
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        model_name="gpt-4o-mini",
        endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
    )
    
    # 6. Generate ICF content
    print(f"Generating ICF content for sections: {TARGET_SECTIONS}")
    result = rag_agent.generate(
        input_query=f"Generate ICF sections for trial {TARGET_NCTID}",
        retriever_tool=retriever_tool,
        target_icf_sections=TARGET_SECTIONS
    )
    
    # 7. Print the generated content
    print("\n--- GENERATED ICF CONTENT ---\n")
    
    if "messages" in result:
        # Extract the last AI message which contains the generated content
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        if ai_messages:
            print(ai_messages[-1].content)
        else:
            print("No AI-generated content found in the result")
    else:
        print("Unexpected result format")
        print(result)

if __name__ == "__main__":
    main()
