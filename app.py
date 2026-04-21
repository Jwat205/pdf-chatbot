import asyncio
import os
from huggingface_hub import AsyncInferenceClient
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings


class _PersistentAsyncInferenceClient(AsyncInferenceClient):
    """Subclass that ignores close() so the httpx session stays alive across agent steps."""
    async def close(self):
        pass


# Configure models
Settings.llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
)
Settings.llm._async_client = _PersistentAsyncInferenceClient(
    **Settings.llm._get_inference_client_kwargs()
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Load or build index
STORAGE_DIR = "storage"

if os.path.exists(STORAGE_DIR):
    print("Loading existing index...")
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context)
else:
    print("Building index from PDFs...")
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(STORAGE_DIR)

query_engine = index.as_query_engine()

# Define RAG tool
async def search_pdf(query: str) -> str:
    """Search through uploaded PDF documents to answer questions about their content."""
    try:
        response = await query_engine.aquery(query)
        return str(response)
    except Exception as e:
        return f"Error searching documents: {str(e)}"

# Build agent
agent = AgentWorkflow.from_tools_or_functions(
    [search_pdf],
    llm=Settings.llm,
    system_prompt="""You are a helpful PDF assistant.
    Use the search_pdf tool to answer questions about the uploaded documents.
    Always cite specific information from the documents in your answers."""
)

async def main():
    ctx = Context(agent)
    print("PDF Chatbot ready! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
        if not user_input:
            continue

        try:
            response = await agent.run(user_input, ctx=ctx)
            print(f"\nAssistant: {response}\n")
        except Exception as e:
            print(f"Error: {str(e)}\n")

if __name__ == "__main__":
    asyncio.run(main())
