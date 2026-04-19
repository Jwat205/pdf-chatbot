# PDF Chatbot using LlamaIndex

A RAG-powered chatbot that answers questions about PDF documents
using LlamaIndex and HuggingFace models.

## Stack
- LlamaIndex — RAG pipeline and agent framework
- HuggingFace Inference API — LLM (Qwen2.5) and embeddings (BGE)
- BAAI/bge-small-en-v1.5 — embedding model
- VectorStoreIndex — semantic search over document chunks

## Setup
1. Clone the repo
2. Install dependencies:
   pip install -r requirements.txt
3. Set your HuggingFace token:
   export HF_TOKEN=your_token_here
4. Add PDFs to the data/ folder
5. Run:
   python app.py

## How it Works
1. PDFs are loaded and chunked using SimpleDirectoryReader
2. Chunks are embedded using BAAI/bge-small-en-v1.5
3. Embeddings stored in VectorStoreIndex
4. Agent uses ReAct pattern to decide when to search documents
5. Index persisted to disk — no re-embedding on restart

## Architecture
User Query → Agent (ReAct) → search_pdf tool → VectorStoreIndex
→ Relevant chunks → LLM synthesis → Answer

## Example
```
You: What are the main topics covered in the document?
Assistant: Based on the document...
```
