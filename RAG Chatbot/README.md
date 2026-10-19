# RAG Chatbot

A document-based question-answering system (RAG) with Persian language support.

## Project Structure

| File | Role |
|------|------|
| `chunking.py` | Load TXT/PDF files and split text into chunks with overlap |
| `embedding.py` | Generate embeddings using `intfloat/multilingual-e5-base` |
| `vector_store.py` | Store and search chunks in ChromaDB (cosine similarity) |
| `retreval_and_generation.py` | Retrieve relevant chunks and generate answers with an LLM |

## Installation

```bash
pip install pypdf sentence-transformers chromadb anthropic ollama numpy
```

## Usage

### 1. Build the index

```python
from embedding import EmbeddingModel
from vector_store import VectorStore, build_index_from_files

model = EmbeddingModel()
store = VectorStore(collection_name="docs", persist_dir="./chroma_db")
build_index_from_files(["doc1.pdf", "doc2.txt"], model, store)
```

### 2. Ask questions

```python
from retreval_and_generation import RAGChatbot, OllamaProvider

bot = RAGChatbot(model, store, OllamaProvider(), top_k=4)
print(bot.ask("Your question"))
```

## LLM Options

- **Local (free):** `OllamaProvider` — requires [Ollama](https://ollama.com) and `ollama pull llama3.1:8b`
- **Claude API:** `ClaudeProvider` — requires `ANTHROPIC_API_KEY`

## Notes

- Default chunk_size: 800 characters, overlap: 150
- Embedding model is multilingual and works well for Persian.
- Answers are grounded only in the provided documents (no hallucination).