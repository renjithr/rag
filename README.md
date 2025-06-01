# 🔍 Teach AI Your Internal Library (RAG for Custom Codebases)

This project shows how to turn your internal codebase into a searchable, intelligent assistant using **Retrieval-Augmented Generation (RAG)**, without retraining any models or exposing private code.

## 💡 Why This Matters

Most organizations use custom libraries that AI coding agents can't access. This project bridges that gap by:

* Parsing your code into meaningful chunks
* Generating natural language summaries with a local LLM
* Storing and retrieving these chunks with a vector DB
* Serving them to an LLM via API for intelligent Q\&A

---

## 🧱 Architecture

```
Codebase → Parser → Chunks → LLM Summaries → Vector DB
                                        ↓
                                 RAG API via FastAPI
                                        ↓
                             → Smart AI Answers
```

You can also inject this context into any **MCP server** used by coding agents.

---

## 📂 Components

### 1. `Parser.py` / `Parser_chroma.py`

* Parses `.swift` files (or any similar structured language)
* Extracts declarations (classes, funcs, etc.)
* Summarizes each chunk using Phi-3-mini
* Outputs to markdown or ChromaDB

### 2. `rag_server.py`

* FastAPI server
* Accepts natural language queries
* Uses semantic search from ChromaDB
* Builds prompts and sends to the LLM
* Returns helpful contextual answers

---

## 🚀 How to Run

### Prerequisites

* Python 3.10+
* Local LLM running at `http://localhost:1234` (e.g., Phi-3-mini via LM Studio)
* `chroma` for vector storage

```bash
pip install httpx chromadb fastapi uvicorn
```

### Step 1: Parse & Summarize

```bash
python Parser_chroma.py
```

This will:

* Walk through the `./BP` directory
* Extract Swift code chunks
* Generate summaries
* Store them in ChromaDB

### Step 2: Start the RAG API Server

```bash
uvicorn rag_server:app --reload
```

### Step 3: Ask Questions

Send a POST request:

```json
POST /ask
{
  "query": "What does the LoginCard component do?",
  "top_k": 5
}
```

---

## 🧪 Tested On

* Custom Swift-like component library mimicking Bootstrap
* Local setup using `Phi-3-mini` LLM
* Works offline and secure (no cloud calls)

---

## 🧠 Future Additions

* Tree-sitter-based language support
* UI interface for interactive exploration
* Support for other languages beyond Swift

---

## 📜 License

MIT License — Use it, extend it, contribute to it.

---

## 🙌 Credits

Created by Renjith
For devs who want their AI to actually understand their code.
