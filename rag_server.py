from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from chromadb import PersistentClient
import httpx
import asyncio
from typing import List

app = FastAPI(title="Swift RAG Assistant")

# --- ChromaDB Setup ---
client = PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(name="swift_chunks")

# --- Request schema ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# --- Search context ---
def search_context(query: str, top_k: int = 5) -> List[str]:
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    return [f"[{m.get('type', '')} {m.get('name', '')}]\n{doc}" for doc, m in zip(documents, metadatas)]

# --- Prompt builder ---
def build_prompt(query: str, contexts: List[str]) -> str:
    context_block = "\n\n".join(contexts)
    return f"""
You are an expert Swift engineer. Use the following code snippets as context to answer the question.

Context:
{context_block}

Question: {query}
Answer:
"""

# --- LLM Query ---
async def query_phi3(prompt: str) -> str:
    payload = {
        "model": "phi-3-mini-4k-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful and accurate assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 800
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post("http://192.168.1.5:1234/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error querying model: {e}"

# --- FastAPI Endpoint ---
@app.post("/ask", response_class=PlainTextResponse)
async def ask_question(request: QueryRequest):
    contexts = search_context(request.query, request.top_k)
    if not contexts:
        raise HTTPException(status_code=404, detail="No context found for the query.")

    prompt = build_prompt(request.query, contexts)
    answer = await query_phi3(prompt)
    return answer