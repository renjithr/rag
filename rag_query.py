from chromadb import PersistentClient
import httpx
import asyncio
from typing import List

# --- Connect to ChromaDB ---
client = PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(name="swift_chunks")

# --- Search ChromaDB ---
def search_context(query: str, top_k: int = 5) -> List[str]:
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    return [f"[{m.get('type', '')} {m.get('name', '')}]\n{doc}" for doc, m in zip(documents, metadatas)]

# --- Build Prompt ---
def build_prompt(query: str, contexts: List[str]) -> str:
    context_block = "\n\n".join(contexts)
    return f"""
You are an expert Swift engineer. Use the following code snippets as context to answer the question.

Context:
{context_block}

Question: {query}
Answer:
"""

# --- Query LLM ---
async def query_phi3(prompt: str) -> str:
    payload = {
        "model": "phi-3-mini-4k-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful and accurate assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 500
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post("http://192.168.1.5:1234/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}"

# --- Main flow (async) ---
async def rag_ask(query: str):
    contexts = search_context(query)
    prompt = build_prompt(query, contexts)
    print("\n--- Prompt Sent to LLM ---\n")
    print(prompt)
    print("\n--- Answer ---\n")
    answer = await query_phi3(prompt)
    print(answer)

# --- Entry Point ---
if __name__ == "__main__":
    asyncio.run(rag_ask("How is the BootstrapCard initialized and rendered?"))