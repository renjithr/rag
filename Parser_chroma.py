import os
import re
import asyncio
from typing import List, Dict, Any
import httpx
from chromadb import PersistentClient

# --- Configuration ---
SWIFT_CODEBASE_ROOT = "./BP"
LLM_API_URL = os.getenv("LLM_API_URL", "http://192.168.1.5:1234/v1/chat/completions")
MAX_FILE_SIZE = 15 * 1024  # 15 KB

# --- Initialize ChromaDB ---
client = PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(name="swift_chunks")

def add_to_chroma(doc: Dict[str, Any]):
    safe_metadata = {
        "filepath": doc.get("filepath", ""),
        "start_line": doc.get("start_line", 0),
        "end_line": doc.get("end_line", 0),
        "type": doc.get("type") or "",
        "name": doc.get("name") or ""
    }

    collection.add(
        documents=[doc["content"]],
        metadatas=[safe_metadata],
        ids=[f"{doc['filepath']}:{doc['start_line']}:{doc['end_line']}"]
    )

# --- Swift Parser ---
def parse_swift_file(filepath: str) -> List[Dict[str, Any]]:
    chunks = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        declaration_pattern = re.compile(
            r'^\s*(?:'
            r'(?:public|internal|fileprivate|private|open)?\s*'
            r'(?:final|static|class)?\s*'
            r'(?:class|struct|enum|protocol)\s+[A-Za-z0-9_]+.*?\{.*?'
            r'|'
            r'(?:public|internal|fileprivate|private|open)?\s*'
            r'(?:static|class)?\s*'
            r'func\s+[A-Za-z0-9_]+\s*\(.*?\)\s*(?:->\s*[^\{]+)?\s*\{.*?'
            r'|'
            r'(?:public|internal|fileprivate|private|open)?\s*'
            r'extension\s+[A-Za-z0-9_]+.*?\{.*?'
            r')',
            re.DOTALL | re.MULTILINE
        )

        matches = list(re.finditer(declaration_pattern, content))
        last_end = 0

        for match in matches:
            start, end = match.span()
            declaration_type = None
            name = None

            if any(kw in match.group(0) for kw in ['class', 'struct', 'enum', 'protocol']):
                decl_match = re.search(r'(class|struct|enum|protocol)\s+([A-Za-z0-9_]+)', match.group(0))
                if decl_match:
                    declaration_type = decl_match.group(1)
                    name = decl_match.group(2)
            elif 'func' in match.group(0):
                declaration_type = 'func'
                name_match = re.search(r'func\s+([A-Za-z0-9_]+)', match.group(0))
                if name_match:
                    name = name_match.group(1)
            elif 'extension' in match.group(0):
                declaration_type = 'extension'
                name_match = re.search(r'extension\s+([A-Za-z0-9_]+)', match.group(0))
                if name_match:
                    name = name_match.group(1)

            if start > last_end:
                raw_code = content[last_end:start].strip()
                if raw_code:
                    chunks.append({
                        "content": raw_code,
                        "filepath": filepath,
                        "start_line": len(content[:last_end].splitlines()) + 1,
                        "end_line": len(content[:start].splitlines()),
                        "type": "raw_code",
                        "name": None
                    })

            chunk_content = content[start:end].strip()
            if chunk_content:
                chunks.append({
                    "content": chunk_content,
                    "filepath": filepath,
                    "start_line": len(content[:start].splitlines()) + 1,
                    "end_line": len(content[:end].splitlines()),
                    "type": declaration_type or "unknown_declaration",
                    "name": name
                })

            last_end = end

        if last_end < len(content):
            raw_code = content[last_end:].strip()
            if raw_code:
                chunks.append({
                    "content": raw_code,
                    "filepath": filepath,
                    "start_line": len(content[:last_end].splitlines()) + 1,
                    "end_line": len(content.splitlines()),
                    "type": "raw_code",
                    "name": None
                })

        print(f"🔍 Found {len(chunks)} chunks in {filepath}")

    except Exception as e:
        print(f"Error parsing file {filepath}: {e}")

    return chunks

# --- LLM Summarizer ---
async def generate_text_summary(code_chunk_content: str) -> str:
    prompt = (
        f"Analyze the following Swift code snippet and provide a concise, "
        f"high-level summary of its purpose and functionality. "
        f"Focus on what it does, not how it's implemented in detail. "
        f"Keep the summary to 2–3 sentences.\n\n"
        f"```swift\n{code_chunk_content}\n```\n\nSummary:"
    )

    payload = {
        "model": "phi-3-mini-4k-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that summarizes Swift code."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 150
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(LLM_API_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"⚠️ Error calling LLM: {e}")
        return "Error generating summary from LLM."

# --- Main Processor ---
async def process_swift_codebase_and_generate_md(root_dir: str):
    print(f"📂 Scanning Swift codebase in: {root_dir}")
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".swift"):
                filepath = os.path.join(dirpath, filename)
                print(f"🔄 Parsing: {filepath}")
                chunks = parse_swift_file(filepath)

                for chunk in chunks:
                    print(f"💬 Generating summary for: {chunk['type']} {chunk['name']}")
                    summary = await generate_text_summary(chunk["content"])
                    print(f"🧠 Summary received for: {chunk['type']} {chunk['name']}")
                    chunk["llm_summary"] = summary
                    add_to_chroma(chunk)

# --- Entry Point ---
if __name__ == "__main__":
    print("🚀 Starting Swift parser and summarizer with ChromaDB...")
    asyncio.run(process_swift_codebase_and_generate_md(SWIFT_CODEBASE_ROOT))
    print("🏁 Finished processing.")
