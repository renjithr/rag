from chromadb import PersistentClient
from pprint import pprint

# --- Connect to the existing ChromaDB collection ---
client = PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(name="swift_chunks")

# --- Fetch all content (limit for demo) ---
results = collection.get(
    include=["documents", "metadatas"],
    limit=50  # change/remove this for full dump
)

# --- Display nicely ---
for doc, meta in zip(results["documents"], results["metadatas"]):
    print("\n===============================")
    print(f"📄 Type: {meta.get('type')} | Name: {meta.get('name')}")
    print(f"📂 File: {meta.get('filepath')} | Lines: {meta.get('start_line')}–{meta.get('end_line')}")
    print("🔍 Summary:")
    print(doc)
    print("===============================")
