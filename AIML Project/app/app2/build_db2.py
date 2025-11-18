from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

print("🔧 BUILDING CHROMA DB...")

# 1. Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

# 2. Load cases.txt
with open("cases.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 3. Split into chunks
chunks = raw_text.split("=== CASE ===")

# Convert each chunk into a Document object
documents = [Document(page_content=c.strip()) for c in chunks if c.strip()]

print(f"📌 Total chunks created: {len(documents)}")

# 4. Build DB with ONE embedding per chunk
db = Chroma.from_documents(
    documents,
    embedding=embeddings,
    persist_directory="chroma_index"
)

db.persist()
print("✅ DONE! Chroma DB saved to chroma_index/")

# 5. Print chunks for manual verification
print("\n📌 VERIFYING CHUNKS...\n")
for i, c in enumerate(documents):
    print(f"\n================ CHUNK {i+1} ================\n")
    print(c.page_content)
    print("\n============================================\n")

print("\n🔍 VERIFYING EMBEDDINGS IN CHROMA...\n")

# Access the underlying Chroma collection
collection = db._collection

# How many embeddings were stored?
print("Total embeddings stored:", collection.count())

# Peek one record to inspect
sample = collection.peek(1)

embedding_vector = sample["embeddings"][0]
print("Embedding vector dimension:", len(embedding_vector))
print("Embedding preview (first 10 values):", embedding_vector[:10])
