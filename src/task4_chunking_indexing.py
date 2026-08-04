"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho cả tiếng Việt lẫn tiếng Anh)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

import hashlib
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Hybrid structural chunking: giữ ranh giới Chương/Mục/Điều/Khoản bằng heading,
# sau đó recursive split để mọi chunk không vượt quá giới hạn của pipeline.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "markdown_header_recursive"

# OpenAI embedding cân bằng tốt giữa chất lượng, tốc độ và chi phí.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# ChromaDB chạy local; cosine phù hợp với normalized dense embeddings.
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "vietnam_labor_law_youth_qa_openai"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        relative_path = md_file.relative_to(STANDARDIZED_DIR).as_posix()
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "source_path": relative_path,
                "type": "legal" if "legal" in md_file.parts else "news",
                "domain": "vietnam_labor_law",
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "heading_1"),
            ("##", "heading_2"),
            ("###", "heading_3"),
            ("####", "heading_4"),
        ],
        strip_headers=False,
    )
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\nĐiều ", "\nĐiều ",
            "\n\nKhoản ", "\nKhoản ",
            "\n\n", "\n", ". ", "; ", " ", "",
        ],
    )

    chunks = []
    for doc in documents:
        sections = header_splitter.split_text(doc["content"])
        chunk_index = 0
        for section in sections:
            for chunk_text in recursive_splitter.split_text(section.page_content):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        **section.metadata,
                        "chunk_index": chunk_index,
                    },
                })
                chunk_index += 1
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    if not chunks:
        return chunks

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI()
    texts = [chunk["content"] for chunk in chunks]
    embeddings = []
    batch_size = 100
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            dimensions=EMBEDDING_DIM,
        )
        embeddings.extend(item.embedding for item in response.data)
        print(f"  Embedded {min(start + batch_size, len(texts))}/{len(texts)} chunks")

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    if not chunks:
        raise ValueError("No chunks to index. Convert documents to data/standardized first.")
    if any("embedding" not in chunk for chunk in chunks):
        raise ValueError("Every chunk must have an embedding before indexing.")

    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "domain": "vietnam_labor_law",
            "embedding_model": EMBEDDING_MODEL,
            "chunking_method": CHUNKING_METHOD,
        },
    )

    ids = []
    for chunk in chunks:
        identity = (
            f"{chunk['metadata']['source_path']}:{chunk['metadata']['chunk_index']}"
        )
        ids.append(hashlib.sha1(identity.encode("utf-8")).hexdigest())

    collection.upsert(
        ids=ids,
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
    return collection


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
