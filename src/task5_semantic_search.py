"""Task 5 - Semantic Search (Dense Retrieval) và HyDE tùy chọn.

Module dùng cùng embedding model và ChromaDB collection với Task 4. Corpus được
chia bằng Markdown Header trước, sau đó Recursive Character Splitter; metadata
heading của mỗi chunk được giữ nguyên trong kết quả tìm kiếm. Các thư viện nặng
được import lazy để module vẫn import được khi môi trường chưa setup.
"""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from typing import Any

from .task4_chunking_indexing import (
    CHROMA_DIR,
    CHUNKING_METHOD,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)


HYDE_MODEL = "gpt-4o-mini"

HYDE_SYSTEM_PROMPT = """Bạn hỗ trợ truy xuất văn bản pháp luật lao động Việt Nam.
Với câu hỏi được cung cấp, hãy viết một đoạn trả lời giả định ngắn bằng tiếng Việt,
giống nội dung có thể xuất hiện trong văn bản pháp luật hoặc bài hướng dẫn chính
thống. Dùng đúng thuật ngữ pháp lý liên quan. Chỉ trả về đoạn văn giả định, không
thêm lời dẫn, trích dẫn giả hoặc nhận xét về độ chắc chắn."""


def _validate_inputs(query: str, top_k: int) -> str:
    """Kiem tra input va tra ve query da loai bo khoang trang thua."""
    if not isinstance(query, str):
        raise TypeError("query phai la chuoi")
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query khong duoc rong")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k phai la so nguyen")
    if top_k <= 0:
        raise ValueError("top_k phai lon hon 0")
    return cleaned_query


@lru_cache(maxsize=1)
def _get_embedding_model() -> Any:
    """Tai embedding model mot lan duy nhat trong moi process."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Chua cai sentence-transformers. Hay cai requirements.txt truoc khi chay Task 5."
        ) from exc
    return SentenceTransformer(EMBEDDING_MODEL)


def _validate_collection_config(collection: Any) -> None:
    """Phát hiện index cũ được tạo bằng model hoặc chunking khác Task 4."""
    metadata = getattr(collection, "metadata", None) or {}
    indexed_model = metadata.get("embedding_model")
    indexed_chunking = metadata.get("chunking_method")

    if indexed_model and indexed_model != EMBEDDING_MODEL:
        raise RuntimeError(
            f"Collection dùng embedding model '{indexed_model}', nhưng Task 5 cần "
            f"'{EMBEDDING_MODEL}'. Hãy xóa index cũ và chạy lại Task 4."
        )
    if indexed_chunking and indexed_chunking != CHUNKING_METHOD:
        raise RuntimeError(
            f"Collection dùng chunking '{indexed_chunking}', nhưng Task 4 đang dùng "
            f"'{CHUNKING_METHOD}'. Hãy xóa index cũ và chạy lại Task 4."
        )


@lru_cache(maxsize=1)
def _get_collection() -> Any:
    """Mo ChromaDB collection do Task 4 tao; khong tu dong index du lieu."""
    if not CHROMA_DIR.exists():
        raise RuntimeError(
            f"Chua co vector index tai {CHROMA_DIR}. Hay chay Task 4 truoc."
        )

    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "Chua cai chromadb. Hay cai requirements.txt truoc khi chay Task 5."
        ) from exc

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Khong tim thay ChromaDB collection '{COLLECTION_NAME}'. "
            "Hay chay Task 4 voi dung cau hinh."
        ) from exc

    if collection.count() <= 0:
        raise RuntimeError(
            f"ChromaDB collection '{COLLECTION_NAME}' chua co du lieu. "
            "Hay chay Task 4 truoc."
        )
    _validate_collection_config(collection)
    return collection


def _generate_hypothetical_document(query: str) -> str:
    """Sinh hypothetical document bằng OpenAI để dùng cho HyDE."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        # .env la tien ich; API key van co the duoc truyen qua system environment.
        pass

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Chưa cấu hình OPENAI_API_KEY cho HyDE")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Chua cai openai package; khong the su dung HyDE"
        ) from exc

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=HYDE_MODEL,
        messages=[
            {"role": "system", "content": HYDE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.2,
        max_tokens=250,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("OpenAI trả về hypothetical document rỗng")
    return content.strip()


def _to_vector(value: Any) -> list[float]:
    """Chuan hoa output cua SentenceTransformer thanh list de gui cho ChromaDB."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(number) for number in value]


def _first_result_list(results: dict, key: str) -> list:
    """Lay batch dau tien tu response cua ChromaDB mot cach an toan."""
    batches = results.get(key) or []
    return list(batches[0]) if batches else []


def semantic_search(
    query: str,
    top_k: int = 10,
    use_hyde: bool = False,
) -> list[dict]:
    """Tim chunk gan query nhat bang cosine similarity.

    Args:
        query: Cau truy van cua nguoi dung.
        top_k: So ket qua toi da, phai lon hon 0.
        use_hyde: Neu True, sinh hypothetical document truoc khi embedding. Neu
            OpenAI khong san sang, ham tu dong quay ve query goc.

    Returns:
        Danh sach ``{'content': str, 'score': float, 'metadata': dict}``, sap
        xep theo score giam dan.

    Raises:
        ValueError/TypeError: Input khong hop le.
        RuntimeError: Thieu dependency, vector index hoac collection Task 4.
    """
    cleaned_query = _validate_inputs(query, top_k)
    search_text = cleaned_query

    if use_hyde:
        try:
            search_text = _generate_hypothetical_document(cleaned_query)
        except Exception as exc:
            warnings.warn(
                f"HyDE khong kha dung ({exc}); fallback ve query goc.",
                RuntimeWarning,
                stacklevel=2,
            )

    collection = _get_collection()
    result_count = min(top_k, collection.count())
    model = _get_embedding_model()
    query_vector = _to_vector(
        model.encode(search_text, normalize_embeddings=True)
    )

    raw_results = collection.query(
        query_embeddings=[query_vector],
        n_results=result_count,
        include=["documents", "metadatas", "distances"],
    )
    documents = _first_result_list(raw_results, "documents")
    metadatas = _first_result_list(raw_results, "metadatas")
    distances = _first_result_list(raw_results, "distances")

    output = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        if document is None or distance is None:
            continue
        score = min(1.0, max(0.0, 1.0 - float(distance)))
        output.append(
            {
                "content": str(document),
                "score": round(score, 4),
                "metadata": dict(metadata or {}),
            }
        )

    output.sort(key=lambda item: item["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    results = semantic_search(
        "Mức lương tối thiểu vùng áp dụng cho người lao động như thế nào?",
        top_k=5,
    )
    for result in results:
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
