"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import re
from pathlib import Path

from src.task4_chunking_indexing import chunk_documents, load_documents

# Corpus dùng chung cho cả BM25 và TF-IDF, load lazy từ data/standardized/
# (tái sử dụng đúng logic chunking của Task 4 để nhất quán với Task 9 hybrid merge).
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}

_bm25_index = None
_tfidf_vectorizer = None
_tfidf_matrix = None


def _tokenize(text: str) -> list[str]:
    """Tokenize tiếng Việt đơn giản: lowercase + tách từ, bỏ dấu câu."""
    return re.findall(r"\w+", text.lower())


def _load_corpus() -> list[dict]:
    """Load + chunk documents từ data/standardized/ (dùng lại Task 4), cache lại CORPUS."""
    global CORPUS
    if not CORPUS:
        docs = load_documents()
        CORPUS = chunk_documents(docs)
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def build_tfidf_index(corpus: list[dict]):
    """
    Xây dựng TF-IDF index từ corpus (dùng để so sánh với BM25 — bonus demo).

    Returns:
        (vectorizer, tfidf_matrix)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(tokenizer=_tokenize, lowercase=False, token_pattern=None)
    matrix = vectorizer.fit_transform([doc["content"] for doc in corpus])
    return vectorizer, matrix


def _get_bm25():
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = build_bm25_index(_load_corpus())
    return _bm25_index


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    corpus = _load_corpus()
    if not corpus:
        return []

    bm25 = _get_bm25()
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    import numpy as np

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": corpus[idx]["metadata"],
            })
    return results


def tfidf_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng TF-IDF + cosine similarity.

    Dùng để so sánh cơ chế với BM25 trong buổi demo (bonus +5 điểm):
    - TF-IDF: điểm dựa trên tần suất từ trong toàn corpus, không tính đến độ dài document.
    - BM25: có thêm term saturation (k1) và length normalization (b) → xử lý document
      dài/ngắn không đều tốt hơn TF-IDF thuần.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sorted descending.
    """
    global _tfidf_vectorizer, _tfidf_matrix

    corpus = _load_corpus()
    if not corpus:
        return []

    if _tfidf_vectorizer is None:
        _tfidf_vectorizer, _tfidf_matrix = build_tfidf_index(corpus)

    from sklearn.metrics.pairwise import cosine_similarity

    query_vec = _tfidf_vectorizer.transform([query])
    scores = cosine_similarity(query_vec, _tfidf_matrix)[0]

    import numpy as np

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": corpus[idx]["metadata"],
            })
    return results


if __name__ == "__main__":
    query = "hợp đồng lao động"

    print(f"=== BM25 search: '{query}' ===")
    for r in lexical_search(query, top_k=5):
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")

    print(f"\n=== TF-IDF search: '{query}' ===")
    for r in tfidf_search(query, top_k=5):
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
