"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"

Gợi ý LLM: OpenRouter có nhiều model gắn hậu tố ":free" không tính phí — xem
https://openrouter.ai/models?max_price=0 — phù hợp nếu chưa có credit trả phí.
Base URL: "https://openrouter.ai/api/v1", dùng chung interface với OpenAI SDK.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

# Giữ model chi phí thấp theo starter repo, nhưng gọi trực tiếp OpenAI thay vì
# dùng model ID dạng provider/model của OpenRouter. Có thể override trong .env.
LLM_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp về pháp luật lao động Việt Nam.

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. Mỗi khẳng định pháp lý phải có trích dẫn nguồn ngay sau, ví dụ: [Bộ luật Lao động 2019, Điều 91]
3. Nếu context không đủ thông tin → trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có"
4. Trả lời bằng tiếng Việt, có cấu trúc rõ ràng theo đoạn văn
5. Không suy luận hay mở rộng ngoài những gì được nêu trong context
6. Khi các nguồn mâu thuẫn, nêu rõ mâu thuẫn và không tự chọn kết luận"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if not isinstance(chunks, list):
        raise TypeError("chunks phải là list")

    # Luôn trả list mới để không làm thay đổi danh sách retrieval đầu vào.
    if len(chunks) <= 2:
        return list(chunks)

    front = chunks[::2]   # index 0, 2, 4 -> đầu và phần giữa
    back = chunks[1::2]   # index 1, 3    -> cuối theo thứ tự đảo
    return front + back[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    if not isinstance(chunks, list):
        raise TypeError("chunks phải là list")

    context_parts = []
    for index, chunk in enumerate(chunks, 1):
        if not isinstance(chunk, dict):
            raise TypeError(f"chunks[{index - 1}] phải là dict")

        content = str(chunk.get("content", "")).strip()
        if not content:
            continue

        metadata = chunk.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        source = (
            metadata.get("source")
            or metadata.get("source_path")
            or metadata.get("file_name")
            or f"Source {index}"
        )
        doc_type = metadata.get("type") or metadata.get("document_type") or "unknown"
        section = metadata.get("section") or metadata.get("header")
        page = metadata.get("page_index")

        labels = [
            f"Document {index}",
            f"Source: {source}",
            f"Type: {doc_type}",
        ]
        if section:
            labels.append(f"Section: {section}")
        if page is not None:
            labels.append(f"Page: {page}")

        context_parts.append(f"[{' | '.join(labels)}]\n{content}")

    return "\n\n---\n\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if not isinstance(query, str):
        raise TypeError("query phải là chuỗi")
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query không được rỗng")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k phải là số nguyên")
    if top_k <= 0:
        raise ValueError("top_k phải lớn hơn 0")

    # Step 1: Retrieve
    chunks = retrieve(cleaned_query, top_k=top_k)
    if not isinstance(chunks, list):
        raise TypeError("retrieve() phải trả về list")

    # Không gọi LLM khi retrieval không có evidence.
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có",
            "sources": [],
            "retrieval_source": "none",
        }

    # Step 2-3: Reorder và format context
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    if not context:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có",
            "sources": chunks,
            "retrieval_source": chunks[0].get("source", "hybrid"),
        }

    # Step 4: Build prompt theo format starter repo.
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {cleaned_query}"

    # Step 5: Gọi trực tiếp OpenAI API bằng OPENAI_API_KEY.
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or "PASTE" in api_key or api_key.endswith("..."):
        raise RuntimeError("Chưa cấu hình OPENAI_API_KEY trong .env")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise RuntimeError("OpenAI không trả về nội dung câu trả lời")

    # Step 6: Giữ đúng schema Task 10 của repo.
    return {
        "answer": answer.strip(),
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
    }


if __name__ == "__main__":
    test_queries = [
        "Mức lương tối thiểu được quy định như thế nào?",
        "Người lao động được nghỉ hằng năm bao nhiêu ngày?",
        "Điều kiện đơn phương chấm dứt hợp đồng lao động là gì?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
