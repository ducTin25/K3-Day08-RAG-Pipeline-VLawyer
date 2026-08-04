"""Streamlit demo for the Vietnamese labor-law RAG pipeline.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Hỏi đáp Luật Lao động Việt Nam",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def run_rag_query(query: str, top_k: int) -> tuple[str, list[dict]]:
    """Run Task 10 and normalize its response for the chat UI."""
    from src.task10_generation import generate_with_citation

    response = generate_with_citation(query, top_k=top_k)
    if not isinstance(response, dict):
        raise TypeError("generate_with_citation() phải trả về dict")

    answer = str(response.get("answer") or "").strip()
    if not answer:
        answer = "Hệ thống chưa tạo được câu trả lời."

    sources = response.get("sources") or []
    if not isinstance(sources, list):
        sources = []
    return answer, sources


def render_sources(sources: list[dict]) -> None:
    """Render retrieved evidence consistently for current and stored messages."""
    if not sources:
        return

    with st.expander(f"📚 Nguồn tham khảo ({len(sources)} đoạn)"):
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata") or {}
            source_name = (
                metadata.get("source")
                or metadata.get("source_path")
                or metadata.get("file_name")
                or "Không rõ nguồn"
            )
            section = metadata.get("section") or metadata.get("header")
            score = float(source.get("score") or 0)
            heading = f"**[{index}] {source_name}**"
            if section:
                heading += f" — {section}"
            st.markdown(f"{heading} | score: `{score:.4f}`")
            content = str(source.get("content") or "")
            st.text(content[:500] + ("..." if len(content) > 500 else ""))
            st.divider()


with st.sidebar:
    st.title("⚖️ Trợ lý Luật Lao động")
    st.caption(
        "Hỏi đáp quy định lao động Việt Nam dành cho người trẻ và người mới đi làm."
    )
    st.info(
        "Câu trả lời dùng để tham khảo, không thay thế tư vấn pháp lý từ người có chuyên môn."
    )

    st.divider()
    st.subheader("💡 Câu hỏi gợi ý")
    suggestions = [
        "Thời gian thử việc tối đa là bao lâu?",
        "Người lao động được nghỉ phép năm bao nhiêu ngày?",
        "Khi nào người lao động được đơn phương chấm dứt hợp đồng?",
        "Làm thêm giờ được trả lương như thế nào?",
        "Người sử dụng lao động có phải đóng bảo hiểm xã hội không?",
    ]
    for suggestion in suggestions:
        if st.button(
            suggestion,
            use_container_width=True,
            key=f"suggestion_{suggestion}",
        ):
            st.session_state["pending_query"] = suggestion

    st.divider()
    st.subheader("⚙️ Thiết lập")
    top_k = st.slider("Số đoạn tài liệu truy xuất", 3, 10, 5)
    st.caption(
        "Semantic + BM25 → RRF → PageIndex fallback → OpenAI trả lời kèm nguồn"
    )


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

st.title("⚖️ Hỏi đáp Luật Lao động Việt Nam")
st.caption(
    "Tra cứu nhanh quy định về hợp đồng, thử việc, tiền lương, thời giờ làm việc, "
    "nghỉ phép và bảo hiểm."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources") or [])

user_input = st.chat_input("Nhập câu hỏi của bạn về pháp luật lao động...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu quy định và tổng hợp câu trả lời..."):
            try:
                answer, sources = run_rag_query(query, top_k=top_k)
            except Exception as error:
                answer = f"❌ Không thể chạy RAG pipeline: {error}"
                sources = []

            st.markdown(answer)
            render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
