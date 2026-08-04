"""Streamlit chatbot hỏi đáp pháp luật lao động Việt Nam.

Luồng xử lý:
    Streamlit UI -> Retrieval (Task 9) -> Generation (Task 10) -> Citation

Chạy:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.task10_generation import generate_with_citation


APP_TITLE = "Trợ lý Luật Lao động Việt Nam"
SUGGESTED_QUESTIONS = [
    "Mức lương tối thiểu được pháp luật quy định như thế nào?",
    "Người lao động được nghỉ hằng năm bao nhiêu ngày?",
    "Khi nào người lao động được đơn phương chấm dứt hợp đồng?",
    "Thời giờ làm việc bình thường tối đa là bao nhiêu?",
    "Người lao động làm thêm giờ được trả lương như thế nào?",
]


def _source_details(source: dict, index: int) -> tuple[str, str, str, float]:
    """Chuẩn hóa metadata từ hybrid retrieval và PageIndex."""
    metadata = source.get("metadata") or {}
    raw_source = (
        metadata.get("source")
        or metadata.get("source_path")
        or metadata.get("file_name")
        or f"Nguồn {index}"
    )
    source_name = Path(str(raw_source)).name
    doc_type = str(metadata.get("type") or metadata.get("document_type") or "legal")
    section = str(metadata.get("section") or metadata.get("header") or "")
    score = float(source.get("score") or 0.0)
    return source_name, doc_type, section, score


def render_sources(sources: list[dict]) -> None:
    """Hiển thị danh sách chunks đã được dùng để sinh câu trả lời."""
    if not sources:
        return

    with st.expander(f"📚 Tài liệu tham khảo ({len(sources)} đoạn)"):
        for index, source in enumerate(sources, 1):
            source_name, doc_type, section, score = _source_details(source, index)
            heading = f"**[{index}] {source_name}** · `{doc_type}` · score `{score:.4f}`"
            st.markdown(heading)
            if section:
                st.caption(f"Mục/điều: {section}")
            content = str(source.get("content", "")).strip()
            st.text(content[:500] + ("..." if len(content) > 500 else ""))
            if index < len(sources):
                st.divider()


def clear_conversation() -> None:
    st.session_state.messages = []
    st.session_state.pending_query = None


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

with st.sidebar:
    st.title("⚖️ Luật Lao động Việt Nam")
    st.caption(
        "Tra cứu quy định đang có trong kho dữ liệu của nhóm. "
        "Câu trả lời được tạo từ Hybrid RAG và kèm tài liệu tham khảo."
    )

    st.divider()
    st.subheader("💡 Câu hỏi gợi ý")
    for index, suggestion in enumerate(SUGGESTED_QUESTIONS):
        if st.button(
            suggestion,
            use_container_width=True,
            key=f"suggestion_{index}",
        ):
            st.session_state.pending_query = suggestion

    st.divider()
    st.subheader("⚙️ Thiết lập")
    top_k = st.slider(
        "Số đoạn luật dùng làm bằng chứng",
        min_value=3,
        max_value=10,
        value=5,
        help="Số lớn hơn cung cấp nhiều nguồn hơn nhưng làm context dài hơn.",
    )
    st.button(
        "🗑️ Xóa lịch sử hội thoại",
        use_container_width=True,
        on_click=clear_conversation,
    )

    st.divider()
    st.caption("Semantic + BM25 → RRF → PageIndex fallback → OpenAI + citation")

st.title("⚖️ Trợ lý Luật Lao động Việt Nam")
st.caption(
    "Hỏi về tiền lương, hợp đồng lao động, thời giờ làm việc, nghỉ phép, "
    "bảo hiểm xã hội và công đoàn."
)
st.info(
    "Thông tin do hệ thống cung cấp nhằm mục đích tra cứu, không thay thế "
    "tư vấn pháp lý chuyên nghiệp.",
    icon="ℹ️",
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))
            retrieval_source = message.get("retrieval_source")
            if retrieval_source:
                st.caption(f"Phương thức truy xuất: `{retrieval_source}`")

user_input = st.chat_input("Nhập câu hỏi của bạn về pháp luật lao động Việt Nam...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None
    conversation_history = st.session_state.messages[-6:]
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm văn bản liên quan và tổng hợp câu trả lời..."):
            try:
                response = generate_with_citation(
                    query,
                    top_k=top_k,
                    conversation_history=conversation_history,
                )
                answer = response["answer"]
                sources = response.get("sources", [])
                retrieval_source = response.get("retrieval_source", "none")
            except Exception as exc:
                error_text = str(exc)
                if "Insufficient credits" in error_text:
                    answer = (
                        "PageIndex fallback đang tạm thời không khả dụng do tài khoản "
                        "hết credit. Vui lòng bổ sung credit rồi thử lại."
                    )
                else:
                    answer = "Không thể xử lý câu hỏi lúc này. Vui lòng thử lại sau."
                sources = []
                retrieval_source = "error"
                st.error(answer)
                with st.expander("Chi tiết kỹ thuật"):
                    st.code(f"{type(exc).__name__}: {error_text}")

        st.markdown(answer)
        render_sources(sources)
        st.caption(f"Phương thức truy xuất: `{retrieval_source}`")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieval_source": retrieval_source,
        }
    )
