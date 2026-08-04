<<<<<<< HEAD
"""Streamlit demo for the Vietnamese labor-law RAG pipeline.
=======
"""Streamlit chatbot hỏi đáp pháp luật lao động Việt Nam.

Luồng xử lý:
    Streamlit UI -> Retrieval (Task 9) -> Generation (Task 10) -> Citation
>>>>>>> 74a5f52fcaa6912ac6b87b4fa08934f39c6e1d88

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

<<<<<<< HEAD
st.set_page_config(
    page_title="Hỏi đáp Luật Lao động Việt Nam",
=======
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
>>>>>>> 74a5f52fcaa6912ac6b87b4fa08934f39c6e1d88
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

<<<<<<< HEAD

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


=======
>>>>>>> 74a5f52fcaa6912ac6b87b4fa08934f39c6e1d88
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

<<<<<<< HEAD
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
=======
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
>>>>>>> 74a5f52fcaa6912ac6b87b4fa08934f39c6e1d88
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None
<<<<<<< HEAD
=======
    conversation_history = st.session_state.messages[-6:]
>>>>>>> 74a5f52fcaa6912ac6b87b4fa08934f39c6e1d88
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
<<<<<<< HEAD
        with st.spinner("Đang tra cứu quy định và tổng hợp câu trả lời..."):
            try:
                answer, sources = run_rag_query(query, top_k=top_k)
            except Exception as error:
                answer = f"❌ Không thể chạy RAG pipeline: {error}"
=======
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
>>>>>>> 74a5f52fcaa6912ac6b87b4fa08934f39c6e1d88
                sources = []
                retrieval_source = "error"
                st.error(answer)
                with st.expander("Chi tiết kỹ thuật"):
                    st.code(f"{type(exc).__name__}: {error_text}")

<<<<<<< HEAD
            st.markdown(answer)
            render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
=======
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
>>>>>>> 74a5f52fcaa6912ac6b87b4fa08934f39c6e1d88
    )
