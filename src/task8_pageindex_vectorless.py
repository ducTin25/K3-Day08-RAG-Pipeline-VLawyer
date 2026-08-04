"""Task 8 - PageIndex Vectorless RAG cho văn bản pháp luật lao động Việt Nam.

Luồng xử lý:
1. Upload các PDF luật gốc trong ``data/landing/legal`` lên PageIndex.
2. Lưu mapping tên file -> ``doc_id`` vào ``pageindex_doc_ids.json``.
3. Query từng tài liệu đã sẵn sàng và chuẩn hóa ``retrieved_nodes``.

SDK đang dùng: pageindex==0.1.9. Retrieval API là legacy nhưng vẫn được SDK và
PageIndex hỗ trợ để lấy các đoạn liên quan theo cấu trúc tài liệu.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).parent.parent
SOURCE_PDF_DIR = PROJECT_DIR / "data" / "landing" / "legal"
DOC_IDS_PATH = PROJECT_DIR / "pageindex_doc_ids.json"

DOCUMENT_READY_TIMEOUT = 300
RETRIEVAL_TIMEOUT = 120
POLL_INTERVAL = 3


def _get_api_key() -> str:
    """Đọc PageIndex API key mà không ghi key ra log."""
    load_dotenv()
    api_key = os.getenv("PAGEINDEX_API_KEY", "").strip()
    if not api_key or "PASTE" in api_key or api_key.endswith("..."):
        raise RuntimeError(
            "Chưa cấu hình PAGEINDEX_API_KEY trong .env. "
            "Tạo key tại https://dash.pageindex.ai/api-keys"
        )
    return api_key


@lru_cache(maxsize=1)
def _get_client() -> Any:
    """Khởi tạo PageIndex client một lần trong mỗi process."""
    try:
        from pageindex import PageIndexClient
    except ImportError as exc:
        raise RuntimeError(
            "Chưa cài pageindex. Hãy cài requirements.txt trước khi chạy Task 8."
        ) from exc
    return PageIndexClient(api_key=_get_api_key())


def _validate_search_inputs(query: str, top_k: int) -> str:
    if not isinstance(query, str):
        raise TypeError("query phải là chuỗi")
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query không được rỗng")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k phải là số nguyên")
    if top_k <= 0:
        raise ValueError("top_k phải lớn hơn 0")
    return cleaned_query


def _load_doc_ids() -> dict[str, str]:
    if not DOC_IDS_PATH.exists():
        return {}
    try:
        data = json.loads(DOC_IDS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Không đọc được {DOC_IDS_PATH.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{DOC_IDS_PATH.name} phải chứa một JSON object")
    return {
        str(name): str(doc_id)
        for name, doc_id in data.items()
        if name and doc_id
    }


def _save_doc_ids(doc_ids: dict[str, str]) -> None:
    DOC_IDS_PATH.write_text(
        json.dumps(doc_ids, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _remote_document_map(client: Any) -> dict[str, str]:
    """Tận dụng tài liệu đã upload trên tài khoản để tránh upload trùng."""
    try:
        response = client.list_documents(limit=100)
    except Exception as exc:
        warnings.warn(
            f"Không đọc được danh sách tài liệu PageIndex ({exc}); "
            "tiếp tục dùng mapping local.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}

    documents = response.get("documents", []) if isinstance(response, dict) else []
    result = {}
    for document in documents:
        if not isinstance(document, dict):
            continue
        name = document.get("name") or document.get("file_name")
        doc_id = document.get("id") or document.get("doc_id")
        if name and doc_id:
            result[str(name)] = str(doc_id)
    return result


def upload_documents() -> dict[str, str]:
    """Upload PDF luật chưa có trên PageIndex và trả mapping tên file -> doc_id."""
    pdf_files = sorted(SOURCE_PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"Không có PDF để upload trong {SOURCE_PDF_DIR}")

    client = _get_client()
    doc_ids = _load_doc_ids()
    remote_documents = _remote_document_map(client)

    for pdf_path in pdf_files:
        if pdf_path.name in doc_ids:
            print(f"  Reuse local: {pdf_path.name} -> {doc_ids[pdf_path.name]}")
            continue
        if pdf_path.name in remote_documents:
            doc_ids[pdf_path.name] = remote_documents[pdf_path.name]
            _save_doc_ids(doc_ids)
            print(f"  Reuse remote: {pdf_path.name} -> {doc_ids[pdf_path.name]}")
            continue

        print(f"  Uploading: {pdf_path.name}")
        response = client.submit_document(str(pdf_path))
        doc_id = response.get("doc_id") or response.get("id")
        if not doc_id:
            raise RuntimeError(
                f"PageIndex không trả doc_id khi upload {pdf_path.name}: {response}"
            )
        doc_ids[pdf_path.name] = str(doc_id)
        _save_doc_ids(doc_ids)
        print(f"  Uploaded: {pdf_path.name} -> {doc_id}")

    return doc_ids


def _wait_until_document_ready(client: Any, doc_id: str) -> None:
    deadline = time.monotonic() + DOCUMENT_READY_TIMEOUT
    while time.monotonic() < deadline:
        if client.is_retrieval_ready(doc_id):
            return
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(
        f"Tài liệu {doc_id} chưa sẵn sàng sau {DOCUMENT_READY_TIMEOUT} giây"
    )


def _wait_for_retrieval(client: Any, retrieval_id: str) -> dict:
    deadline = time.monotonic() + RETRIEVAL_TIMEOUT
    while time.monotonic() < deadline:
        response = client.get_retrieval(retrieval_id)
        status = str(response.get("status", "")).lower()
        if status in {"completed", "complete", "success", "succeeded"}:
            return response
        if status in {"failed", "error", "cancelled", "canceled"}:
            raise RuntimeError(
                f"PageIndex retrieval {retrieval_id} thất bại: {response}"
            )
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(
        f"Retrieval {retrieval_id} chưa hoàn tất sau {RETRIEVAL_TIMEOUT} giây"
    )


def _iter_relevant_items(value: Any) -> Iterator[dict]:
    """Hỗ trợ cả schema list[dict] mới và list[list[dict]] cũ."""
    if isinstance(value, dict):
        if value.get("relevant_content") or value.get("content"):
            yield value
        for nested_key in ("relevant_contents", "items", "results"):
            if nested_key in value:
                yield from _iter_relevant_items(value[nested_key])
    elif isinstance(value, list):
        for item in value:
            yield from _iter_relevant_items(item)


def _parse_retrieval(
    response: dict,
    *,
    source_name: str,
    doc_id: str,
) -> list[dict]:
    """Chuyển PageIndex retrieved_nodes sang schema retrieval chung."""
    results = []
    seen_contents = set()
    nodes = response.get("retrieved_nodes", [])

    for node in nodes if isinstance(nodes, list) else []:
        if not isinstance(node, dict):
            continue
        node_title = node.get("title") or node.get("section_title") or ""
        for item in _iter_relevant_items(node.get("relevant_contents", [])):
            content = str(
                item.get("relevant_content") or item.get("content") or ""
            ).strip()
            if not content or content in seen_contents:
                continue
            seen_contents.add(content)
            rank = len(results) + 1
            results.append(
                {
                    "content": content,
                    # PageIndex legacy retrieval không trả relevance score.
                    "score": round(1.0 / rank, 4),
                    "metadata": {
                        "source": source_name,
                        "doc_id": doc_id,
                        "node_id": node.get("node_id"),
                        "section": item.get("section_title") or node_title,
                        "page_index": item.get("page_index"),
                    },
                    "source": "pageindex",
                }
            )
    return results


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Truy vấn các PDF đã upload bằng PageIndex structural retrieval."""
    cleaned_query = _validate_search_inputs(query, top_k)
    doc_ids = _load_doc_ids()
    if not doc_ids:
        raise RuntimeError(
            f"Chưa có {DOC_IDS_PATH.name}. Hãy chạy upload_documents() trước."
        )

    client = _get_client()
    combined_results = []
    successful_queries = 0
    errors = []

    for source_name, doc_id in doc_ids.items():
        try:
            print(f"  Querying: {source_name}")
            _wait_until_document_ready(client, doc_id)
            submitted = client.submit_query(
                doc_id=doc_id,
                query=cleaned_query,
                thinking=False,
            )
            retrieval_id = submitted.get("retrieval_id") or submitted.get("id")
            if not retrieval_id:
                raise RuntimeError(f"PageIndex không trả retrieval_id: {submitted}")
            response = _wait_for_retrieval(client, str(retrieval_id))
            successful_queries += 1
            combined_results.extend(
                _parse_retrieval(
                    response,
                    source_name=source_name,
                    doc_id=doc_id,
                )
            )
        except Exception as exc:
            errors.append(f"{source_name}: {exc}")
            warnings.warn(
                f"Bỏ qua PageIndex document {source_name}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    if successful_queries == 0:
        raise RuntimeError("Không query được tài liệu PageIndex nào: " + "; ".join(errors))

    # Mỗi tài liệu có rank riêng; sort lại và loại nội dung trùng giữa tài liệu.
    combined_results.sort(key=lambda item: item["score"], reverse=True)
    unique_results = []
    seen = set()
    for item in combined_results:
        if item["content"] in seen:
            continue
        seen.add(item["content"])
        unique_results.append(item)
    return unique_results[:top_k]


if __name__ == "__main__":
    print("Uploading/reusing PageIndex documents...")
    upload_documents()

    print("\nTest query:")
    test_results = pageindex_search(
        "Mức lương tối thiểu được quy định như thế nào?",
        top_k=3,
    )
    for result in test_results:
        print(
            f"[{result['score']:.3f}] "
            f"{result['metadata'].get('source')}: {result['content'][:100]}..."
        )
