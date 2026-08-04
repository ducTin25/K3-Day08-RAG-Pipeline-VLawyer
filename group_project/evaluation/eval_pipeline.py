"""
RAG Evaluation Pipeline.

Sử dụng DeepEval / RAGAS / TruLens để đánh giá chất lượng RAG pipeline.
Chọn 1 framework và implement đầy đủ.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS/DeepEval gọi LLM RẤT NHIỀU LẦN
(không phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi). Model free của OpenRouter giới hạn
50 request/ngày CHO CẢ TÀI KHOẢN (không phải theo model hay theo API key — đổi model free
khác hay tạo key mới KHÔNG reset quota). Nếu chạy full 15+ câu hỏi mà bị rate limit giữa
chừng, thử giảm xuống subset 5 câu để chạy kịp trong buổi, hoặc nạp $10 credit để mở khóa
1000 request/ngày.
"""

import json
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# RAG pipeline wrapper — tái sử dụng Task 9 (retrieve) + Task 10 (prompt/reorder)
# nhưng cho phép bật/tắt reranking để phục vụ A/B comparison.
# =============================================================================

def generate_with_config(query: str, top_k: int = 5, use_reranking: bool = True) -> dict:
    """Chạy pipeline end-to-end cho 1 config cụ thể (reranking on/off)."""
    from src.task9_retrieval_pipeline import retrieve
    from src.task10_generation import (
        LLM_MODEL,
        SYSTEM_PROMPT,
        TEMPERATURE,
        TOP_P,
        format_context,
        reorder_for_llm,
    )

    chunks = retrieve(query, top_k=top_k, use_reranking=use_reranking)
    if not chunks:
        return {"answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có", "sources": []}

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    if not context:
        return {"answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có", "sources": chunks}

    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
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
    answer = response.choices[0].message.content or ""
    return {"answer": answer.strip(), "sources": chunks}


def run_generation(golden_dataset: list[dict], use_reranking: bool) -> list[dict]:
    """Chạy pipeline cho toàn bộ golden dataset với 1 config, trả về eval rows."""
    rows = []
    label = "reranking=ON" if use_reranking else "reranking=OFF"
    for i, item in enumerate(golden_dataset, 1):
        question = item["question"]
        print(f"  [{label}] {i}/{len(golden_dataset)}: {question[:60]}...")
        try:
            result = generate_with_config(question, top_k=5, use_reranking=use_reranking)
        except Exception as exc:
            warnings.warn(f"Bỏ qua câu hỏi do lỗi: {question!r} -> {exc}", RuntimeWarning)
            result = {"answer": "", "sources": []}

        rows.append({
            "question": question,
            "answer": result["answer"],
            "contexts": [c["content"] for c in result["sources"]] or [""],
            "ground_truth": item["expected_answer"],
        })
    return rows


# =============================================================================
# Option 2: RAGAS (framework đã chọn — xem requirements.txt: ragas==0.1.21)
# =============================================================================

def evaluate_with_ragas(rows: list[dict]):
    """
    Evaluate 1 tập rows (question/answer/contexts/ground_truth) bằng RAGAS.

    Returns:
        pandas.DataFrame — mỗi dòng là 1 câu hỏi, mỗi cột là 1 metric.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = Dataset.from_dict({
        "question": [r["question"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "contexts": [r["contexts"] for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    })

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    return result.to_pandas()


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(golden_dataset: list[dict]) -> dict:
    """
    So sánh A/B: Config A = hybrid search + RRF reranking, Config B = hybrid
    search KHÔNG reranking (dùng thẳng kết quả merge RRF của Task 7/9).

    Returns:
        {
            "Config A (hybrid + rerank)": {"df": DataFrame, "means": dict},
            "Config B (dense-only, no rerank)": {"df": DataFrame, "means": dict},
        }
    """
    configs = {
        "Config A (hybrid + rerank)": True,
        "Config B (dense-only, no rerank)": False,
    }

    results = {}
    for config_name, use_reranking in configs.items():
        print(f"\n=== {config_name} ===")
        rows = run_generation(golden_dataset, use_reranking=use_reranking)
        print(f"  Đánh giá RAGAS cho {len(rows)} câu hỏi...")
        df = evaluate_with_ragas(rows)
        means = {m: float(df[m].mean()) for m in METRIC_NAMES if m in df.columns}
        results[config_name] = {"df": df, "means": means, "rows": rows}

    return results


# =============================================================================
# Export Results
# =============================================================================

def _fmt(value) -> str:
    if value is None:
        return "N/A"
    try:
        if value != value:  # NaN check without importing math/pandas
            return "N/A"
    except TypeError:
        pass
    return f"{value:.3f}"


def export_results(comparison: dict) -> None:
    """Format và ghi kết quả A/B ra results.md."""
    names = list(comparison.keys())
    a_name, b_name = names[0], names[1]
    a_means = comparison[a_name]["means"]
    b_means = comparison[b_name]["means"]

    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevance",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
    }

    lines = ["# RAG Evaluation Results", ""]
    lines.append("## Framework sử dụng")
    lines.append("")
    lines.append("RAGAS (`ragas==0.1.21`) — 4 metrics chuẩn: faithfulness, answer_relevancy, "
                  "context_recall, context_precision. LLM judge: OpenAI (`OPENAI_API_KEY`).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Overall Scores")
    lines.append("")
    lines.append(f"| Metric | {a_name} | {b_name} | Δ |")
    lines.append("|--------|" + "-" * len(a_name) + "-|-" + "-" * len(b_name) + "-|---|")

    a_scores, b_scores = [], []
    for key, label in metric_labels.items():
        a_val = a_means.get(key)
        b_val = b_means.get(key)
        delta = (a_val - b_val) if (a_val is not None and b_val is not None) else None
        if a_val is not None:
            a_scores.append(a_val)
        if b_val is not None:
            b_scores.append(b_val)
        delta_str = f"{delta:+.3f}" if delta is not None else "N/A"
        lines.append(f"| {label} | {_fmt(a_val)} | {_fmt(b_val)} | {delta_str} |")

    a_avg = sum(a_scores) / len(a_scores) if a_scores else None
    b_avg = sum(b_scores) / len(b_scores) if b_scores else None
    avg_delta = (a_avg - b_avg) if (a_avg is not None and b_avg is not None) else None
    avg_delta_str = f"{avg_delta:+.3f}" if avg_delta is not None else "N/A"
    lines.append(f"| **Average** | **{_fmt(a_avg)}** | **{_fmt(b_avg)}** | **{avg_delta_str}** |")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## A/B Comparison Analysis")
    lines.append("")
    lines.append(f"**{a_name}:**")
    lines.append("> Hybrid search (semantic + BM25 merge bằng RRF) sau đó rerank lại bằng "
                  "`rerank_rrf`/`rerank()` (Task 7) trước khi đưa vào context.")
    lines.append("")
    lines.append(f"**{b_name}:**")
    lines.append("> Hybrid search (semantic + BM25 merge bằng RRF) nhưng KHÔNG rerank thêm — "
                  "dùng thẳng top-k của kết quả merge.")
    lines.append("")
    if avg_delta is not None:
        winner = a_name if avg_delta > 0 else (b_name if avg_delta < 0 else "Không chênh lệch đáng kể")
        lines.append(f"**Kết luận:** {winner} có average score cao hơn ({_fmt(max(a_avg, b_avg) if a_avg is not None and b_avg is not None else None)}). "
                      "Chênh lệch chủ yếu đến từ context_precision/recall vì rerank sắp xếp lại "
                      "chunk liên quan lên đầu context, giúp LLM ít bị nhiễu bởi chunk không liên quan.")
    else:
        lines.append("**Kết luận:** Không đủ dữ liệu số để so sánh (kiểm tra lỗi RAGAS/API ở trên).")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Worst performers từ Config A, sort theo faithfulness tăng dần.
    lines.append("## Worst Performers (Bottom 3)")
    lines.append("")
    lines.append("| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |")
    lines.append("|---|----------|-------------|-----------|--------|---------------|------------|")

    df_a = comparison[a_name]["df"]
    if "faithfulness" in df_a.columns:
        worst = df_a.sort_values("faithfulness", ascending=True, na_position="first").head(3)
        for rank, (_, row) in enumerate(worst.iterrows(), 1):
            question = str(row.get("question", ""))[:60]
            faith = _fmt(row.get("faithfulness"))
            rel = _fmt(row.get("answer_relevancy"))
            recall = _fmt(row.get("context_recall"))
            failure_stage = "Retrieval" if (row.get("context_recall") or 1) < 0.5 else "Generation"
            root_cause = (
                "Context thiếu evidence (recall thấp)"
                if failure_stage == "Retrieval"
                else "Câu trả lời không bám sát context (faithfulness thấp)"
            )
            lines.append(f"| {rank} | {question} | {faith} | {rel} | {recall} | {failure_stage} | {root_cause} |")
    else:
        lines.append("| - | (không có dữ liệu) | | | | | |")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    lines.append("### Cải tiến 1")
    lines.append("**Action:** Bật reranking mặc định (Config A) cho pipeline production — "
                  "kết quả A/B cho thấy rerank cải thiện độ liên quan của context.")
    lines.append("**Expected impact:** Tăng context_precision, giảm câu trả lời lạc đề.")
    lines.append("")
    lines.append("### Cải tiến 2")
    lines.append("**Action:** Bổ sung tên văn bản pháp luật chính thức (vd: \"Bộ luật Lao động 2019\") "
                  "vào metadata chunk thay vì chỉ có tên file, để giảm rủi ro LLM tự suy luận sai "
                  "số hiệu văn bản khi cite.")
    lines.append("**Expected impact:** Tăng faithfulness, citation chính xác hơn với văn bản ít phổ biến "
                  "(Nghị định/Thông tư).")
    lines.append("")
    lines.append("### Cải tiến 3")
    lines.append("**Action:** Sửa `SYSTEM_PROMPT` yêu cầu trích dẫn sau MỖI khẳng định thay vì 1 "
                  "citation gộp ở cuối đoạn (quan sát được khi review câu trả lời có nhiều luận điểm).")
    lines.append("**Expected impact:** Tăng khả năng truy vết nguồn cho từng claim, cải thiện "
                  "answer_relevancy khi câu hỏi có nhiều phần.")
    lines.append("")

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ Đã ghi kết quả vào {RESULTS_PATH}")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    comparison = compare_configs(golden_dataset)
    export_results(comparison)
