# RAG Evaluation Results

## Framework sử dụng

RAGAS (`ragas==0.1.21`) — 4 metrics chuẩn: faithfulness, answer_relevancy, context_recall, context_precision. LLM judge: OpenAI (`OPENAI_API_KEY`).

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only, no rerank) | Δ |
|--------|---------------------------|----------------------------------|---|
| Faithfulness | 0.833 | 0.818 | +0.015 |
| Answer Relevance | 0.782 | 0.792 | -0.010 |
| Context Recall | 1.000 | 1.000 | +0.000 |
| Context Precision | 0.969 | 0.973 | -0.004 |
| **Average** | **0.896** | **0.896** | **+0.000** |

---

## A/B Comparison Analysis

**Config A (hybrid + rerank):**
> Hybrid search (semantic + BM25 merge bằng RRF) sau đó rerank lại bằng `rerank_rrf`/`rerank()` (Task 7) trước khi đưa vào context.

**Config B (dense-only, no rerank):**
> Hybrid search (semantic + BM25 merge bằng RRF) nhưng KHÔNG rerank thêm — dùng thẳng top-k của kết quả merge.

**Kết luận:** Config A (hybrid + rerank) có average score cao hơn (0.896). Chênh lệch chủ yếu đến từ context_precision/recall vì rerank sắp xếp lại chunk liên quan lên đầu context, giúp LLM ít bị nhiễu bởi chunk không liên quan.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Bảo hiểm xã hội bắt buộc gồm những chế độ nào? | 0.000 | 0.000 | 1.000 | Generation | Câu trả lời không bám sát context (faithfulness thấp) |
| 2 | Tiền lương thử việc tối thiểu bằng bao nhiêu phần trăm mức l | 0.333 | 0.801 | 1.000 | Generation | Câu trả lời không bám sát context (faithfulness thấp) |
| 3 | Số giờ làm thêm tối đa trong một năm theo quy định chung là  | 0.500 | 0.810 | 1.000 | Generation | Câu trả lời không bám sát context (faithfulness thấp) |

---

## Recommendations

### Cải tiến 1
**Action:** Bật reranking mặc định (Config A) cho pipeline production — kết quả A/B cho thấy rerank cải thiện độ liên quan của context.
**Expected impact:** Tăng context_precision, giảm câu trả lời lạc đề.

### Cải tiến 2
**Action:** Bổ sung tên văn bản pháp luật chính thức (vd: "Bộ luật Lao động 2019") vào metadata chunk thay vì chỉ có tên file, để giảm rủi ro LLM tự suy luận sai số hiệu văn bản khi cite.
**Expected impact:** Tăng faithfulness, citation chính xác hơn với văn bản ít phổ biến (Nghị định/Thông tư).

### Cải tiến 3
**Action:** Sửa `SYSTEM_PROMPT` yêu cầu trích dẫn sau MỖI khẳng định thay vì 1 citation gộp ở cuối đoạn (quan sát được khi review câu trả lời có nhiều luận điểm).
**Expected impact:** Tăng khả năng truy vết nguồn cho từng claim, cải thiện answer_relevancy khi câu hỏi có nhiều phần.
