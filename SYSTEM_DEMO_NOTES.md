# Hệ thống hỏi đáp luật lao động cho giới trẻ — Demo notes

## 1. Mục tiêu hệ thống

Hệ thống xây dựng một RAG pipeline hỗ trợ người trẻ tra cứu luật lao động Việt Nam bằng câu hỏi tự nhiên. Nguồn trả lời được lấy từ văn bản pháp luật và các bài viết hướng dẫn đã thu thập, thay vì chỉ dựa vào kiến thức có sẵn của LLM.

Ví dụ câu hỏi mục tiêu:

- Người lao động trẻ thử việc tối đa bao lâu?
- Làm thêm giờ được tính lương như thế nào?
- Khi nghỉ việc có cần báo trước không?
- Người lao động có quyền lợi bảo hiểm xã hội nào?
- Công đoàn bảo vệ người lao động ra sao?

## 2. Dữ liệu hiện có

### Dữ liệu thô — `data/landing/`

- 4 PDF pháp lý trong `data/landing/legal/`.
- 5 bài viết dạng JSON trong `data/landing/news/`.
- Các chủ đề hiện có gồm Luật Lao động, Luật Bảo hiểm xã hội, Luật Công đoàn và thông tư liên quan.

### Dữ liệu chuẩn hóa — `data/standardized/`

Hiện có 8 file Markdown:

- 3 văn bản pháp lý đã chuyển đổi thành Markdown.
- 5 bài viết tin tức/hướng dẫn đã chuyển đổi thành Markdown.
- Các file vẫn giữ cấu trúc thư mục `legal/` và `news/` để phục vụ metadata và filtering.

Một PDF Luật Lao động còn lại là tài liệu scan/không có text layer nên chưa có Markdown tương ứng. Tài liệu này cần OCR trước khi đưa vào index.

## 3. Pipeline đã chuẩn bị

Luồng xử lý của hệ thống:

```text
PDF/JSON thô
    ↓
Markdown chuẩn hóa
    ↓
Chunk theo cấu trúc pháp luật
    ↓
Embedding BGE-M3
    ↓
ChromaDB
    ↓
Semantic Search + BM25
    ↓
RRF Reranking
    ↓
LLM trả lời kèm nguồn
```

## 4. Chunking strategy trong Task 4

File triển khai: `src/task4_chunking_indexing.py`.

Cấu hình:

```python
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "markdown_header_recursive"
```

Chiến lược gồm hai tầng:

1. `MarkdownHeaderTextSplitter` tách theo heading để giữ ngữ cảnh Chương, Mục và tiêu đề.
2. `RecursiveCharacterTextSplitter` tiếp tục tách các phần dài, ưu tiên ranh giới `Điều`, `Khoản`, đoạn văn, câu và từ.

Lý do chọn:

- Văn bản pháp luật phụ thuộc mạnh vào cấu trúc Điều/Khoản.
- Tránh ghép nội dung của hai điều luật không liên quan vào cùng một chunk.
- Giới hạn 800 ký tự giúp retrieval tập trung hơn.
- Overlap 100 ký tự giảm nguy cơ mất ngữ cảnh tại ranh giới chunk.

## 5. Embedding model

Cấu hình:

```python
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
```

Lý do chọn BGE-M3:

- Hỗ trợ tiếng Việt và nhiều ngôn ngữ.
- Phù hợp cả câu hỏi ngắn và tài liệu dài.
- Chạy local, không cần trả phí API embedding.
- Sinh dense embedding 1024 chiều.
- Embedding được normalize trước khi lưu để tìm kiếm cosine ổn định.

## 6. Vector database và collection

Cấu hình:

```python
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "vietnam_labor_law_youth_qa"
```

Tên collection đã được đổi từ domain đại học sang đúng domain hỏi đáp luật lao động Việt Nam cho giới trẻ.

ChromaDB được chọn vì:

- Chạy local và lưu persistent trong `chroma_db/`.
- Không cần Docker hoặc dịch vụ cloud.
- Hỗ trợ cosine similarity và metadata filtering.

Metadata của mỗi chunk gồm:

- Tên và đường dẫn tài liệu nguồn.
- Loại tài liệu: `legal` hoặc `news`.
- Domain: `vietnam_labor_law`.
- Heading chứa chunk.
- Chỉ số chunk.

ID chunk được sinh ổn định bằng SHA-1 từ đường dẫn nguồn và chỉ số chunk.

## 7. Những phần Task 4 đã implement

- `load_documents()`: đọc toàn bộ Markdown và tạo metadata.
- `chunk_documents()`: chunk hai tầng theo cấu trúc pháp luật.
- `embed_chunks()`: tạo normalized embeddings bằng BGE-M3.
- `index_to_vectorstore()`: upsert chunks và embeddings vào ChromaDB.
- Kiểm tra dữ liệu rỗng và embedding bị thiếu trước khi index.

## 8. Trạng thái kiểm thử

Kết quả test toàn repository tại thời điểm ghi chú:

```text
16 passed, 19 skipped
```

Riêng Task 4:

```text
4 passed
```

Các test bị skip chủ yếu thuộc những task retrieval, reranking, fallback và generation chưa được hoàn thiện đầy đủ.

## 9. Nội dung nên show cho mentor

1. Mở `data/landing/legal/` và `data/landing/news/` để giới thiệu dữ liệu thô.
2. Mở một file trong `data/standardized/legal/` và một file trong `news/` để cho thấy dữ liệu Markdown sạch.
3. Mở cấu hình đầu file `src/task4_chunking_indexing.py` để giải thích chunking, embedding và collection.
4. Trình bày metadata của chunk và lý do cần giữ Điều/Khoản.
5. Chạy test Task 4:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_individual.py -k Task4 -q
```

6. Sau khi index thật, kiểm tra collection và số chunk trong ChromaDB.

## 10. Việc tiếp theo

- OCR PDF Luật Lao động còn thiếu text layer.
- Chạy `src.task4_chunking_indexing` để tải BGE-M3 và tạo ChromaDB.
- Hoàn thiện Semantic Search, BM25, RRF và fallback.
- Kết nối retrieval với LLM để trả lời kèm citation.
- Tạo golden dataset và đánh giá bằng RAGAS.

## 11. Câu trình bày ngắn

> Nhóm đang xây dựng RAG chatbot cho luật lao động Việt Nam hướng đến người trẻ. Hiện hệ thống đã thu thập dữ liệu pháp lý và tin tức, chuẩn hóa được 8 tài liệu Markdown, đồng thời hoàn thiện bước chunking và indexing. Văn bản được chunk theo heading, Điều và Khoản với kích thước 800, overlap 100. Nhóm chọn BGE-M3 vì hỗ trợ tiếng Việt và lưu vector 1024 chiều vào ChromaDB collection `vietnam_labor_law_youth_qa`. Task 4 hiện vượt qua toàn bộ 4 test; bước tiếp theo là tạo index thật và nối semantic search, BM25, reranking cùng LLM có citation.
