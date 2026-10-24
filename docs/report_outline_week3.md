# Đề Cương Báo Cáo Đồ Án & Snapshot Kiến Trúc Tuần 3 (Milestone Gate A)

**Tên đề tài**: Nền tảng LLM Serving có khả năng quan sát và điều phối trên hạ tầng đa GPU (*Adaptive LLM Serving Platform for Observable Multi-GPU Inference*)  
**Nhóm thực hiện**: 02 sinh viên  
**Mốc hiện tại**: Hoàn thành Tuần 3 (Walking Skeleton End-to-End: Browser → Gateway → Worker → GPU)

---

## Phần I. Đề Cương Báo Cáo Tổng Thể (Final Report Outline — 15 Tuần)

Đề cương này đóng vai trò khung xương cho báo cáo tổng kết đồ án (theo kế hoạch 15 tuần), được cập nhật dần qua từng milestone:

### Chương 1. Giới thiệu & Đặt vấn đề (Introduction & Problem Statement)
- **1.1. Bối cảnh thực tế**: Nhu cầu triển khai và vận hành các mô hình ngôn ngữ lớn (LLM) tự chủ trên hạ tầng GPU phòng lab/doanh nghiệp với độ trễ thấp (TTFT, E2E) và thông lượng cao.
- **1.2. Các thách thức kỹ thuật**:
  - Đặc thù hai pha của suy luận LLM: *Prefill* (compute-bound) và *Decode* (memory-bandwidth-bound) cùng cơ chế quản lý bộ nhớ *KV Cache* / *Continuous Batching*.
  - Sự thiếu hụt khả năng quan sát (observability) ở mức từng request (end-to-end request lifecycle tracing) khi đi qua nhiều lớp (Browser → Gateway → Router → Worker → GPU).
  - Bài toán điều phối tải (routing) giữa nhiều replica khi độ dài prompt/output không đồng nhất.
- **1.3. Mục tiêu và phạm vi đề tài (Scope P0 / P1 / P2)**:
  - **P0 (Bắt buộc)**: Hệ thống chạy thực tế end-to-end (Web UI, FastAPI Gateway, Worker Registry, Health Check, $\ge 2$ workers, Round-Robin & Least-Loaded routing, request tracing/metrics, reproducible benchmark runner).
  - **P1 (Mở rộng khi P0 ổn định)**: Shared-prefix workload và Cache-Aware routing.
  - **P2 (Ngoài phạm vi học kỳ)**: Disaggregated Prefill/Decode, RDMA/NIXL, Kubernetes operator.

### Chương 2. Cơ sở lý thuyết & Công nghệ nền tảng (Background)
- **2.1. Kiến trúc suy luận LLM hiện đại**: Prefill vs. Decode, PagedAttention, Continuous Batching trong vLLM.
- **2.2. Các chỉ số hiệu năng cốt lõi**:
  - **TTFT (Time To First Token / First Content)**: Độ trễ phản hồi đầu tiên.
  - **TPOT (Time Per Output Token)** vs. **Inter-Content-Chunk Latency**: Sự khác biệt giữa khoảng cách các frame mạng SSE và số lượng token thực tế từ bộ tokenizer.
  - **E2E Latency & Throughput**: Tổng thời gian hoàn tất request và số lượng request/token xử lý trên giây.
- **2.3. Các chiến lược điều phối (Routing Policies)**: Single-Worker (Baseline Tuần 3), Round-Robin (Tuần 4), Least-Loaded (Tuần 7), và Cache-Aware (Tuần 11).

### Chương 3. Phân tích Yêu cầu & Thiết kế Kiến trúc Hệ thống (System Architecture)
- **3.1. Nguyên lý thiết kế**: Modular Monolith, tách biệt Data Plane (fast path không chạm DB) và Control Plane, áp dụng nguyên lý SOLID (Dependency Inversion qua `InferenceWorker` và `RoutingPolicy`).
- **3.2. Kiến trúc tổng thể và các phân hệ**:
  - `apps/web`: Giao diện điều khiển và kiểm thử tương tác (Instrumentation Deck).
  - `gateway`: Cổng giao tiếp chuẩn OpenAI-compatible SSE, quản lý CORS, `X-Request-ID`, `X-Worker-ID`, xử lý lỗi và ngắt kết nối (abort propagation).
  - `routing`: Giao diện `RoutingPolicy` và các thuật toán chọn worker trong bộ nhớ.
  - `workers`: Lớp bọc `InferenceWorker` (`VLLMWorkerAdapter` cho GPU thật và `MockInferenceWorker` cho kiểm thử phi GPU `--no-gpu`).
  - `telemetry`: Bộ thu thập 6 mốc vòng đời request (`RequestTrace`) và kho lưu trữ giới hạn kích thước (`InMemoryTraceStore`).
  - `control` & `benchmark`: Quản lý danh mục worker, kiểm tra sức khỏe (Week 4) và bộ chạy thực nghiệm tái lập (Week 8–9).

### Chương 4. Hiện thực hóa Hệ thống (Implementation — Cập nhật đến Tuần 3)
- **4.1. Tuần 1 — Thiết lập Worker vLLM trên 1 GPU**: Khắc phục tương thích môi trường WSL2, CUDA 12.4 (`cu124`), `float16` trên kiến trúc Turing (RTX 2080 8GB VRAM) với mô hình `Qwen/Qwen2.5-0.5B-Instruct`.
- **4.2. Tuần 2 — Xây dựng Data-Plane Inference Gateway**:
  - Chuẩn hóa schema đầu vào (`ChatCompletionRequest`), middleware gán `X-Request-ID`, chuyển tiếp dòng sự kiện SSE (`safe_event_generator`).
  - Ánh xạ lỗi tầng mạng/worker (`WorkerTimeoutError` 504, `WorkerUnavailableError` 503) thành sự kiện lỗi SSE an toàn.
- **4.3. Tuần 3 — Luồng Web End-to-End & Quan sát Vòng đời Request**:
  - Xây dựng ứng dụng Next.js `apps/web` kết nối trực tiếp tới Gateway qua CORS.
  - Ghi nhận 6 mốc thời gian (`arrival`, `routing_start`, `routing_end`, `worker_start`, `first_token`, `last_token`) và đối chiếu cùng một `request_id` giữa trình duyệt (`RequestInspector`) và backend (`/v1/traces/{request_id}`).
  - Cơ chế kiểm thử kép (Dual-Mode Verification): Chạy thật với GPU (mặc định, hỗ trợ truyền câu lệnh `--prompt "prompt"`) và chạy giả lập không cần GPU (`--no-gpu`) trên CI.

### Chương 5. Phương pháp Thực nghiệm & Đánh giá (Methodology & Evaluation — Tuần 8–12)
- **5.1. Cấu hình phần cứng và môi trường thực nghiệm (Experiment Manifest)**.
- **5.2. Kịch bản tải (Workloads)**: Steady, Burst, Heterogeneous.
- **5.3. Kết quả kiểm thử chức năng & nghiệm thu từng giai đoạn (Gate A, Gate B, Gate C)**.
- **5.4. Kết quả so sánh hiệu năng giữa Round-Robin và Least-Loaded**.

### Chương 6. Hạn chế hiện tại & Hướng phát triển (Limitations & Future Work)
- **6.1. Các giới hạn đã biết (Known Limitations ở mốc Tuần 3)**.
- **6.2. Hướng mở rộng sau đồ án**.

---

## Phần II. Snapshot Kiến Trúc & Trạng Thái Kỹ Thuật Cuối Tuần 3

### 1. Sơ đồ luồng dữ liệu thực tế ở Tuần 3

```text
[Trình duyệt (Next.js Port 3000)]
       │
       │ (1) POST /v1/chat/completions (stream=True)
       │     Response Headers: X-Request-ID, X-Worker-ID
       ▼
[FastAPI Gateway (Port 8080)]
       ├──> [RequestTrace.mark_arrival()]
       ├──> [SingleWorkerRouter.select_worker()] -> mark_routing_start / mark_routing_end
       ├──> [safe_event_generator()]             -> mark_worker_start
       │           │
       │           ├── (Chế độ mặc định GPU) ──> [VLLMWorkerAdapter] ──> [vLLM Server (Port 8000, GPU)]
       │           └── (Chế độ --no-gpu CI)  ──> [MockInferenceWorker (In-Memory)]
       │
       ├──> Nhận chunk nội dung đầu tiên         -> mark_first_token (Tính Server TTFT)
       ├──> Kết thúc stream ([DONE]/Abort/Error) -> mark_last_token  (Tính Server E2E, Inter-Chunk)
       └──> Lưu vào [InMemoryTraceStore (FIFO 500)] + Ghi Structured JSON Log
       │
       │ (2) GET /v1/traces/{request_id} (Frontend tự động gọi ngay khi kết thúc stream)
       ▼
[RequestInspector trên Web UI hiển thị song song Client Metrics & Server Trace]
```

### 2. Các giới hạn đã biết tại Tuần 3 (Known Limitations)

1. **Single-Worker Routing (`SingleWorkerRouter`)**: Ở Tuần 3, hệ thống định tuyến tới 1 worker cấu hình sẵn (`worker-vllm-0`). Danh mục nhiều worker (`WorkerRegistry`), kiểm tra sức khỏe chủ động và thuật toán `RoundRobinRouter` sẽ được triển khai ở **Tuần 4**.
2. **Đo lường `inter_content_chunk_ms` thay vì `tpot_ms` mặc định**: Do một sự kiện SSE từ vLLM không phải lúc nào cũng tương ứng chính xác 1 token (có thể gộp token hoặc gửi frame rỗng khởi tạo role), backend công bố `inter_content_chunk_ms` và giữ `tpot_ms = None` trừ khi luồng SSE có trả về `usage.completion_tokens` tường minh.
3. **Lưu trữ Trace trong bộ nhớ (`InMemoryTraceStore`)**: Trace của 500 request gần nhất và lịch sử chat 50 lượt trên trình duyệt được giữ trong RAM theo đúng phạm vi P0 của Giai đoạn I; chưa sử dụng cơ sở dữ liệu ngoài để tránh làm chậm đường truyền suy luận.
