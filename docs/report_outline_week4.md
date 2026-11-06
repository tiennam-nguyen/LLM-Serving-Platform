# Đề Cương Báo Cáo Đồ Án & Snapshot Kiến Trúc Tuần 4 (Worker Registry, Health Monitor & Round-Robin Routing)

**Tên đề tài**: Nền tảng LLM Serving có khả năng quan sát và điều phối trên hạ tầng đa GPU (*Adaptive LLM Serving Platform for Observable Multi-GPU Inference*)  
**Nhóm thực hiện**: 02 sinh viên  
**Mốc hiện tại**: Hoàn thành Tuần 4 (Control-Plane `WorkerRegistry`, `HealthMonitor`, `RoundRobinRouter`, `/control/v1/workers` & Giao diện `/workers` Dual-Mode GPU / No-GPU)

---

## Phần I. Đề Cương Báo Cáo Tổng Thể & Tiến Độ Tích Lũy (Tuần 1 → Tuần 4)

### Chương 1. Giới thiệu & Đặt vấn đề (Introduction & Problem Statement)
- **1.1. Bối cảnh thực tế**: Nhu cầu vận hành các cụm suy luận mô hình ngôn ngữ lớn (LLM) tự chủ với độ trễ thấp (TTFT, E2E), khả năng chịu lỗi khi worker ngắt kết nối và khả năng quan sát chi tiết vòng đời từng request.
- **1.2. Các thách thức kỹ thuật**:
  - Đặc thù hai pha *Prefill* và *Decode* cùng cơ chế quản lý *KV Cache* / *Continuous Batching*.
  - Nhu cầu tách biệt **Data Plane** (đường truyền SSE tốc độ cao, không bị chặn bởi network probe hay database) và **Control Plane** (quản lý danh mục worker, kiểm tra sức khỏe định kỳ và cung cấp snapshot trạng thái).
  - Yêu cầu loại bỏ tức thời các worker suy giảm/mất kết nối (`UNHEALTHY`) khỏi tập ứng viên định tuyến mà không làm gián đoạn luồng suy luận của các worker còn lại.
- **1.3. Mục tiêu và phạm vi đề tài (Scope P0 / P1 / P2)**:
  - **P0 (Bắt buộc)**: Hệ thống chạy thực tế end-to-end (Web UI Playground & Workers Topology, FastAPI Gateway, Worker Registry, Health Monitor, $\ge 2$ workers, Round-Robin & Least-Loaded routing, request tracing/metrics, reproducible benchmark runner).
  - **P1 (Mở rộng khi P0 ổn định)**: Shared-prefix workload và Cache-Aware routing.
  - **P2 (Ngoài phạm vi học kỳ)**: Disaggregated Prefill/Decode, RDMA/NIXL, Kubernetes operator.

### Chương 2. Cơ sở lý thuyết & Công nghệ nền tảng (Background)
- **2.1. Kiến trúc suy luận LLM hiện đại**: Prefill vs. Decode, PagedAttention, Continuous Batching trong vLLM.
- **2.2. Các chỉ số hiệu năng cốt lõi**: TTFT, TPOT vs. Inter-Content-Chunk Latency, Routing Latency (`routing_ms`), E2E Latency & Throughput.
- **2.3. Các chiến lược điều phối (Routing Policies)**:
  - **Single-Worker (`SingleWorkerRouter`)**: Baseline Tuần 3.
  - **Round-Robin (`RoundRobinRouter` — Tuần 4)**: Phân phối tuần tự xoay vòng trên tập các worker đang ở trạng thái `HEALTHY` trong `WorkerRegistry`; tự động bỏ qua các worker `STARTING` hoặc `UNHEALTHY`, và trả về mã lỗi `503 Service Unavailable` an toàn khi tập worker khỏe mạnh rỗng.
  - **Least-Loaded (`LeastLoadedRouter` — Tuần 7)** và **Cache-Aware (Tuần 11)**.

### Chương 3. Thiết kế Kiến trúc Hệ thống Tuần 4 (SOLID & Clean Architecture)
- **3.1. Tuân thủ nguyên lý SOLID**:
  - **Single Responsibility Principle (SRP)**:
    - `WorkerRegistry` (`control/registry.py`) chỉ quản lý danh mục `WorkerRecord` bất biến (frozen dataclass) và danh sách tham chiếu runtime `healthy_workers()` có khóa `RLock` an toàn luồng.
    - `HealthMonitor` (`control/health.py`) chỉ đảm nhận việc probe bất đồng bộ (`check_once()`) với timeout độc lập cho từng worker (`asyncio.wait_for`), không trộn lẫn vào luồng xử lý request.
    - `RoundRobinRouter` (`routing/router.py`) chỉ đọc danh sách `registry.healthy_workers()` từ bộ nhớ RAM và xoay vòng con trỏ (`_cursor`), hoàn toàn không thực hiện truy vấn mạng.
  - **Open-Closed Principle (OCP)**:
    - Giao diện `RoutingPolicy` giữ nguyên hợp đồng `select_worker(trace: RequestTrace) -> InferenceWorker`; việc nâng cấp từ `SingleWorkerRouter` (Tuần 3) lên `RoundRobinRouter` (Tuần 4) không làm thay đổi cấu trúc hàm xử lý SSE (`safe_event_generator`).
    - Bảng `MILESTONE_GATES` trong `infra/scripts/verify.py` mở rộng thêm cổng `"w4"` (`gate_week4_registry_and_round_robin`) bên cạnh `"w1"`, `"w2"`, `"w3"` mà không xóa hay sửa đổi các bài kiểm thử của các tuần trước.
  - **Liskov Substitution Principle (LSP) & Dependency Inversion Principle (DIP)**:
    - Cả `VLLMWorkerAdapter` (GPU thật) và `MockInferenceWorker` (`--no-gpu`) đều thực thi đầy đủ hợp đồng `InferenceWorker`.
    - Hàm factory `create_worker` và `populate_registry` (`workers/factory.py`) khởi tạo tập worker dựa trên `GatewayConfig` (`WORKER_ENDPOINTS` cho cụm GPU hoặc `MOCK_WORKER_COUNT=2` cho chế độ `--no-gpu`).

### Chương 4. Hiện thực hóa Hệ thống (Cập nhật Chi tiết Tuần 4)
- **4.1. Danh mục Worker (`WorkerRegistry`) và Máy trạng thái Sức khỏe (`HealthMonitor`)**:
  - Mỗi worker khi đăng ký qua `registry.register(worker, endpoint, model)` bắt đầu ở trạng thái `STARTING` (`last_checked_at = None`).
  - Khi Gateway khởi động (`lifespan` trong `gateway/main.py`), `await health_monitor.check_once()` được thực thi ngay lập tức để kiểm tra toàn bộ worker song song và chuyển trạng thái sang `HEALTHY` hoặc `UNHEALTHY` kèm mốc thời gian chuẩn UTC ISO-8601 (`...Z`), đồng thời khởi chạy vòng lặp nền `_periodic_health_loop` định kỳ theo `HEALTH_CHECK_INTERVAL_S`.
- **4.2. Read-Only Control API (`/control/v1/workers`) & Giao diện `/workers`**:
  - Gateway mount `create_control_router(registry)` cung cấp `GET /control/v1/workers` (trả về mảng JSON sắp xếp theo `worker_id`) và `GET /control/v1/workers/{worker_id}` (trả về chi tiết hoặc 404), được bọc bởi `CORSMiddleware` ngoài cùng.
  - Trang `/workers` trên Next.js (`apps/web/src/components/workers/WorkersView.tsx`) hiển thị danh sách worker, model, endpoint, trạng thái (`STARTING`, `HEALTHY`, `UNHEALTHY`), thời điểm kiểm tra cuối cùng và nút `Refresh` thủ công.
- **4.3. Thuật toán `RoundRobinRouter` & Cô lập Worker lỗi (Fault Isolation)**:
  - Khi có nhiều worker `HEALTHY` (ví dụ `worker-mock-0` và `worker-mock-1`, hoặc `worker-vllm-0` và `worker-vllm-1`), các request liên tiếp được điều phối luân phiên (`0 → 1 → 0 → 1`) và gắn header `X-Worker-ID` tương ứng.
  - Khi một worker chuyển sang `UNHEALTHY`, `registry.healthy_workers()` lập tức loại worker đó khỏi danh sách trả về; toàn bộ các request mới chỉ được gửi tới các worker `HEALTHY` còn lại.
  - Khi toàn bộ worker đều `UNHEALTHY`, `RoundRobinRouter.select_worker(trace)` đánh dấu `trace.mark_routing_end(worker_id="none")` và ném `WorkerUnavailableError` (503). Gateway ghi nhận trace lỗi (`status="error"`) và trả về HTTP `503` kèm thông điệp rõ ràng, đảm bảo không gửi bất kỳ request nào tới worker đang lỗi.
- **4.4. Kiểm thử 3 Phương thức Hỗ trợ Kép (Có GPU & Không có GPU `--no-gpu`)**:
  1. **Kiểm thử E2E hoặc từng phần (`infra/scripts/verify.py`)**:
     - Chạy toàn bộ (mặc định có GPU hoặc `--no-gpu`): `python infra/scripts/verify.py [--no-gpu]`
     - Chạy từng phần: `python infra/scripts/verify.py --target {lint,unit,integration,w1,w2,w3,w4,gates,web} [--no-gpu]`
  2. **Kiểm thử tự động trên GitHub Actions (`.github/workflows/ci.yml`)**:
     - Tự động chạy `python infra/scripts/verify.py --no-gpu` trên mỗi `push` / `pull_request`, và hỗ trợ `workflow_dispatch` để chọn `target` bất kỳ (`all`, `w4`, `unit`, ...) cũng như bật/tắt `use_gpu` (`true`/`false`).
  3. **Kiểm thử với vai trò người dùng trên trình duyệt (`infra/scripts/dev_stack.py`)**:
     - Có GPU: `python infra/scripts/dev_stack.py` (hoặc truyền `--worker-endpoints` cho nhiều GPU).
     - Không có GPU: `python infra/scripts/dev_stack.py --no-gpu` (tự động mở Gateway `:8080` với 2 worker `worker-mock-0`, `worker-mock-1` và Web UI `:3000` để test chat xoay vòng Round-Robin tại `http://127.0.0.1:3000` và xem trạng thái worker tại `http://127.0.0.1:3000/workers`).

---

## Phần II. Kết Quả Kiểm Chứng Thực Tế Tuần 4 (Milestone Evidence)

| Hạng mục Kiểm thử | Lệnh Thực thi | Kết quả Thực tế (No-GPU Cloud & Mock Transport GPU) |
| :--- | :--- | :--- |
| **Static & Syntax Check (`lint`)** | `python infra/scripts/verify.py --target lint --no-gpu` | **PASSED** (42/42 file Python đạt chuẩn cú pháp và độ dài dòng $\le 100$ ký tự). |
| **Unit Test Suite (`unit`)** | `python infra/scripts/verify.py --target unit --no-gpu` | **PASSED** (38/38 bài kiểm thử: `WorkerRegistry`, `HealthMonitor`, `RoundRobinRouter`, `VLLMWorkerAdapter` multi-worker failover, CORS & traces). |
| **Integration Suite (`integration`)** | `python infra/scripts/verify.py --target integration --no-gpu` | **PASSED** (8/8 bài kiểm thử: Dual-mode Week 3 & Week 4 E2E, timeout/error propagation, stream cancellation). |
| **Cumulative Gates (`w1` → `w4`)** | `python infra/scripts/verify.py --target gates --no-gpu` | **PASSED** (`w1`, `w2`, `w3`, `w4` đều vượt qua kiểm định lũy tiến). |
| **Week 4 E2E Fault Isolation (`w4`)** | `python infra/scripts/verify_week4_e2e.py --no-gpu --save-evidence` | **PASSED** (Round-Robin `['worker-mock-0', 'worker-mock-1', 'worker-mock-0', 'worker-mock-1']` $\rightarrow$ Failover `['worker-mock-1', 'worker-mock-1', 'worker-mock-1']` $\rightarrow$ All-Down HTTP `503` $\rightarrow$ Recovery HTTP `200`). |
| **Live Port Smoke Test (`dev_stack.py`)** | `python infra/scripts/dev_stack.py --no-gpu --gateway-only --smoke-test` | **PASSED** (Uvicorn thật mở port HTTP, phục vụ `/health`, `/control/v1/workers`, `/v1/chat/completions` xoay vòng qua 2 worker và lưu `/v1/traces`). |
