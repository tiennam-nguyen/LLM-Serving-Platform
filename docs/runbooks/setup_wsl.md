# Nhật Ký Debug & Hướng Dẫn Setup vLLM Chuẩn Mượt Mà

---

## 1. Tổng Quan Phần Cứng & Môi Trường
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **GPU:** NVIDIA GeForce RTX 2080 (8GB VRAM) – Kiến trúc Turing (Compute Capability 7.5)
- **NVIDIA Driver:** `560.35.03` (Hỗ trợ tối đa CUDA 12.6 - `found version 12060`)
- **Package Manager:** `uv` / `pip` (Python 3.11 hoặc 3.12)

---

## 2. Quá Trình Debug & Các Lỗi Đã Gặp

### Lỗi 1: `ImportError: cannot import name 'LLM' from 'vllm'`
- **Nguyên nhân:** Dự án khởi tạo mặc định với **Python 3.13**. Thư viện `vLLM` hiện chưa hỗ trợ các binary wheels chính thức trên Python 3.13 (chỉ hỗ trợ Python 3.11 -> 3.12).
- **Khắc phục:**
  1. Giới hạn phiên bản Python trong `pyproject.toml`: `requires-python = ">=3.11,<3.13"`
  2. Ghim Python 3.12 khi dùng `uv`: `uv python pin 3.12`

---

### Lỗi 2: `RuntimeError: The NVIDIA driver on your system is too old (found version 12060)`
- **Nguyên nhân:** Khi chạy `uv add vllm`, `uv` lấy bản PyTorch mặc định từ PyPI được biên dịch với CUDA 12.8 / 13.0. Driver NVIDIA 560.35 trên máy chỉ tương thích tối đa với CUDA 12.6.
- **Khắc phục:** Cấu hình thêm kho chứa wheel PyTorch CUDA 12.4 (`cu124`) vào `pyproject.toml`:
  ```toml
  [[tool.uv.index]]
  name = "pytorch-cu124"
  url = "https://download.pytorch.org/whl/cu124"
  ```

---

### Lỗi 3: `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended`
- **Nguyên nhân:** Gói `transformers` bản mới nhất (`>=5.0` / `4.49+`) đã xóa bỏ thuộc tính `all_special_tokens_extended`, gây phá vỡ tương thích với `vLLM`.
- **Khắc phục:** Ràng buộc phiên bản `transformers<4.49.0` trong `pyproject.toml`.

---

### Lỗi 4: `ValueError: Bfloat16 is only supported on GPUs with compute capability of at least 8.0`
- **Nguyên nhân:** GPU RTX 2080 có Compute Capability 7.5 (Turing), **không hỗ trợ bfloat16 bằng phần cứng**. Mặc định mô hình Qwen 2.5 sẽ load dạng `bfloat16`.
- **Khắc phục:** Khởi chạy vLLM với tham số `--dtype float16` (đã tích hợp sẵn trong `infra/scripts/start_worker_wsl.sh`).

---

### Lỗi 5: `Connection refused` khi gọi `127.0.0.1:8000` hoặc `localhost:8000` trong WSL2
- **Nguyên nhân:** Cấu hình `.wslconfig` đang bật `networkingMode=mirrored`.
- **Khắc phục:**
  1. Truyền `--host 0.0.0.0` khi khởi chạy vLLM server.
  2. Nếu dùng WSL2 mirrored networking, cấu hình `VLLM_BASE_URL="http://127.0.1.1:8000"` trong `.env`.

---

## 3. Các Bước Khởi Chạy Nhanh (Cheat Sheet)

```bash
# Bước 1: Đồng bộ hóa môi trường Python 3.12 (trên máy có GPU)
uv python pin 3.12
uv sync

# Bước 2: Khởi chạy vLLM GPU Worker ở Terminal 1
bash infra/scripts/start_worker_wsl.sh

# Bước 3: Khởi chạy Gateway (:8080) + Web UI (:3000) ở Terminal 2
python infra/scripts/dev_stack.py
# (Hoặc nếu máy không có GPU: python infra/scripts/dev_stack.py --no-gpu)

# Bước 4: Chạy bộ kiểm thử hợp nhất
python infra/scripts/verify.py           # Có GPU (mặc định)
python infra/scripts/verify.py --no-gpu  # Không có GPU
```
