# Nhật Ký Debug & Hướng Dẫn Setup vLLM Chuẩn Mượt Mà

---

## 1. Tổng Quan Phần Cứng & Môi Trường
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **GPU:** NVIDIA GeForce RTX 2080 (8GB VRAM) – Kiến trúc Turing (Compute Capability 7.5)
- **NVIDIA Driver:** `560.35.03` (Hỗ trợ tối đa CUDA 12.6 - `found version 12060`)
- **Package Manager:** `uv`

---

## 2. Quá Trình Debug & Các Lỗi Đã Gặp

### ❌ Lỗi 1: `ImportError: cannot import name 'LLM' from 'vllm'`
- **Nguyên nhân:** Dự án khởi tạo mặc định với **Python 3.13**. Thư viện `vLLM` hiện chưa hỗ trợ các binary wheels chính thức trên Python 3.13 (chỉ hỗ trợ Python 3.9 -> 3.12).
- **Khắc phục:** 
  1. Giới hạn phiên bản Python trong `pyproject.toml`: `requires-python = ">=3.12,<3.13"`
  2. Ghim Python 3.12: `uv python pin 3.12`

---

### ❌ Lỗi 2: `RuntimeError: The NVIDIA driver on your system is too old (found version 12060)`
- **Nguyên nhân:** Khi chạy `uv add vllm`, `uv` lấy bản PyTorch mặc định từ PyPI được biên dịch với CUDA 12.8 / 13.0. Driver NVIDIA 560.35 trên máy chỉ tương thích tối đa với CUDA 12.6.
- **Khắc phục:** Cấu hình thêm kho chứa wheel PyTorch CUDA 12.4 (`cu124`) vào `pyproject.toml`:
  ```toml
  [[tool.uv.index]]
  name = "pytorch-cu124"
  url = "https://download.pytorch.org/whl/cu124"
  ```

---

### ❌ Lỗi 3: `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended`
- **Nguyên nhân:** Gói `transformers` bản mới nhất (`>=5.0` / `4.49+`) đã xóa bỏ thuộc tính `all_special_tokens_extended`, gây phá vỡ tương thích với `vLLM`.
- **Khắc phục:** Ràng buộc phiên bản `transformers<4.49.0` trong `pyproject.toml`:
  ```toml
  dependencies = [
      "vllm",
      "transformers<4.49.0",
  ]
  ```

---

### ❌ Lỗi 4: `ValueError: Bfloat16 is only supported on GPUs with compute capability of at least 8.0`
- **Nguyên nhân:** GPU RTX 2080 có Compute Capability 7.5 (Turing), **không hỗ trợ bfloat16 bằng phần cứng**. Mặc định mô hình Qwen 2.5 sẽ load dạng `bfloat16`.
- **Khắc phục:** Khởi tạo `LLM` với tham số chỉ định kiểu dữ liệu `dtype="float16"` (hoặc `"half"`):
  ```python
  llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="float16")
  ```

---

## 3. Cấu Hình Chuẩn Cho Dự Án (`pyproject.toml`)

Để cài đặt lại từ đầu trên một máy mới hoặc thư mục mới một cách mượt mà nhất, tệp `pyproject.toml` nên được cấu hình sẵn như sau:

```toml
[project]
name = "hai-nam"
version = "0.1.0"
description = "vLLM Inference Project"
readme = "README.md"
requires-python = ">=3.12,<3.13"
dependencies = [
    "vllm",
    "transformers<4.49.0",
]

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
```

---

## 4. Các Bước Setup Lại Nhanh & Mượt Nhất (Cheat Sheet)

```bash
# Bước 1: Ghim phiên bản Python 3.12
uv python pin 3.12

# Bước 2: Cài đặt và đồng bộ hóa thư viện
uv sync

# Bước 3: Chạy ứng dụng suy luận
uv run python main.py
```

---

## 5. Kinh Nghiệm Rút Ra
1. **Python Version:** Với các dự án Deep Learning / Large Language Model (vLLM, PyTorch), nên ưu tiên dùng **Python 3.12** thay vì bản Python mới nhất (3.13).
2. **Kích thước & Trọng số:** Trọng số mô hình được lưu tự động tại `~/.cache/huggingface/hub/`. Khi xóa virtual environment `.venv` hoặc làm sạch cache pip/uv, trọng số mô hình đã tải **không bị mất**, lần chạy sau không cần tải lại từ internet.
3. **GPU Turing (RTX 20xx / GTX 16xx):** Luôn ghi nhớ truyền `dtype="float16"` cho mô hình.
