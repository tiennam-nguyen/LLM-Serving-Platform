#!/usr/bin/env bash
# Read-only inventory. A nonzero exit means the GPU gate failed.
set -euo pipefail
warnings=0
failures=0
warn() { printf 'WARN: %s\n' "$*"; warnings=$((warnings + 1)); }
report() {
    local label="$1"; shift
    printf '\n--- %s ---\n' "$label"
    if command -v "$1" >/dev/null 2>&1; then
        if ! "$@"; then warn "$label failed"; fi
    else
        warn "$1 unavailable"
    fi
}
report Host hostname
report 'Date/time' date -Ins
report OS cat /etc/os-release
report Kernel uname -a
PYTHON=""
if command -v python >/dev/null 2>&1; then PYTHON=python
elif command -v python3 >/dev/null 2>&1; then PYTHON=python3
fi
if [[ -n "$PYTHON" ]]; then report Python "$PYTHON" --version
else warn 'Python unavailable'; fi
report uv uv --version
report Git git --version
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    report Revision git rev-parse HEAD
    if tree="$(git status --porcelain)"; then
        if [[ -z "$tree" ]]; then echo 'Working tree: CLEAN'
        else echo 'Working tree: DIRTY'; printf '%s\n' "$tree"; warn 'uncommitted changes'; fi
    else warn 'cannot inspect Git working tree'; fi
fi
printf '\n--- NVIDIA ---\n'
if command -v nvidia-smi >/dev/null 2>&1; then
    if ! nvidia-smi; then
        echo 'FAIL: nvidia-smi failed'; failures=$((failures + 1))
    elif gpu_info="$(nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader)" && [[ -n "$gpu_info" ]]; then
        printf 'GPU name, UUID, VRAM, driver:\n%s\n' "$gpu_info"
    else
        echo 'FAIL: no NVIDIA GPU inventory'; failures=$((failures + 1))
    fi
else
    echo 'FAIL: nvidia-smi unavailable; NVIDIA GPU not verified'; failures=$((failures + 1))
fi
report 'IPv4 interfaces' ip -br -4 addr
report Routes ip route
printf '\n--- Listening lab ports (8000/8080/3000) ---\n'
if command -v ss >/dev/null 2>&1; then
    if ports="$(ss -ltnp)"; then
        # grep exit 1 means no matching listeners, which is normal before startup.
        printf '%s\n' "$ports" | grep -E 'State|:(8000|8080|3000)\b' || true
    else warn 'ss failed'; fi
else warn 'ss unavailable'; fi
if [[ -n "$PYTHON" ]]; then
    if ! "$PYTHON" - <<'PY'
import importlib.metadata
import sys
for package in ("vllm", "transformers"):
    try:
        print(f"{package}: {importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"WARN: {package} not installed in this Python")
try:
    import torch
except ImportError:
    print("WARN: PyTorch cannot be imported; run again after GPU installation")
    sys.exit(0)
print(f"torch: {torch.__version__}; torch CUDA: {torch.version.cuda}")
available = torch.cuda.is_available()
print(f"torch.cuda.is_available(): {available}")
if not available:
    sys.exit("FAIL: installed PyTorch cannot access CUDA")
for index in range(torch.cuda.device_count()):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}; "
          f"compute capability: {torch.cuda.get_device_capability(index)}")
PY
    then failures=$((failures + 1)); fi
fi
printf '\n%s | WARN tools/tree: %s | FAIL GPU/runtime: %s\n' \
    "$([[ "$failures" == 0 ]] && echo PASS || echo FAIL)" "$warnings" "$failures"
echo 'Python package warnings above may be expected before installation. This inventory does not prove LAN connectivity.'
(( failures == 0 ))
