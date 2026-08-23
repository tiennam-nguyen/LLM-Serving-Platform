# Security & Lab Safety Policy

This document outlines the operational security guidelines and safety rules for the **Adaptive LLM Serving Platform** project. All contributors must adhere to these policies.

---

## 1. Secrets and Credential Management

- **Zero-Commit Policy**: Never commit credentials, API keys, private tokens, certificates, or SSH keys to the Git repository.
- **Environment Isolation**: Secrets and configuration must be loaded from local environment variables or private `.env` files.
- **`.env` Ignored**: The local `.env` file is excluded in `.gitignore`. Only `.env.example` is committed, containing mock/placeholder names without real values.
- **Accidental Leak Protocol**: If a secret or key is inadvertently exposed or committed:
  1. Immediately revoke and rotate the compromised credential.
  2. Notify the team immediately.
  3. Purge the secret from Git history (do not simply delete it in a follow-up commit).

---

## 2. Privacy & Prompt Data Handling

- **Prompt Logging Policy**: User prompt content and model-generated responses must **not** be logged in plain text by default in server or gateway logs.
- **Metrics vs Payloads**: Telemetry and metrics systems should record timing, token counts, error codes, and resource utilization, rather than full user request payloads.

---

## 3. Network and API Surface

- **Internal Worker Endpoints**: Worker management, health probes, and internal routing endpoints are designed strictly for lab-internal communication and should not be exposed to the public internet without proper authentication/isolation.
- **No Remote Shell Interfaces**: Future web frontend or gateway APIs must never expose arbitrary shell execution or unchecked command injection vectors.

---

## 4. Model Weights and Large Artifacts

- **No Model Weights in Git**: LLM weights (e.g., `.safetensors`, `.bin`, `.pt`, `.gguf`) and cache directories must never be checked into Git. Always use local storage directories or shared artifact caches.

---

## 5. Reporting Security Concerns

To report a vulnerability or accidental secret exposure within the project, contact the project maintainers directly:
- **Maintainer**: Tien Nam Nguyen (`nguyentiennam011106@gmail.com`)
