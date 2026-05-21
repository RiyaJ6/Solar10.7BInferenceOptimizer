<div align="center">
<img width="1328" height="574" alt="image" src="https://github.com/user-attachments/assets/73f8ce80-9d1d-45cb-ac1a-d1684385c9b9" />

# SOLAR-10.7B // Inference Optimizer

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C.svg)](#)
[![Status: CI Deployed](https://img.shields.io/badge/Status-Benchmark_CI_Deployed-success.svg)](#)

## 📌 Architecture Objective
An experimental pipeline executing 4-bit Activation aware Weight Quantization (AWQ).

This repository directly addresses the **latency, throughput, and cost trade-off** inherent in large scale LLM deployment. By compressing weights to 4 bit integers (128 group size), the VRAM footprint is reduced by ~70%. This hardens the model for high throughput, memory constrained serving via `vLLM` or `SGLang` on enterprise GPU clusters maximizing space for KV Cache offloading and continuous batching.

## ⚙️ Core Components
1. **`quantize.py`**: Executes the AWQ compression engine to generate the 4 bit quantized model weights.
2. **`benchmark_ci.py`**: An automated Continuous Integration (CI) evaluation script. It enforces production Service Level Objectives (SLOs) by measuring Time To First Token (TTFT) and token generation throughput post quantization.

## 🚀 Quickstart

**Prerequisites:** Ubuntu 22.04 | CUDA 11.8+ | 24GB+ VRAM (for calibration)

```bash
# 1. Clone and Install
git clone [https://github.com/](https://github.com/)[RiyaJ6]/solar-10.7b-inference-optimizer.git
cd solar-10.7b-inference-optimizer
pip install -r requirements.txt

# 2. Execute Quantization Pipeline
python quantize.py

# 3. Validate Production KPIs (TTFT & Throughput)
python benchmark_ci.py

# 4. Evaluation Target
The CI pipeline ensures that the structural compression does not degrade TTFT beyond strict enterprise latency thresholds validating the model's readiness for high oncurrency production endpoints.
```

**You trained the intelligence. I engineer the inference.**
