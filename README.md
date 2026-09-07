# FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution

Implementation of the research paper **arXiv:2608.16157** (*FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution* by FlashML-org).

## Overview

Mixture-of-Experts (MoE) architectures allow models to scale parameter counts significantly without proportional increases in compute. However, serving MoE models at the edge (on consumer GPUs such as NVIDIA RTX 30, 40, and 50 series) presents a severe memory bottleneck: limited VRAM prevents holding all experts simultaneously.

`freetoken` solves this problem through:
1. **$q^\star$ Bandwidth-Adaptive Execution:** Dynamically profiles PCIe Host-to-Device transfer rates ($B_p$) vs Host CPU computation throughput ($B_h$). On cache misses, it calculates an optimal partition $q^\star$ to overlap GPU PCIe streaming with concurrent CPU execution without pipeline bubbles.
2. **Global LRU Expert Cache:** Maintains a dynamically managed set of resident expert blocks on GPU VRAM, evicting the least-recently-used specialists.
3. **Elastic Memory Management:** Rebalances GPU VRAM dynamically between the KV cache (which grows with sequence length) and expert cache slots without restarting or reloading weights.
4. **Fast Token Weight (FTW) Format:** 64-byte aligned, zero-copy, page-aligned format designed for fast DMA transfers over PCIe.
5. **OpenAI / Anthropic Compatible HTTP Serving Engine:** Provides SSE streaming endpoints (`/v1/chat/completions`) for direct integration with agentic tools and chat interfaces.

---

## Installation

```bash
# Using uv (recommended)
uv pip install -e ".[test,accel]"

# Or using pip
pip install -e ".[test,accel]"
```

---

## Quickstart

### 1. Python API

```python
import torch
from freetoken import FreeTokenEngine, EngineConfig, MoEModelConfig

# Configure MoE model
model_config = MoEModelConfig(
    vocab_size=32000,
    hidden_dim=2048,
    intermediate_dim=5632,
    num_layers=12,
    num_experts=16,
    top_k=2
)

# Configure Edge Engine (e.g. 8GB VRAM target)
engine_config = EngineConfig(
    total_vram_gb=8.0,
    base_model_vram_gb=2.5,
    reserved_vram_gb=0.5,
    initial_expert_capacity=8
)

engine = FreeTokenEngine(model_config, engine_config)

# Autoregressive generation with streaming
prompt = [101, 2054, 2003, 1037, 3231, 102]
for token in engine.generate(prompt, max_new_tokens=20):
    print(f"Token: {token}")

# Inspect engine telemetry
print(engine.get_metrics())
```

### 2. Fast Token Weight (FTW) Format

```python
import torch
from freetoken.format import FTWFormat

tensors = {
    "expert_0_gate": torch.randn(1024, 2048, dtype=torch.float16),
    "expert_0_up": torch.randn(1024, 2048, dtype=torch.float16),
}

# Save into zero-copy, aligned binary format
FTWFormat.save("weights.ftw", tensors, metadata={"model": "deepseek-flash"})

# Load directly into memory
loaded = FTWFormat.load("weights.ftw", device="cpu")
```

### 3. Serving via CLI

```bash
# Profile PCIe bandwidth and CPU compute
freetoken benchmark

# Start OpenAI-compatible API server
freetoken serve --host 0.0.0.0 --port 8000 --vram 8.0
```

---

## Citation

```bibtex
@article{freetoken2026,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026}
}
```
# FreeToken
