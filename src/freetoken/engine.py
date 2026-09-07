"""
FreeToken Serving Engine
Reference: arXiv:2608.16157

Coordinates:
- Model instantiation with FreeToken MoE layers
- Elastic Memory Coordinator for dynamic KV / Expert VRAM re-allocation
- Double-buffered prefill and autoregressive token generation
- Dynamic profiling of PCIe and CPU performance
"""

from __future__ import annotations
import dataclasses
import time
from typing import Dict, List, Optional, Generator, Tuple
import torch
import torch.nn as nn

from freetoken.policy import QStarPolicy, HardwareBandwidthProfile
from freetoken.cache import GlobalLRUExpertCache
from freetoken.coordinator import ElasticMemoryCoordinator, MemoryBudget
from freetoken.layer import FreeTokenMoELayer


@dataclasses.dataclass
class MoEModelConfig:
    vocab_size: int = 32000
    hidden_dim: int = 2048
    intermediate_dim: int = 5632
    num_layers: int = 12
    num_experts: int = 16
    top_k: int = 2
    max_seq_len: int = 4096


@dataclasses.dataclass
class EngineConfig:
    total_vram_gb: float = 8.0          # e.g. RTX 4060 (8GB), RTX 4070 (12GB), RTX 3090 (24GB)
    base_model_vram_gb: float = 2.5     # Attention, embeddings, norms permanently on GPU
    reserved_vram_gb: float = 0.5       # PyTorch workspace & CUDA context
    initial_expert_capacity: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MockDecoderLayer(nn.Module):
    """
    Decoder Layer pairing Self-Attention with a FreeToken MoE Layer.
    """
    def __init__(
        self,
        layer_idx: int,
        config: MoEModelConfig,
        cache: GlobalLRUExpertCache,
        policy: QStarPolicy,
        device: torch.device
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.device = device
        self.norm1 = nn.LayerNorm(config.hidden_dim, device=device)
        self.self_attn = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False, device=device)
        self.norm2 = nn.LayerNorm(config.hidden_dim, device=device)
        self.moe = FreeTokenMoELayer(
            layer_idx=layer_idx,
            num_experts=config.num_experts,
            top_k=config.top_k,
            hidden_dim=config.hidden_dim,
            intermediate_dim=config.intermediate_dim,
            expert_cache=cache,
            q_star_policy=policy,
            device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Self-Attention
        residual = x
        x_norm = self.norm1(x)
        x = residual + self.self_attn(x_norm)

        # Pre-LN MoE
        residual = x
        x_norm = self.norm2(x)
        moe_out = self.moe(x_norm)
        return residual + moe_out


class FreeTokenEngine:
    """
    FreeToken Edge-Native MoE Serving Engine.
    """
    def __init__(self, model_config: MoEModelConfig, engine_config: EngineConfig):
        self.model_cfg = model_config
        self.engine_cfg = engine_config
        self.device = torch.device(engine_config.device)

        # 1. Initialize Hardware Bandwidth Profiling and QStar Policy
        measured_pcie = HardwareBandwidthProfile.benchmark_pcie(self.device)
        self.profile = HardwareBandwidthProfile(pcie_bandwidth_gbps=measured_pcie)
        self.policy = QStarPolicy(self.profile)

        # 2. Global LRU Expert Cache
        self.cache = GlobalLRUExpertCache(
            capacity_experts=engine_config.initial_expert_capacity,
            device=self.device
        )

        # 3. Elastic Memory Coordinator
        bytes_per_expert = int(3 * model_config.hidden_dim * model_config.intermediate_dim * 2)  # FP16
        bytes_per_kv_token = int(model_config.num_layers * 2 * model_config.hidden_dim * 2)      # K+V per layer
        
        budget = MemoryBudget(
            total_vram_bytes=int(engine_config.total_vram_gb * (1024 ** 3)),
            base_model_vram_bytes=int(engine_config.base_model_vram_gb * (1024 ** 3)),
            reserved_vram_bytes=int(engine_config.reserved_vram_gb * (1024 ** 3)),
            bytes_per_expert=bytes_per_expert,
            bytes_per_kv_token=bytes_per_kv_token
        )
        self.coordinator = ElasticMemoryCoordinator(
            budget=budget,
            expert_cache=self.cache
        )

        # 4. Construct Model Components
        self.embed = nn.Embedding(model_config.vocab_size, model_config.hidden_dim, device=self.device)
        self.layers = nn.ModuleList([
            MockDecoderLayer(i, model_config, self.cache, self.policy, self.device)
            for i in range(model_config.num_layers)
        ])
        self.final_norm = nn.LayerNorm(model_config.hidden_dim, device=self.device)
        self.lm_head = nn.Linear(model_config.hidden_dim, model_config.vocab_size, bias=False, device=self.device)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a sequence of token IDs.
        """
        # Ensure memory balance is updated for the sequence length
        seq_len = input_ids.shape[-1]
        self.coordinator.rebalance(requested_kv_tokens=seq_len)

        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Generator[int, None, None]:
        """
        Autoregressive generation with streaming tokens.
        """
        curr_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        for _ in range(max_new_tokens):
            logits = self.forward(curr_ids)
            next_token_logits = logits[0, -1, :] / max(temperature, 1e-4)

            # Sampling or greedy
            if temperature > 0.0:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(next_token_logits, dim=-1).item()

            yield next_token
            curr_ids = torch.cat([curr_ids, torch.tensor([[next_token]], device=self.device)], dim=-1)

    def get_metrics(self) -> dict:
        return {
            "pcie_bandwidth_gbps": self.profile.pcie_bandwidth_gbps,
            "cache_stats": self.cache.stats(),
            "memory_status": self.coordinator.get_status()
        }
