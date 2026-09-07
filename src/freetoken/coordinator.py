"""
Elastic Memory Coordinator
Reference: Section 4.2 of arXiv:2608.16157

Dynamically balances available GPU VRAM between:
1. KV cache memory budget (grows with prompt length and concurrent requests)
2. Expert cache capacity (more resident experts -> higher cache hit rate)

Enables runtime re-allocation without restarting the engine or reloading weights.
"""

from __future__ import annotations
import dataclasses
from typing import Optional
from freetoken.cache import GlobalLRUExpertCache


@dataclasses.dataclass
class MemoryBudget:
    total_vram_bytes: int
    base_model_vram_bytes: int
    reserved_vram_bytes: int
    bytes_per_expert: int
    bytes_per_kv_token: int

    @property
    def dynamic_vram_bytes(self) -> int:
        return max(0, self.total_vram_bytes - self.base_model_vram_bytes - self.reserved_vram_bytes)


class ElasticMemoryCoordinator:
    """
    Coordinates dynamic memory partitioning between KV cache and Expert cache.
    """
    def __init__(
        self,
        budget: MemoryBudget,
        expert_cache: GlobalLRUExpertCache,
        min_resident_experts: int = 2,
        min_kv_tokens: int = 1024,
    ):
        self.budget = budget
        self.expert_cache = expert_cache
        self.min_resident_experts = min_resident_experts
        self.min_kv_tokens = min_kv_tokens
        
        # Initial allocation: 50% dynamic VRAM to KV cache, 50% to Expert cache
        self.current_kv_token_capacity = self.min_kv_tokens
        self.rebalance(requested_kv_tokens=min_kv_tokens)

    def rebalance(self, requested_kv_tokens: int) -> int:
        """
        Adjusts the expert cache capacity to accommodate the required KV cache token budget.
        Returns the updated expert cache capacity in number of experts.
        """
        dynamic_vram = self.budget.dynamic_vram_bytes
        
        # Minimum memory needed for requested KV tokens
        required_kv_bytes = max(requested_kv_tokens, self.min_kv_tokens) * self.budget.bytes_per_kv_token
        
        if required_kv_bytes > dynamic_vram:
            # Clamp KV tokens to maximum dynamic VRAM minus minimum expert capacity
            min_expert_bytes = self.min_resident_experts * self.budget.bytes_per_expert
            available_for_kv = max(0, dynamic_vram - min_expert_bytes)
            actual_kv_tokens = available_for_kv // self.budget.bytes_per_kv_token
            required_kv_bytes = actual_kv_tokens * self.budget.bytes_per_kv_token
            self.current_kv_token_capacity = actual_kv_tokens
        else:
            self.current_kv_token_capacity = requested_kv_tokens

        # Remaining dynamic VRAM is allocated to expert cache
        remaining_for_experts = max(0, dynamic_vram - required_kv_bytes)
        new_expert_capacity = max(self.min_resident_experts, remaining_for_experts // self.budget.bytes_per_expert)

        self.expert_cache.set_capacity(new_expert_capacity)
        return new_expert_capacity

    def get_status(self) -> dict:
        return {
            "total_vram_gb": self.budget.total_vram_bytes / (1024 ** 3),
            "dynamic_vram_gb": self.budget.dynamic_vram_bytes / (1024 ** 3),
            "kv_token_capacity": self.current_kv_token_capacity,
            "expert_cache_capacity": self.expert_cache.capacity,
            "expert_cache_current_size": self.expert_cache.current_size,
            "expert_cache_hit_rate": self.expert_cache.hit_rate
        }
