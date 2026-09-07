"""
Global LRU Expert Cache for GPU VRAM Management
Reference: Section 3 of arXiv:2608.16157

Manages active GPU resident experts across layers.
- Supports dynamic capacity updates (e.g. during elastic memory rebalancing with KV cache).
- Implements LRU eviction when new experts are transferred.
- Tracks hit and miss statistics.
"""

from __future__ import annotations
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Any
import torch


class GlobalLRUExpertCache:
    """
    LRU cache for MoE experts resident on GPU VRAM.
    Keys are (layer_idx, expert_idx).
    Values are expert weight dictionaries or torch.nn.Module instances placed on device.
    """
    def __init__(self, capacity_experts: int, device: torch.device):
        self.capacity = max(1, capacity_experts)
        self.device = device
        self._cache: OrderedDict[Tuple[int, int], Any] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def current_size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def set_capacity(self, new_capacity: int, evict_callback: Optional[callable] = None):
        """
        Dynamically adjusts cache capacity (called during elastic memory reallocation).
        """
        self.capacity = max(1, new_capacity)
        while len(self._cache) > self.capacity:
            evicted_key, evicted_val = self._cache.popitem(last=False)
            if evict_callback:
                evict_callback(evicted_key, evicted_val)

    def contains(self, layer_idx: int, expert_idx: int) -> bool:
        return (layer_idx, expert_idx) in self._cache

    def get(self, layer_idx: int, expert_idx: int) -> Optional[Any]:
        key = (layer_idx, expert_idx)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, layer_idx: int, expert_idx: int, expert_weights: Any) -> Optional[Tuple[Tuple[int, int], Any]]:
        """
        Inserts an expert into the cache. If capacity is exceeded, evicts the LRU item
        and returns (evicted_key, evicted_value).
        """
        key = (layer_idx, expert_idx)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = expert_weights
            return None

        evicted = None
        if len(self._cache) >= self.capacity:
            evicted_key, evicted_val = self._cache.popitem(last=False)
            evicted = (evicted_key, evicted_val)

        self._cache[key] = expert_weights
        return evicted

    def get_resident_experts(self, layer_idx: int) -> List[int]:
        """
        Returns list of expert indices resident on GPU for a given layer.
        """
        return [exp_id for (l_id, exp_id) in self._cache.keys() if l_id == layer_idx]

    def clear(self):
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "current_size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate
        }
