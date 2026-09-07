r"""
FreeToken MoE Layer with Bandwidth-Adaptive Co-Execution ($q^\star$)
Reference: Section 3 & 4 of arXiv:2608.16157

Implements:
1. Gating & Top-K Expert Routing.
2. Cache hit evaluation against GPU-resident experts.
3. Execution partitioning via $q^\star$ policy for cache misses.
4. Concurrent CPU computation and GPU streaming & computation.
5. Exact output hidden state merging.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from freetoken.policy import QStarPolicy
from freetoken.cache import GlobalLRUExpertCache


class SingleExpert(nn.Module):
    """
    Standard Feed-Forward / SwiGLU Expert block.
    """
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)  # gate
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)  # down
        self.w3 = nn.Linear(hidden_dim, intermediate_dim, bias=False)  # up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class FreeTokenMoELayer(nn.Module):
    """
    Edge-Native Mixture-of-Experts Layer with Bandwidth-Adaptive Execution.
    """
    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        top_k: int,
        hidden_dim: int,
        intermediate_dim: int,
        expert_cache: GlobalLRUExpertCache,
        q_star_policy: QStarPolicy,
        device: torch.device,
        host_experts: Optional[List[SingleExpert]] = None
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.cache = expert_cache
        self.policy = q_star_policy
        self.device = device

        # Gating network (kept permanently on GPU for fast router decisions)
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False, device=device)

        # Host-resident full expert inventory (in CPU RAM or memory-mapped storage)
        if host_experts is not None:
            self.host_experts = host_experts
        else:
            self.host_experts = [
                SingleExpert(hidden_dim, intermediate_dim).to("cpu")
                for _ in range(num_experts)
            ]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MoE layer:
        Input: hidden_states of shape (batch_size, seq_len, hidden_dim)
        Output: merged hidden states of shape (batch_size, seq_len, hidden_dim)
        """
        orig_shape = hidden_states.shape
        x_flat = hidden_states.view(-1, self.hidden_dim)  # (N, D)
        n_tokens = x_flat.shape[0]

        # 1. Compute router logits & top-k routing weights on GPU
        router_logits = self.gate(x_flat)  # (N, num_experts)
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        # Normalize top-k weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # 2. Identify active experts in this forward step
        flat_indices = topk_indices.view(-1)
        unique_active_experts = torch.unique(flat_indices).tolist()

        # Track tokens routed to each active expert
        tokens_per_expert = []
        for exp_id in unique_active_experts:
            count = (flat_indices == exp_id).sum().item()
            tokens_per_expert.append(count)

        # 3. Categorize active experts into cache hits vs cache misses
        hits = []
        misses = []
        miss_token_counts = []
        for exp_id, count in zip(unique_active_experts, tokens_per_expert):
            if self.cache.contains(self.layer_idx, exp_id):
                hits.append(exp_id)
            else:
                misses.append(exp_id)
                miss_token_counts.append(count)

        # 4. Use q* Policy to partition cache misses between GPU streaming and CPU host execution
        partition = self.policy.compute_optimal_split(
            missing_experts=misses,
            tokens_per_expert=miss_token_counts,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim
        )

        # 5. Stream q* selected missing experts to GPU VRAM and put into cache
        for exp_id in partition.gpu_expert_indices:
            host_expert = self.host_experts[exp_id]
            # Copy expert module / parameters to GPU
            gpu_expert = SingleExpert(self.hidden_dim, self.intermediate_dim).to(self.device)
            gpu_expert.load_state_dict(host_expert.state_dict())
            self.cache.put(self.layer_idx, exp_id, gpu_expert)
            hits.append(exp_id)

        # Output accumulator on GPU
        out_flat = torch.zeros_like(x_flat)

        # 6. Execute GPU Path (resident experts + streamed experts)
        for exp_id in hits:
            expert_module = self.cache.get(self.layer_idx, exp_id)
            if expert_module is None:
                continue
            # Find tokens assigned to this expert
            mask = (topk_indices == exp_id)  # (N, K)
            token_mask = mask.any(dim=-1)     # (N,)
            if not token_mask.any():
                continue

            selected_tokens = x_flat[token_mask]
            expert_output = expert_module(selected_tokens)  # (M, D)

            # Weight by corresponding routing weight
            token_indices_where = torch.nonzero(token_mask).squeeze(-1)
            for local_idx, global_tok_idx in enumerate(token_indices_where):
                k_pos = torch.nonzero(mask[global_tok_idx]).squeeze(-1)[0]
                weight = topk_weights[global_tok_idx, k_pos]
                out_flat[global_tok_idx] += weight * expert_output[local_idx]

        # 7. Execute CPU Path (m - q* experts computed directly on host memory)
        if partition.cpu_expert_indices:
            x_cpu = x_flat.cpu()
            topk_indices_cpu = topk_indices.cpu()
            topk_weights_cpu = topk_weights.cpu()

            cpu_updates = torch.zeros_like(x_cpu)

            for exp_id in partition.cpu_expert_indices:
                host_expert = self.host_experts[exp_id]
                mask_cpu = (topk_indices_cpu == exp_id)
                token_mask_cpu = mask_cpu.any(dim=-1)
                if not token_mask_cpu.any():
                    continue

                selected_tokens_cpu = x_cpu[token_mask_cpu]
                with torch.no_grad():
                    expert_output_cpu = host_expert(selected_tokens_cpu)

                token_indices_where_cpu = torch.nonzero(token_mask_cpu).squeeze(-1)
                for local_idx, global_tok_idx in enumerate(token_indices_where_cpu):
                    k_pos = torch.nonzero(mask_cpu[global_tok_idx]).squeeze(-1)[0]
                    weight = topk_weights_cpu[global_tok_idx, k_pos]
                    cpu_updates[global_tok_idx] += weight * expert_output_cpu[local_idx]

            # Transfer merged CPU results back to GPU accumulator
            out_flat += cpu_updates.to(self.device)

        return out_flat.view(orig_shape)
