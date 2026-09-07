r"""
Bandwidth-Adaptive Execution Policy ($q^\star$)
Reference: Section 3 & 4 of arXiv:2608.16157

At each token generation / layer step:
Given m missing experts (cache misses in GPU VRAM), the engine must decide:
- How many experts to stream across the PCIe bus to GPU (q_gpu)
- How many experts to execute directly on the host CPU (q_cpu)

Let:
- B_p: measured host-to-device PCIe bandwidth (GB/s)
- B_h: host CPU effective computation throughput (GB/s equivalent or FLOPS/expert)
- S_e: expert parameter size (Bytes)
- C_e: compute required per expert per active token (FLOPs)
- T_gpu_exec(q): execution time on GPU
- T_cpu_exec(m - q): execution time on CPU
- T_pcie_transfer(q): transfer time of q experts from Host to Device

The optimal partition q* balances total latency:
min_{q in [0, m]} max( T_pcie(q) + T_gpu(q), T_cpu(m - q) )
"""

from __future__ import annotations
import dataclasses
import time
from typing import List, Tuple, Sequence, Optional
import torch


@dataclasses.dataclass
class HardwareBandwidthProfile:
    """
    Profiles the host-device PCIe link and Host CPU compute throughput.
    """
    pcie_bandwidth_gbps: float = 16.0    # PCIe Gen4 x16 ~ 25-28 GB/s real; Gen3/Gen4 x8 ~ 12-16 GB/s
    cpu_bandwidth_gbps: float = 45.0     # DDR4/DDR5 system memory bandwidth
    cpu_tflops: float = 1.5              # Host CPU FP16/BF16/FP8 matrix throughput (TFLOPS)
    gpu_tflops: float = 60.0             # Consumer GPU (RTX 3090/4080/4090/5080) throughput (TFLOPS)
    expert_size_bytes: int = 128 * 1024 * 1024  # Example 128MB per expert (e.g. 64M params at FP16 or 128M at FP8)

    @classmethod
    def benchmark_pcie(cls, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        size_mb: int = 64) -> float:
        """
        Runs a quick micro-benchmark to measure actual Host-to-Device transfer rate.
        """
        if device.type != "cuda":
            return 16.0  # Fallback synthetic default
        
        num_bytes = size_mb * 1024 * 1024
        # Allocate pinned host memory for realistic PCIe transfer speed
        host_tensor = torch.empty(num_bytes // 4, dtype=torch.float32, pin_memory=True)
        device_tensor = torch.empty(num_bytes // 4, dtype=torch.float32, device=device)

        # Warmup
        for _ in range(2):
            device_tensor.copy_(host_tensor, non_blocking=False)
        torch.cuda.synchronize()

        start = time.perf_counter()
        iters = 5
        for _ in range(iters):
            device_tensor.copy_(host_tensor, non_blocking=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        gb_transferred = (num_bytes * iters) / (1024 ** 3)
        measured_gbps = gb_transferred / elapsed
        return max(0.5, measured_gbps)


@dataclasses.dataclass
class PartitionDecision:
    """
    Result of the q* policy optimization for a given set of missing experts.
    """
    gpu_expert_indices: List[int]
    cpu_expert_indices: List[int]
    optimal_q: int
    predicted_gpu_time_ms: float
    predicted_cpu_time_ms: float
    predicted_total_time_ms: float


class QStarPolicy:
    """
    Implements the FreeToken Bandwidth-Adaptive q* co-execution policy.
    Determines at runtime how many missing experts to stream to GPU vs compute on CPU.
    """
    def __init__(self, profile: Optional[HardwareBandwidthProfile] = None):
        self.profile = profile or HardwareBandwidthProfile()

    def update_profile(self, pcie_bandwidth_gbps: Optional[float] = None, cpu_tflops: Optional[float] = None):
        if pcie_bandwidth_gbps is not None:
            self.profile.pcie_bandwidth_gbps = pcie_bandwidth_gbps
        if cpu_tflops is not None:
            self.profile.cpu_tflops = cpu_tflops

    def compute_optimal_split(
        self,
        missing_experts: Sequence[int],
        tokens_per_expert: Sequence[int],
        hidden_dim: int,
        intermediate_dim: int,
    ) -> PartitionDecision:
        """
        Solves:
        q* = argmin_{q in [0, m]} max( T_gpu_transfer_and_compute(q), T_cpu_compute(m - q) )

        Parameters:
        - missing_experts: list of expert IDs not currently present in GPU VRAM.
        - tokens_per_expert: number of tokens routed to each missing expert.
        - hidden_dim: hidden representation dimension d.
        - intermediate_dim: expert FFN hidden dimension (typically ~ 2-4x d).
        """
        m = len(missing_experts)
        if m == 0:
            return PartitionDecision(
                gpu_expert_indices=[],
                cpu_expert_indices=[],
                optimal_q=0,
                predicted_gpu_time_ms=0.0,
                predicted_cpu_time_ms=0.0,
                predicted_total_time_ms=0.0
            )

        # Cost per expert for compute:
        # FFN FLOPs ~ 2 * (2 * hidden_dim * intermediate_dim) + 2 * (intermediate_dim * hidden_dim)
        # Typically ~ 6 * hidden_dim * intermediate_dim FLOPs per token
        flops_per_token = 6.0 * hidden_dim * intermediate_dim

        best_q = 0
        min_max_latency = float("inf")
        best_gpu_time = 0.0
        best_cpu_time = 0.0

        # Sort missing experts by tokens routed descending (prioritize transferring heavy compute to GPU)
        indexed_missing = sorted(
            range(m),
            key=lambda i: tokens_per_expert[i] if i < len(tokens_per_expert) else 0,
            reverse=True
        )

        # Evaluate all possible partition points q in [0, m]
        for q in range(m + 1):
            gpu_indices = indexed_missing[:q]
            cpu_indices = indexed_missing[q:]

            # 1. GPU Path:
            # Transfer q experts over PCIe:
            # Size in GB = q * expert_size_bytes / 1e9
            transfer_time_sec = (q * self.profile.expert_size_bytes) / (self.profile.pcie_bandwidth_gbps * 1e9)
            
            # GPU Compute:
            gpu_tokens = sum(tokens_per_expert[i] for i in gpu_indices) if gpu_indices else 0
            gpu_compute_flops = gpu_tokens * flops_per_token
            gpu_compute_time_sec = gpu_compute_flops / (self.profile.gpu_tflops * 1e12)
            
            # Total GPU time (transfer + compute)
            total_gpu_time = transfer_time_sec + gpu_compute_time_sec

            # 2. CPU Path:
            # CPU Compute (no PCIe transfer needed, already in RAM):
            cpu_tokens = sum(tokens_per_expert[i] for i in cpu_indices) if cpu_indices else 0
            cpu_compute_flops = cpu_tokens * flops_per_token
            cpu_compute_time_sec = cpu_compute_flops / (self.profile.cpu_tflops * 1e12)
            total_cpu_time = cpu_compute_time_sec

            total_step_latency = max(total_gpu_time, total_cpu_time)

            if total_step_latency < min_max_latency:
                min_max_latency = total_step_latency
                best_q = q
                best_gpu_time = total_gpu_time * 1000.0  # ms
                best_cpu_time = total_cpu_time * 1000.0  # ms

        selected_gpu_experts = [missing_experts[i] for i in indexed_missing[:best_q]]
        selected_cpu_experts = [missing_experts[i] for i in indexed_missing[best_q:]]

        return PartitionDecision(
            gpu_expert_indices=selected_gpu_experts,
            cpu_expert_indices=selected_cpu_experts,
            optimal_q=best_q,
            predicted_gpu_time_ms=best_gpu_time,
            predicted_cpu_time_ms=best_cpu_time,
            predicted_total_time_ms=min_max_latency * 1000.0
        )
