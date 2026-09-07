import os
import tempfile
import pytest
import torch

from freetoken.policy import QStarPolicy, HardwareBandwidthProfile
from freetoken.cache import GlobalLRUExpertCache
from freetoken.coordinator import ElasticMemoryCoordinator, MemoryBudget
from freetoken.format import FTWFormat
from freetoken.layer import FreeTokenMoELayer, SingleExpert
from freetoken.engine import FreeTokenEngine, EngineConfig, MoEModelConfig


def test_hardware_bandwidth_benchmark():
    device = torch.device("cpu")
    bw = HardwareBandwidthProfile.benchmark_pcie(device=device)
    assert bw > 0.0


def test_q_star_policy_partitioning():
    profile = HardwareBandwidthProfile(
        pcie_bandwidth_gbps=16.0,
        cpu_tflops=1.0,
        gpu_tflops=30.0,
        expert_size_bytes=64 * 1024 * 1024
    )
    policy = QStarPolicy(profile)

    missing_experts = [1, 3, 5, 7]
    tokens_per_expert = [10, 5, 20, 2]
    hidden_dim = 512
    intermediate_dim = 1024

    decision = policy.compute_optimal_split(
        missing_experts=missing_experts,
        tokens_per_expert=tokens_per_expert,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim
    )

    assert decision.optimal_q >= 0
    assert len(decision.gpu_expert_indices) + len(decision.cpu_expert_indices) == len(missing_experts)
    assert set(decision.gpu_expert_indices + decision.cpu_expert_indices) == set(missing_experts)
    assert decision.predicted_total_time_ms >= 0


def test_global_lru_expert_cache():
    cache = GlobalLRUExpertCache(capacity_experts=2, device=torch.device("cpu"))
    
    # Put 2 items
    cache.put(0, 1, "exp1")
    cache.put(0, 2, "exp2")
    assert cache.current_size == 2
    assert cache.contains(0, 1)
    assert cache.contains(0, 2)

    # Access exp1 to make exp2 LRU
    val = cache.get(0, 1)
    assert val == "exp1"

    # Put 3rd item -> exp2 evicted
    evicted = cache.put(0, 3, "exp3")
    assert evicted is not None
    assert evicted[0] == (0, 2)
    assert not cache.contains(0, 2)
    assert cache.contains(0, 1)
    assert cache.contains(0, 3)


def test_ftw_format_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        ftw_path = os.path.join(tmpdir, "test_weights.ftw")
        
        tensors = {
            "expert_0_gate": torch.randn(128, 256, dtype=torch.float32),
            "expert_0_up": torch.randn(128, 256, dtype=torch.float16),
            "expert_0_down": torch.randint(0, 100, (64, 64), dtype=torch.int8),
        }

        metadata = {"model": "test-moe", "format_version": 1}
        FTWFormat.save(ftw_path, tensors, metadata=metadata)

        loaded = FTWFormat.load(ftw_path, device="cpu")
        assert len(loaded) == 3
        assert torch.allclose(tensors["expert_0_gate"], loaded["expert_0_gate"])
        assert torch.allclose(tensors["expert_0_up"], loaded["expert_0_up"])
        assert torch.equal(tensors["expert_0_down"], loaded["expert_0_down"])


def test_elastic_memory_coordinator():
    cache = GlobalLRUExpertCache(capacity_experts=10, device=torch.device("cpu"))
    budget = MemoryBudget(
        total_vram_bytes=8 * (1024 ** 3),
        base_model_vram_bytes=2 * (1024 ** 3),
        reserved_vram_bytes=1 * (1024 ** 3),
        bytes_per_expert=100 * (1024 ** 2),   # 100MB per expert
        bytes_per_kv_token=1024               # 1KB per token
    )

    coordinator = ElasticMemoryCoordinator(
        budget=budget,
        expert_cache=cache,
        min_resident_experts=2,
        min_kv_tokens=1024
    )

    # Dynamic VRAM = 8 - 2 - 1 = 5 GB = 5120 MB
    initial_cap = coordinator.expert_cache.capacity
    assert initial_cap > 2

    # Request large KV cache: e.g. 4M tokens (4GB)
    coordinator.rebalance(requested_kv_tokens=4 * 1024 * 1024)
    # Expert cache capacity should dynamically shrink
    assert coordinator.expert_cache.capacity < initial_cap
    assert coordinator.expert_cache.capacity >= coordinator.min_resident_experts


def test_freetoken_moe_layer_forward():
    device = torch.device("cpu")
    hidden_dim = 64
    intermediate_dim = 128
    num_experts = 4
    top_k = 2

    cache = GlobalLRUExpertCache(capacity_experts=2, device=device)
    policy = QStarPolicy()

    layer = FreeTokenMoELayer(
        layer_idx=0,
        num_experts=num_experts,
        top_k=top_k,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        expert_cache=cache,
        q_star_policy=policy,
        device=device
    )

    batch_size = 2
    seq_len = 8
    x = torch.randn(batch_size, seq_len, hidden_dim)

    out = layer(x)
    assert out.shape == (batch_size, seq_len, hidden_dim)
    assert not torch.isnan(out).any()


def test_freetoken_engine_generation():
    m_cfg = MoEModelConfig(
        vocab_size=100,
        hidden_dim=64,
        intermediate_dim=128,
        num_layers=2,
        num_experts=4,
        top_k=2,
        max_seq_len=128
    )
    e_cfg = EngineConfig(
        total_vram_gb=1.0,
        base_model_vram_gb=0.2,
        reserved_vram_gb=0.1,
        initial_expert_capacity=2,
        device="cpu"
    )

    engine = FreeTokenEngine(m_cfg, e_cfg)
    prompt_ids = [12, 45, 78]

    generated_tokens = list(engine.generate(prompt_ids=prompt_ids, max_new_tokens=5, temperature=0.0))
    assert len(generated_tokens) == 5
    for tok in generated_tokens:
        assert isinstance(tok, int)
        assert 0 <= tok < m_cfg.vocab_size

    metrics = engine.get_metrics()
    assert "cache_stats" in metrics
    assert "memory_status" in metrics
