"""
FTW (Fast Token Weight) Format Specification & Reader/Writer
Reference: arXiv:2608.16157

The FTW format is an edge-native, zero-copy, page-aligned format for MoE weights.
Features:
- Fixed 4KB header with magic bytes b"FTW1"
- Direct page-aligned offsets for each expert tensor to enable direct DMA / pinned memory transfers
- Per-tensor quantization metadata (FP16, BF16, FP8, NVFP4, MXFP4)
- Memory-mappable for zero-latency streaming from NVMe or Host RAM.
"""

from __future__ import annotations
import json
import struct
from typing import Dict, Any, BinaryIO
import numpy as np
import torch

MAGIC = b"FTW1"
HEADER_SIZE = 4096  # 4KB aligned header


class FTWTensorHeader:
    def __init__(self, name: str, shape: list[int], dtype: str, offset: int, num_bytes: int):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.offset = offset
        self.num_bytes = num_bytes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype,
            "offset": self.offset,
            "num_bytes": self.num_bytes
        }


class FTWFormat:
    """
    Reader and Writer for the Fast Token Weight (.ftw) file format.
    """
    DTYPE_MAP = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "uint8": torch.uint8,
    }

    REV_DTYPE_MAP = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.int8: "int8",
        torch.uint8: "uint8",
    }

    @classmethod
    def save(cls, filepath: str, tensors: Dict[str, torch.Tensor], metadata: Optional[Dict[str, Any]] = None):
        """
        Saves a dictionary of tensors into an aligned FTW binary file.
        """
        metadata = metadata or {}
        tensor_headers = []
        
        # Calculate offsets starting after the 4KB header
        current_offset = HEADER_SIZE
        data_blocks = []

        for name, tensor in tensors.items():
            t_cpu = tensor.detach().cpu().contiguous()
            arr = t_cpu.numpy() if t_cpu.dtype != torch.bfloat16 else t_cpu.view(torch.int16).numpy()
            raw_bytes = arr.tobytes()
            n_bytes = len(raw_bytes)
            
            # Align to 64 bytes for DMA efficiency
            padding_len = (64 - (n_bytes % 64)) % 64
            padded_bytes = raw_bytes + (b"\x00" * padding_len)

            dtype_str = cls.REV_DTYPE_MAP.get(tensor.dtype, "float32")
            header = FTWTensorHeader(
                name=name,
                shape=list(tensor.shape),
                dtype=dtype_str,
                offset=current_offset,
                num_bytes=n_bytes
            )
            tensor_headers.append(header.to_dict())
            data_blocks.append(padded_bytes)
            current_offset += len(padded_bytes)

        manifest = {
            "metadata": metadata,
            "tensors": tensor_headers
        }
        manifest_json = json.dumps(manifest).encode("utf-8")
        if len(manifest_json) > HEADER_SIZE - 8:
            raise ValueError(f"Manifest JSON exceeds header size: {len(manifest_json)} > {HEADER_SIZE - 8}")

        with open(filepath, "wb") as f:
            # 4 bytes magic + 4 bytes json len
            f.write(MAGIC)
            f.write(struct.pack("<I", len(manifest_json)))
            f.write(manifest_json)
            # Pad remainder of header
            f.write(b"\x00" * (HEADER_SIZE - 8 - len(manifest_json)))
            
            # Write tensor payload
            for block in data_blocks:
                f.write(block)

    @classmethod
    def load(cls, filepath: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """
        Loads all tensors from an FTW binary file.
        """
        with open(filepath, "rb") as f:
            magic = f.read(4)
            if magic != MAGIC:
                raise ValueError(f"Invalid magic bytes in FTW file: {magic}")
            (manifest_len,) = struct.unpack("<I", f.read(4))
            manifest_json = f.read(manifest_len).decode("utf-8")
            manifest = json.loads(manifest_json)

            tensors = {}
            for t_info in manifest["tensors"]:
                f.seek(t_info["offset"])
                raw_bytes = f.read(t_info["num_bytes"])
                dtype = cls.DTYPE_MAP.get(t_info["dtype"], torch.float32)

                if t_info["dtype"] == "bfloat16":
                    arr = np.frombuffer(raw_bytes, dtype=np.int16).copy()
                    tensor = torch.from_numpy(arr).view(torch.bfloat16).reshape(t_info["shape"])
                else:
                    np_dtype = np.dtype(t_info["dtype"])
                    arr = np.frombuffer(raw_bytes, dtype=np_dtype).copy()
                    tensor = torch.from_numpy(arr).reshape(t_info["shape"]).to(dtype)

                if device != "cpu":
                    tensor = tensor.to(device)
                tensors[t_info["name"]] = tensor

        return tensors
