import os
import shutil
from pathlib import Path
from typing import Any, Optional

import onnx
import numpy as np
from onnxruntime.quantization import quantize_dynamic, QuantType


TYPE_MAPPING = {
    "int4": QuantType.QInt4,
    "int8": QuantType.QInt8,
    "uint8": QuantType.QUInt8
}

NPY_FILES_CONFIG = {
    "main": [
        "text_embedding.npy",
        "text_projection_fc1_weight.npy",
        "text_projection_fc1_bias.npy",
        "text_projection_fc2_weight.npy",
        "text_projection_fc2_bias.npy",
        "talker_codec_embedding.npy",
    ],
    "cp_prefix": "cp_codec_embedding_",
    "cp_count": 15
}


class QuantizationTool:
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose
        
    def log(self, message: str):
        if self.verbose:
            print(message)
            
    def get_nodes_to_exclude(
        self,
        onnx_path: Path,
        patterns: list[str],
        op_types: Optional[list[str]] = None
    ) -> list[str]:
        if not onnx_path.exists():
            raise FileNotFoundError(f"Model not found: {onnx_path}")
        
        proto = onnx.load(str(onnx_path))
        excluded_names = set()
        
        for node in proto.graph.node:
            if any(pattern in node.name for pattern in patterns):
                excluded_names.add(node.name)
            if op_types and node.op_type in op_types:
                excluded_names.add(node.name)
        
        self.log(f"Найдено {len(excluded_names)} узлов для исключения")
        return list(excluded_names)
    
    def quantize_onnx(
        self,
        input_path: Path,
        output_path: Path,
        weight_type: str,
        nodes_to_exclude: Optional[list[str]] = None,
        extra_options: Optional[dict[str, Any]] = None,
        per_channel: bool = False
    ) -> None:
        self.log(f"Start ONNX quantization: {input_path} -> {output_path}")
        self.log(f"  Type: {weight_type}, PerChannel: {per_channel}")
        
        if nodes_to_exclude:
            self.log(f"  Excluding {len(nodes_to_exclude)} nodes")

        quantized_type = TYPE_MAPPING.get(weight_type)
        if not quantized_type:
            raise ValueError(f"Неизвестный тип квантования: {weight_type}")

        has_external = os.path.exists(str(input_path) + ".data")
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=quantized_type,
            per_channel=per_channel,
            nodes_to_exclude=nodes_to_exclude or [],
            use_external_data_format=has_external,
            extra_options=extra_options or {}
        )
        self.log(f"Success: {output_path}")
        
    def convert_numpy_weights(
        self,
        input_dir: Path,
        output_dir: Path,
        target_dtype: str,
        quantize_int: bool = False
    ) -> None:
        self.log(f"Start weights conversion: {input_dir} -> {output_dir}")
        self.log(f"  Target dtype: {target_dtype}, Quantize Int: {quantize_int}")

        output_dir.mkdir(parents=True, exist_ok=True)
        config_src = input_dir / "config.json"
        if config_src.exists():
            shutil.copy2(config_src, output_dir / "config.json")
            self.log(f"  Copied config.json")

        files_to_process = NPY_FILES_CONFIG["main"].copy()
        for i in range(NPY_FILES_CONFIG["cp_count"]):
            files_to_process.append(f"{NPY_FILES_CONFIG['cp_prefix']}{i}.npy")

        for filename in files_to_process:
            src = input_dir / filename
            if not src.exists():
                self.log(f"  Skip (not found): {filename}")
                continue
            
            dst = output_dir / filename
            arr = np.load(src)
            
            if target_dtype == "float16":
                arr_out = arr.astype(np.float16)
            elif target_dtype == "float32":
                arr_out = arr.astype(np.float32)
            else:
                raise ValueError(f"Unsupported dtype for weights: {target_dtype}")
            dtype_str = target_dtype

            np.save(dst, arr_out)
            src_size = src.stat().st_size / (1024 * 1024)
            dst_size = dst.stat().st_size / (1024 * 1024)
            self.log(f"  {filename}: {arr.dtype} -> {dtype_str} ({src_size:.2f}MB -> {dst_size:.2f}MB)")

        self.log("Weights conversion done.")
