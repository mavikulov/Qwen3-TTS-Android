import os
import shutil
import argparse
from pathlib import Path

import onnx 
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxconverter_common import float16


EXCLUDE = ["lm_head", "out_proj", "o_proj", "down_proj", "fc2",
            "head", "output", "proj_out", "final_norm", "logits",
           "layers.20", "layers.21", "layers.22", "layers.23",
           "layers.24", "layers.25", "layers.26", "layers.27",
           "_to_copy", "val_36", "val_60", "slice_2", "slice_4"]


DECODE_PATTERNS = [
    "lm_head", "o_proj", "down_proj", "fc2", "out_proj",
    "add", "rms_norm", "layer_norm", "residual",
    "layers.0", "layers.1", "layers.2",
    "layers.24", "layers.25", "layers.26", "layers.27"
]


def get_nodes_to_exclude(onnx_path, patterns, op_types_to_exclude=None):
    proto = onnx.load(onnx_path)
    excluded = []
    for n in proto.graph.node:
        if any(p in n.name for p in patterns):
            excluded.append(n.name)
        if op_types_to_exclude and n.op_type in op_types_to_exclude:
            excluded.append(n.name)
    print(f"  Исключено {len(excluded)} узлов")
    return excluded


if __name__ == "__main__":
    onnx_path = Path("/Users/ruaoccj/qwen3_tts_bundle/talker_decode.onnx")
    patterns = EXCLUDE.copy()
    patterns.extend(DECODE_PATTERNS)
    fp32_path = onnx_path
    excluded = get_nodes_to_exclude(
        fp32_path,
        patterns,
        op_types_to_exclude=["Slice", "Concat", "Where", "Cast"])
    path_to_quantized = Path("/Users/ruaoccj/talker_prefil_custom.onnx")
    
    extra_options = {
        "MatMulConstBOnly": True,              # уже есть — квантуем только веса
        "ForceSymmetric": False,                # симметричная квантизация (быстрее на ARM/CPU)
        "ActivationSymmetric": True,           # симметричные активации
        "WeightSymmetric": True,               # явное указание (по умолчанию True)
        "EnableSubgraph": True,                # квантизация подграфов целиком
        "AddQDQToOutput": False,               # не добавляем QDQ на выход — меньше накладных расходов
        "QuantizeLinearMatMul": True,          # явное включение QLinearMatMul ядер
        "ReduceOp": True,                      # квантовать Reduce-операции
    }

    data_file = str(fp32_path) + ".data"
    has_external = os.path.exists(data_file)

    quantize_dynamic(
        fp32_path,
        path_to_quantized,
        weight_type=QuantType.QInt4,
        nodes_to_exclude=excluded,
        use_external_data_format=has_external,
        extra_options=extra_options,
    )