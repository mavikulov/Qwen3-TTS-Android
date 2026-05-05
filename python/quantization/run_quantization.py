import argparse

from onnxruntime.quantization import QuantType

from package.utils import resolve_path
from quantizer import QuantizationTool


DEFAULT_EXCLUDE_PATTERNS = [
    "lm_head", "out_proj", "o_proj", "down_proj", "fc2",
    "head", "output", "proj_out", "final_norm", "logits",
    "_to_copy", "val_36", "val_60", "slice_2", "slice_4"
]

DEFAULT_EXTRA_OPTIONS = {
    "MatMulConstBOnly": True,
    "ForceSymmetric": False,
    "ActivationSymmetric": True,
    "WeightSymmetric": True,
    "EnableSubgraph": True,
    "AddQDQToOutput": False,
    "QuantizeLinearMatMul": True,
    "ReduceOp": True
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Universal Model Quantization Tool")
    parser.add_argument("--input", required=True, help="Path to input ONNX model or embeddings directory")
    parser.add_argument("--output", required=True, help="Path to output file or directory")
    parser.add_argument(
        "--type", 
        choices=["int4", "int8", "uint8", "float16", "float32"], 
        default="int8",
        help="Target quantization type"
    )
    parser.add_argument("--per-channel", action="store_true", help="Enable per-channel quantization (ONNX only)")
    parser.add_argument("--exclude-patterns", nargs="+", default=[], help="Additional name patterns to exclude")
    parser.add_argument("--exclude-ops", nargs="+", default=[], help="Operation types to exclude (e.g., Slice, Cast)")
    parser.add_argument("--no-extra-options", action="store_true", help="Disable default extra options")
    parser.add_argument("--quantize-int-weights", action="store_true", help="Quantize numpy weights to int8 (experimental)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def main():
    args = parse_args()
    tool = QuantizationTool(verbose=args.verbose)
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    is_onnx = input_path.suffix.lower() == ".onnx"
    is_dir = input_path.is_dir()
    
    if not is_onnx and not is_dir:
        raise ValueError("Input must be either an .onnx file or a directory containing .npy files")

    if is_onnx:
        exclude_patterns = DEFAULT_EXCLUDE_PATTERNS + args.exclude_patterns
        nodes_to_exclude = tool.get_nodes_to_exclude(
            input_path, 
            patterns=exclude_patterns, 
            op_types=args.exclude_ops
        )
        
        extra_opts = None if args.no_extra_options else DEFAULT_EXTRA_OPTIONS
        tool.quantize_onnx(
            input_path=input_path,
            output_path=output_path,
            weight_type=args.type,
            nodes_to_exclude=nodes_to_exclude,
            extra_options=extra_opts,
            per_channel=args.per_channel
        )

    elif is_dir:
        if args.type not in ["float16", "float32"] and not args.quantize_int_weights:
            print(f"Warning: Type '{args.type}' requested for weights. Defaulting to float16 conversion unless --quantize-int-weights is set.")
            target = "float16" if args.type in ["int4", "int8"] else args.type
        else:
            target = args.type
            
        tool.convert_numpy_weights(
            input_dir=input_path,
            output_dir=output_path,
            target_dtype=target,
            quantize_int=args.quantize_int_weights
        )

    print("\nOperation completed successfully.")


if __name__ == "__main__":
    main()
