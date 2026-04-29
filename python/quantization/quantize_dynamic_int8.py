import argparse
from pathlib import Path

from onnxruntime.quantization import quantize_dynamic, QuantType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to source ONNX model")
    parser.add_argument("--output", required=True, help="Path to output quantized ONNX model")
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Enable per-channel quantization where supported"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        per_channel=args.per_channel,
    )

    print(f"Saved INT8 dynamic model to: {output_path}")


if __name__ == "__main__":
    main()