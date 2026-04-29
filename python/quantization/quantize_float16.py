import argparse
from pathlib import Path

import onnx
from onnxconverter_common.float16 import convert_float_to_float16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to source talker_prefill.onnx")
    parser.add_argument("--output", required=True, help="Path to output talker_prefill_fp16.onnx")
    parser.add_argument(
        "--disable-shape-infer",
        action="store_true",
        help="Disable ONNX shape/type inference during conversion",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(input_path), load_external_data=True)

    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=args.disable_shape_infer,
    )

    onnx.save_model(
        model_fp16,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_path.name + ".data",
        size_threshold=1024,
        convert_attribute=False,
    )

    print(f"Saved FP16 model to: {output_path}")
    print(f"External data: {output_path}.data")


if __name__ == "__main__":
    main()