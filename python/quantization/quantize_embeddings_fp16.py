import argparse
from pathlib import Path
import shutil
import numpy as np


MAIN_FILES = [
    "text_embedding.npy",
    "text_projection_fc1_weight.npy",
    "text_projection_fc1_bias.npy",
    "text_projection_fc2_weight.npy",
    "text_projection_fc2_bias.npy",
    "talker_codec_embedding.npy",
]

CP_PREFIX = "cp_codec_embedding_"
CP_COUNT = 15


def convert_array_file(src: Path, dst: Path) -> None:
    arr = np.load(src)
    arr_fp16 = arr.astype(np.float16)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, arr_fp16)
    src_size = src.stat().st_size / (1024 * 1024)
    dst_size = dst.stat().st_size / (1024 * 1024)
    print(
        f"{src.name}: {arr.dtype} {arr.shape} -> {arr_fp16.dtype}, "
        f"{src_size:.2f} MB -> {dst_size:.2f} MB"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Original embeddings directory")
    parser.add_argument("--output_dir", required=True, help="FP16 embeddings directory")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_src = input_dir / "config.json"
    config_dst = output_dir / "config.json"
    shutil.copy2(config_src, config_dst)
    print(f"Copied config: {config_src.name}")

    for name in MAIN_FILES:
        convert_array_file(input_dir / name, output_dir / name)

    for i in range(CP_COUNT):
        name = f"{CP_PREFIX}{i}.npy"
        convert_array_file(input_dir / name, output_dir / name)

    print("\nDone.")


if __name__ == "__main__":
    main()