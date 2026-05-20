import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

from export_utils import configure_output_encoding


DEFAULT_REPO_ID = "mavikulov/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX"


def main():
    configure_output_encoding()
    parser = argparse.ArgumentParser(description="Download ONNX models from HuggingFace Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace repo ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="onnx_runtime",
        help="Local directory to save models (default: onnx_runtime)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading models from {args.repo_id} → {output_dir.resolve()}")
    files = list_repo_files(args.repo_id)
    files = [f for f in files if f != "README.md" and not f.startswith(".")]
    total = len(files)
    
    for idx, filename in enumerate(sorted(files), 1):
        local_path = output_dir / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists():
            print(f"  [{idx}/{total}] ✓ {filename} (already exists)")
            continue

        print(f"  [{idx}/{total}] ↓ {filename} ...")
        downloaded = hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            local_dir=str(output_dir),
        )

    print(f"\n✅ All models downloaded to {output_dir.resolve()}")
    print(f"\nRun the C# app:")
    print(f"  dotnet run --project src/QwenTTS -- --model-dir {output_dir} --text \"Hello world\" --speaker ryan --language english")


if __name__ == "__main__":
    main()
