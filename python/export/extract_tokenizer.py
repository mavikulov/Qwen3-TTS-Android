import argparse
import json
from pathlib import Path

from export_utils import configure_output_encoding
from transformers import AutoTokenizer


DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
OUTPUT_DIR = Path(__file__).parent / "tokenizer_artifacts"


def extract_tokenizer_files(tokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved tokenizer files to {output_dir}")

    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size:,} bytes)")


def build_validation_cases(tokenizer) -> list[dict]:
    cases = []

    def add(description: str, text: str) -> None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        cases.append({
            "description": description,
            "text": text,
            "token_ids": ids,
            "num_tokens": len(ids),
        })

    add("simple_english", "Hello, world!")
    add("english_sentence", "The quick brown fox jumps over the lazy dog.")
    add("english_tts_phrase", "Welcome to the Qwen text to speech system.")

    add("simple_chinese", "你好世界")
    add("chinese_sentence", "今天天气真不错，我们一起出去走走吧。")

    add("mixed_en_zh", "Hello你好，Welcome欢迎!")
    add("mixed_with_numbers", "第3次世界大战在2045年")

    add("punctuation", "Hello! How are you? I'm fine, thanks.")
    add("special_chars", "Email: user@example.com — price: $19.99")
    add("unicode_symbols", "Temperature: 72°F → 22°C ✓")

    add("pure_numbers", "1234567890")
    add("decimal_number", "3.14159265")

    add("single_space", " ")
    add("single_char", "A")
    add("newline", "\n")

    assistant_text = "<|im_start|>assistant\nHello, world!<|im_end|>\n<|im_start|>assistant\n"
    add(
        "chat_template_english",
        assistant_text,
    )

    assistant_text_zh = "<|im_start|>assistant\n你好世界<|im_end|>\n<|im_start|>assistant\n"
    add(
        "chat_template_chinese",
        assistant_text_zh,
    )

    instruct_text = "<|im_start|>user\nSpeak slowly and clearly.<|im_end|>\n"
    add(
        "instruct_format",
        instruct_text,
    )

    return cases


def save_special_tokens_summary(tokenizer: AutoTokenizer, output_dir: Path) -> None:
    summary = {
        "eos_token": {"token": str(tokenizer.eos_token), "id": tokenizer.eos_token_id},
        "pad_token": {"token": str(tokenizer.pad_token), "id": tokenizer.pad_token_id},
        "bos_token": {"token": str(tokenizer.bos_token), "id": tokenizer.bos_token_id},
    }

    added_tokens = {}
    if hasattr(tokenizer, "added_tokens_encoder"):
        for token_str, token_id in sorted(
            tokenizer.added_tokens_encoder.items(), key=lambda x: x[1]
        ):
            added_tokens[token_str] = token_id

    summary["added_tokens"] = added_tokens
    summary["vocab_size"] = tokenizer.vocab_size

    out_path = output_dir / "special_tokens_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved special tokens summary to {out_path}")


def main() -> None:
    configure_output_encoding()
    parser = argparse.ArgumentParser(description="Extract BPE tokenizer artifacts for C#")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID to load tokenizer from (default: {DEFAULT_MODEL_ID})",
    )
    args = parser.parse_args()

    model_id = args.model
    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"Tokenizer type: {type(tokenizer).__name__}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    extract_tokenizer_files(tokenizer, OUTPUT_DIR)
    save_special_tokens_summary(tokenizer, OUTPUT_DIR)
    cases = build_validation_cases(tokenizer)
    validation_path = OUTPUT_DIR / "validation_cases.json"
    with open(validation_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(cases)} validation cases to {validation_path}")

    print("\n--- Validation Case Preview ---")
    for case in cases:
        desc = case["description"]
        text_preview = case["text"][:40].replace("\n", "\\n")
        n = case["num_tokens"]
        print(f"  {desc:30s} | {n:3d} tokens | {text_preview!r}")

    print("\nDone. All artifacts saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
