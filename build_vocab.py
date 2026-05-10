import os
import json
import glob

# ── Paths ────────────────────────────────────────────────────────────────────
# TODO: change these to match your own environment
TOKENS_DIR  = "./tokens"
OUTPUT_PATH = "./data/vocab.json"

# ── Special tokens ────────────────────────────────────────────────────────────
# Order matters: padding must be index 0 so PyTorch's default ignore_index works.
SPECIAL_TOKENS = [
    "<pad>",   # 0 — used to pad sequences to the same length in a batch
    "<sos>",   # 1 — start-of-sequence (prepended to every target sequence)
    "<eos>",   # 2 — end-of-sequence   (appended to every target sequence)
    "<unk>",   # 3 — unknown token     (safety fallback; shouldn't appear often)
    "Lv.1",    # 4 — difficulty conditioning tokens (Section 3.1.2 of paper)
    "Lv.2",    # 5
    "Lv.3",    # 6
    "Lv.4",    # 7
]


def build_vocab(tokens_dir: str, output_path: str) -> None:
    """
    Scan every .json file under tokens_dir, collect all unique tokens,
    then write a vocab.json containing:
        {
            "token_to_id": { "<pad>": 0, "<sos>": 1, ..., "bar": 8, ... },
            "id_to_token": { "0": "<pad>", "1": "<sos>", ..., "8": "bar", ... }
        }
    """
    # ── Step 1: collect all token files ──────────────────────────────────────
    json_files = glob.glob(os.path.join(tokens_dir, "**", "*.json"), recursive=True)
    print(f"Found {len(json_files)} token files under '{tokens_dir}'")

    if not json_files:
        print("ERROR: No token files found. Has tokenize_all.py finished running?")
        return

    # ── Step 2: scan every file and collect unique tokens ────────────────────
    unique_tokens: set[str] = set()
    failed = 0

    for i, path in enumerate(json_files):
        try:
            with open(path, encoding='utf-8') as f:
                tokens: list[str] = json.load(f)
            unique_tokens.update(tokens)
        except Exception as e:
            failed += 1
            print(f"  Warning: could not read {path}: {e}")

        if (i + 1) % 10_000 == 0:
            print(f"  Scanned {i + 1}/{len(json_files)} files, "
                  f"{len(unique_tokens)} unique tokens so far …")

    print(f"\nFinished scanning. "
          f"{len(json_files) - failed} files OK, {failed} failed.")
    print(f"Unique tokens found in corpus: {len(unique_tokens)}")

    # ── Step 3: remove any token that overlaps with our special tokens ────────
    # (shouldn't happen, but just in case)
    unique_tokens -= set(SPECIAL_TOKENS)

    # ── Step 4: build the final ordered vocabulary ────────────────────────────
    # Special tokens come first (fixed indices), then corpus tokens sorted
    # alphabetically for reproducibility.
    vocab_list = SPECIAL_TOKENS + sorted(unique_tokens)
    token_to_id = {tok: idx for idx, tok in enumerate(vocab_list)}
    id_to_token = {str(idx): tok for idx, tok in enumerate(vocab_list)}

    print(f"Final vocabulary size: {len(vocab_list)} tokens")

    # ── Step 5: save ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump({"token_to_id": token_to_id, "id_to_token": id_to_token},
                  f, indent=2, ensure_ascii=False)

    print(f"Saved vocab to '{output_path}'")

    # ── Step 6: quick sanity check ────────────────────────────────────────────
    print("\n── Sanity check ──────────────────────────────────────────────────")
    for tok in SPECIAL_TOKENS:
        print(f"  {tok!r:12s} → id {token_to_id[tok]}")
    print(f"  First 5 corpus tokens: "
          f"{[vocab_list[len(SPECIAL_TOKENS) + i] for i in range(min(5, len(unique_tokens)))]}")


if __name__ == "__main__":
    build_vocab(TOKENS_DIR, OUTPUT_PATH)