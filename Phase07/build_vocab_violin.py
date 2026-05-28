import os
import json
import glob

# ── Paths ─────────────────────────────────────────────────────────────────────
PIANO_VOCAB_PATH  = "../data/vocab.json"        # 現有的鋼琴 vocab，只讀不動
VIOLIN_TOKENS_DIR = "./violin_tokens"           # 小提琴 token 目錄
OUTPUT_PATH       = "../data/vocab_violin.json"  # 新產生的小提琴 vocab

# ── Special tokens ────────────────────────────────────────────────────────────
# 完全繼承現有的 special tokens 順序，再加 piano / violin
# 這樣鋼琴模型 fine-tune 時，原本學過的 Lv.* index 不會變
SPECIAL_TOKENS = [
    "<pad>",   # 0
    "<sos>",   # 1
    "<eos>",   # 2
    "<unk>",   # 3
    "Lv.1",    # 4
    "Lv.2",    # 5
    "Lv.3",    # 6
    "Lv.4",    # 7
    "piano",   # 8  — instrument conditioning（新增）
    "violin",  # 9  — instrument conditioning（新增）
]


def build_vocab_violin():
    # ── Step 1: 讀取現有的鋼琴 vocab，取出所有 corpus token ──────────────────
    print(f"Loading piano vocab from '{PIANO_VOCAB_PATH}' ...")
    if not os.path.exists(PIANO_VOCAB_PATH):
        print("ERROR: piano vocab not found. Run build_vocab.py first.")
        return

    with open(PIANO_VOCAB_PATH, encoding='utf-8') as f:
        piano_vocab = json.load(f)

    piano_token_to_id = piano_vocab["token_to_id"]
    # corpus tokens = 現有 vocab 裡，不在我們 SPECIAL_TOKENS 的那些
    piano_corpus = set(piano_token_to_id.keys()) - set(SPECIAL_TOKENS)
    print(f"  Piano vocab size     : {len(piano_token_to_id)}")
    print(f"  Piano corpus tokens  : {len(piano_corpus)}")

    # ── Step 2: 掃描小提琴 token 目錄，收集新出現的 token ────────────────────
    print(f"\nScanning violin tokens from '{VIOLIN_TOKENS_DIR}' ...")
    json_files = glob.glob(
        os.path.join(VIOLIN_TOKENS_DIR, "**", "*.json"), recursive=True
    )
    print(f"  Found {len(json_files)} violin token files")

    violin_corpus: set[str] = set()
    failed = 0

    for i, path in enumerate(json_files):
        try:
            with open(path, encoding='utf-8') as f:
                tokens = json.load(f)
            if isinstance(tokens, list):
                violin_corpus.update(tokens)
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  Warning: could not read {path}: {e}")

        if (i + 1) % 500 == 0:
            print(f"  Scanned {i + 1}/{len(json_files)} files ...")

    violin_corpus -= set(SPECIAL_TOKENS)
    print(f"  Violin corpus tokens : {len(violin_corpus)}")
    print(f"  Failed               : {failed}")

    # ── Step 3: 找出小提琴獨有的新 token（鋼琴 vocab 裡沒有的）────────────────
    new_tokens = violin_corpus - piano_corpus
    print(f"\nNew tokens not in piano vocab: {len(new_tokens)}")
    if new_tokens:
        print(f"  {sorted(new_tokens)}")

    # ── Step 4: 建立新詞表 ────────────────────────────────────────────────────
    # 繼承鋼琴的所有 corpus token，再加上小提琴獨有的新 token
    all_corpus = piano_corpus | violin_corpus
    vocab_list  = SPECIAL_TOKENS + sorted(all_corpus)
    token_to_id = {tok: idx for idx, tok in enumerate(vocab_list)}
    id_to_token = {str(idx): tok for idx, tok in enumerate(vocab_list)}

    print(f"\nFinal vocab_violin size: {len(vocab_list)} tokens")
    print(f"  (piano vocab was {len(piano_token_to_id)}, "
          f"added {len(vocab_list) - len(piano_token_to_id)} tokens)")

    # ── Step 5: 儲存 ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding='utf-8') as f:
        json.dump({"token_to_id": token_to_id, "id_to_token": id_to_token},
                  f, indent=2, ensure_ascii=False)
    print(f"Saved to '{OUTPUT_PATH}'")
    print(f"(Piano vocab '{PIANO_VOCAB_PATH}' is unchanged)")

    # ── Step 6: Sanity check ──────────────────────────────────────────────────
    print("\n── Sanity check ──────────────────────────────────────────────────")
    for tok in SPECIAL_TOKENS:
        print(f"  {tok!r:12s} → id {token_to_id[tok]}")


if __name__ == "__main__":
    build_vocab_violin()