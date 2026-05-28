"""
infer.py — Inference script for Piano Score Rearrangement

Rearranges an input MXL piano score to a target difficulty level using
the trained seq2seq Transformer.

Usage:
    python infer.py --input input.mxl --output output.mxl --level Lv.2
    python infer.py --input input.mxl --output output.mxl --level Lv.1 --checkpoint data/checkpoints/best.pt
    python infer.py --input input.mxl --output output.mxl --level Lv.3 --seg_len 6 --device cuda:0

Pipeline:
    Input MXL
       ↓ MusicXML_to_tokens()     tokenize to ST+ format
       ↓ split_into_bars()        split into bars
       ↓ assign_level()           detect source difficulty
       ↓ segment into chunks      non-overlapping windows of --seg_len bars
       ↓ [Dsrc, Dtgt] prepend    difficulty conditioning
       ↓ model.greedy_decode()    autoregressive generation
       ↓ strip Dtgt token         remove conditioning prefix from output
       ↓ validate every bar       fall back to source if any bar is malformed
       ↓ concatenate segments     stitch all segments back together
       ↓ tokens_to_score()        detokenize to music21 Score
    Output MXL
"""

import argparse
import json
import os
import sys

import torch

from model import build_model
from score_to_tokens import MusicXML_to_tokens
from tokens_to_score import tokens_to_score
from build_pairs import split_into_bars, bars_to_tokens, assign_level


VALID_LEVELS  = ('Lv.1', 'Lv.2', 'Lv.3', 'Lv.4')
MAX_SRC_LEN   = 1024   # must match PositionalEncoding max_len in model.py

# Tokens that are part of a note group but are not the note/len themselves
_NOTE_GROUP_CONTINUATIONS = {'stem', 'beam', 'tie', 'staccato', 'accent', 'tenuto'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_segment(bar_tokens, src_level, tgt_level, token_to_id, eos_id):
    """
    Build the encoder input IDs for one segment:
        [Dsrc, Dtgt, <segment_tokens...>, <eos>]

    Tokens not present in the vocabulary are silently skipped.
    If the result exceeds MAX_SRC_LEN, the middle tokens are truncated
    while keeping the conditioning prefix and the <eos> suffix intact.
    """
    ids = [token_to_id[src_level], token_to_id[tgt_level]]
    for tok in bar_tokens:
        if tok in token_to_id:
            ids.append(token_to_id[tok])
    ids.append(eos_id)

    # Truncate to positional encoding limit: keep [Dsrc, Dtgt, ...truncated..., <eos>]
    if len(ids) > MAX_SRC_LEN:
        ids = ids[:MAX_SRC_LEN - 1] + [eos_id]

    return ids


def ids_to_tokens(id_list, id_to_token):
    """Convert a list of integer token IDs to token strings."""
    return [id_to_token[i] for i in id_list if i in id_to_token]


def _bar_is_valid(bar_tokens):
    """
    Return True if a single bar's token list is structurally valid:
      1. Has at least one 'R' hand marker (required by split_header_R_L).
      2. Every note/rest group has at least one 'len_' token before the
         group ends — mirrors the same group-boundary logic used by
         group_related_tokens in tokens_to_score.py so that note_token_to_obj
         never receives an empty lengths list (which causes IndexError).
    """
    if 'R' not in bar_tokens:
        return False

    in_note = False
    has_len = False
    for t in bar_tokens:
        prefix = t.split('_')[0]
        if prefix in ('note', 'rest'):
            if in_note and not has_len:
                return False          # previous group had no length
            in_note = True
            has_len = False
        elif prefix == 'len':
            has_len = True
        elif prefix in _NOTE_GROUP_CONTINUATIONS or t in ('slur_start', 'slur_stop'):
            pass                      # still inside the note group
        else:
            if in_note and not has_len:
                return False          # group ended without a length
            in_note = False
            has_len = False

    if in_note and not has_len:
        return False                  # last group in bar had no length
    return True


def is_valid_segment(tokens):
    """
    Return True if every bar in the decoded segment is structurally valid.
    Falls back to source if any bar fails _bar_is_valid.
    """
    bars = []
    current = []
    for t in tokens:
        if t == 'bar':
            if current:
                bars.append(current)
            current = []
        else:
            current.append(t)
    if current:
        bars.append(current)
    return bool(bars) and all(_bar_is_valid(b) for b in bars)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Rearrange a piano score to a target difficulty level.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--input', required=True, help='Input MXL or XML file')
    p.add_argument('--output', required=True, help='Output MXL file')
    p.add_argument(
        '--level', required=True, choices=VALID_LEVELS,
        help='Target difficulty level',
    )
    p.add_argument(
        '--checkpoint', default='data/checkpoints/best.pt',
        help='Trained model checkpoint (.pt file)',
    )
    p.add_argument(
        '--vocab', default='data/vocab.json',
        help='Vocabulary file (vocab.json)',
    )
    p.add_argument(
        '--seg_len', type=int, default=8,
        help='Bars per inference segment (4–8 recommended)',
    )
    p.add_argument(
        '--max_decode_len', type=int, default=1024,
        help='Max tokens generated per segment',
    )
    p.add_argument(
        '--temperature', type=float, default=1.2,
        help='Sampling temperature (>1 adds variety, 1.0 = near-greedy). '
             'Use with --top_k for best results.',
    )
    p.add_argument(
        '--top_k', type=int, default=10,
        help='Top-k sampling (0 = greedy argmax, 5–20 recommended for music)',
    )
    p.add_argument(
        '--device', default=None,
        help='Device override (e.g. cpu, cuda:0). Auto-detected if omitted.',
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Strip accidental whitespace from file paths (Windows path errors if present)
    args.input  = args.input.strip()
    args.output = args.output.strip()

    # ── device ────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device          : {device}')

    # ── vocab ─────────────────────────────────────────────────────────────
    with open(args.vocab, encoding='utf-8') as f:
        vocab_data = json.load(f)

    token_to_id = vocab_data['token_to_id']
    id_to_token = {v: k for k, v in token_to_id.items()}  # int → str

    vocab_size = len(token_to_id)
    pad_id     = token_to_id['<pad>']
    sos_id     = token_to_id['<sos>']
    eos_id     = token_to_id['<eos>']

    for lv in VALID_LEVELS:
        if lv not in token_to_id:
            print(f'Error: level token "{lv}" missing from vocab.', file=sys.stderr)
            sys.exit(1)

    # ── model ─────────────────────────────────────────────────────────────
    print(f'Loading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(vocab_size, pad_id).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'  Epoch {ckpt.get("epoch", "?"):>4}  val_loss={ckpt.get("val_loss", float("nan")):.4f}  '
          f'params={model.count_parameters():,}')

    # ── tokenize input ────────────────────────────────────────────────────
    print(f'\nTokenizing: {args.input}')
    try:
        tokens = MusicXML_to_tokens(args.input, bar_major=True, note_name=True)
    except Exception as e:
        print(f'Error tokenizing input: {e}', file=sys.stderr)
        sys.exit(1)

    bars      = split_into_bars(tokens)
    src_level = assign_level(bars)
    tgt_level = args.level

    print(f'  Bars            : {len(bars)}')
    print(f'  Source level    : {src_level}')
    print(f'  Target level    : {tgt_level}')

    if len(bars) == 0:
        print('Error: no bars found in input score.', file=sys.stderr)
        sys.exit(1)

    if src_level == tgt_level:
        print('  Warning: source and target levels are the same — output may be unchanged.')

    # ── segment → model → collect outputs ─────────────────────────────────
    seg_len    = max(4, min(args.seg_len, 8))
    starts     = list(range(0, len(bars), seg_len))
    n_segs     = len(starts)
    n_fallback = 0
    print(f'\nRunning model: {n_segs} segment(s), up to {seg_len} bars each')

    all_output_tokens = []

    for seg_idx, start in enumerate(starts):
        seg_bars   = bars[start: start + seg_len]
        seg_tokens = bars_to_tokens(seg_bars)

        src_ids    = encode_segment(seg_tokens, src_level, tgt_level, token_to_id, eos_id)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

        decoded_ids = model.greedy_decode(
            src_tensor,
            sos_id,
            eos_id,
            max_len=args.max_decode_len,
            init_token_idx=token_to_id[tgt_level],
            temperature=args.temperature,
            top_k=args.top_k,
        )[0]

        decoded_tokens = ids_to_tokens(decoded_ids, id_to_token)

        # init_token_idx forces Dtgt as the first output — strip it before stitching
        if decoded_tokens and decoded_tokens[0] == tgt_level:
            decoded_tokens = decoded_tokens[1:]

        # Validate every bar: requires 'R' in each bar AND every note/rest
        # group must have a len_ token, otherwise note_token_to_obj crashes.
        if not is_valid_segment(decoded_tokens):
            n_fallback += 1
            decoded_tokens = seg_tokens

        all_output_tokens.extend(decoded_tokens)

        if (seg_idx + 1) % 10 == 0 or (seg_idx + 1) == n_segs:
            print(f'  [{seg_idx + 1}/{n_segs}]  output tokens so far: {len(all_output_tokens)}'
                  + (f'  (fallbacks so far: {n_fallback})' if n_fallback else ''))

    if n_fallback:
        print(f'  Note: {n_fallback}/{n_segs} segment(s) used source pass-through '
              f'(model output had bars with missing R or note-length tokens).')

    if not all_output_tokens:
        print('Error: model produced no output tokens.', file=sys.stderr)
        sys.exit(1)

    # ── detokenize & write output ─────────────────────────────────────────
    print(f'\nDetokenizing {len(all_output_tokens)} tokens...')
    try:
        score = tokens_to_score(all_output_tokens)
    except Exception as e:
        print(f'Error detokenizing: {e}', file=sys.stderr)
        print('Try --temperature 1.0 --top_k 0 for stricter greedy decoding.', file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    try:
        score.write('musicxml', fp=args.output)
    except Exception as e:
        print(f'Error writing output: {e}', file=sys.stderr)
        sys.exit(1)

    print(f'Output written  : {args.output}')


if __name__ == '__main__':
    main()