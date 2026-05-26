"""build_pairs_duet.py -- Build training pairs for Phase 6.2.

Reads per-track tokenized duets from Phase06/tokens_duet/, synthesizes a
Pseudo Piano Solo by merging violin notes into the piano right hand (onset-
aligned chord merge), segments into 4-8 bar sliding windows, and emits two
training tasks per segment to Phase06/pairs_duet.jsonl:

    Dataset A (task='A'):  [Pseudo Piano Solo] -> [Violin Melody]
    Dataset B (task='B'):  [Pseudo Piano Solo] -> <track_violin> ... <track_piano> ...

The <track_violin> / <track_piano> markers are the same delimiters Phase 6.3
will add to vocab.json -- emitting them now keeps the data format stable.

Pseudo-solo synthesis details:
  - For each bar, parse piano R into ordered note/rest events and compute
    each event's onset by accumulating len_* durations.
  - Do the same for the violin bar; treat each violin note event as an
    interval [onset, onset + dur).
  - For each piano-R event window [onset, onset + dur), fold in any violin
    pitches whose interval overlaps the window as additional chord notes
    (deduplicated). Piano's rhythm is the master clock; violin rhythm is
    approximated by overlap, which trades some rhythmic precision for a
    clean single-voice output that looks like real piano-solo notation.
  - If a piano-R event is a rest but violin notes overlap, the rest is
    upgraded to a note made of the violin pitches.
  - Bars containing <voice>/</voice> markers (multi-voice within one hand)
    are kept unchanged -- onset math is ambiguous across concurrent voices
    and merging would corrupt the bar. Counted in stats.

Output format (one JSON object per line):
  {
    "task":      "A" | "B",
    "src":       [...tokens...],         # Pseudo Piano Solo segment
    "tgt":       [...tokens...],         # Violin only (A) or violin+piano with track markers (B)
    "song":      "0/0/Qma....json",      # Path relative to TOKENS_DIR
    "bar_start": 4,
    "bar_end":   10,
  }
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from fractions import Fraction
from glob import glob

# ──────────────────────────── configuration ──────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
TOKENS_DIR  = os.path.join(SCRIPT_DIR, 'tokens_duet')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'pairs_duet.jsonl')

SEG_MIN    = 4
SEG_MAX    = 8
SEG_STRIDE = 2

TRACK_VIOLIN_TOKEN = '<track_violin>'
TRACK_PIANO_TOKEN  = '<track_piano>'

RANDOM_SEED = 42

# Force UTF-8 stdout so foreign-language paths don't crash Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ──────────────────────────── token helpers ──────────────────────────────────

PITCH_RE = re.compile(r'^note_([A-G])(##|#|bb|b)?(-?\d+)$')
PITCH_CLASS = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
ACC_OFFSET  = {None: 0, '': 0, '#': 1, '##': 2, 'b': -1, 'bb': -2}


def note_token_to_midi(token: str) -> int:
    """'note_C#4' -> 61, 'note_Bb3' -> 58. Returns 0 on parse failure (rare)."""
    m = PITCH_RE.match(token)
    if not m:
        return 0
    letter, acc, octv = m.group(1), m.group(2) or '', int(m.group(3))
    return 12 * (octv + 1) + PITCH_CLASS[letter] + ACC_OFFSET.get(acc, 0)


def len_token_to_fraction(token: str) -> Fraction:
    """'len_3/4' -> Fraction(3, 4); 'len_2' -> Fraction(2)."""
    return Fraction(token.replace('len_', '', 1))


def split_into_bars(tokens):
    """Split flat token list at 'bar' markers; returns list of per-bar lists
    (with 'bar' separators removed)."""
    bars, cur = [], []
    for t in tokens:
        if t == 'bar':
            if cur:
                bars.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        bars.append(cur)
    return bars


def bars_to_tokens(bars):
    """Inverse of split_into_bars: re-insert 'bar' markers in front of each bar."""
    out = []
    for b in bars:
        out.append('bar')
        out.extend(b)
    return out


# ──────────────────────────── event parsing ──────────────────────────────────

def parse_events(tokens):
    """Parse a bar's token list (no 'bar' separator) into (header, events).

    Each event is a 4-tuple (kind, pitches, length_tok, attrs):
        kind:       'note' or 'rest'
        pitches:    list of 'note_*' tokens (empty for rests)
        length_tok: the 'len_*' token (string)
        attrs:      list of post-length tokens (stem_*, beam_*, tie_*, ...)
                    consumed until the next 'note_*' or 'rest'

    'header' is everything before the first 'note_*' / 'rest' (clef, key, time, ...).
    Stops parsing on malformed input (an event without a following len_*).
    """
    header, events = [], []
    n = len(tokens)
    i = 0
    while i < n and not (tokens[i].startswith('note_') or tokens[i] == 'rest'):
        header.append(tokens[i])
        i += 1
    while i < n:
        if tokens[i] == 'rest':
            kind, pitches = 'rest', []
            i += 1
        else:
            kind, pitches = 'note', []
            while i < n and tokens[i].startswith('note_'):
                pitches.append(tokens[i])
                i += 1
        if i >= n or not tokens[i].startswith('len_'):
            break  # malformed: drop the trailing partial event
        length_tok = tokens[i]
        i += 1
        attrs = []
        while i < n and not (tokens[i].startswith('note_') or tokens[i] == 'rest'):
            attrs.append(tokens[i])
            i += 1
        events.append((kind, pitches, length_tok, attrs))
    return header, events


def events_to_tokens(header, events):
    """Inverse of parse_events. Pitches in each event are sorted ascending."""
    out = list(header)
    for kind, pitches, length_tok, attrs in events:
        if kind == 'rest':
            out.append('rest')
        else:
            for p in sorted(pitches, key=note_token_to_midi):
                out.append(p)
        out.append(length_tok)
        out.extend(attrs)
    return out


def split_piano_bar(bar_tokens):
    """Return (shared, R_tokens, L_tokens) for a single piano bar.

    Two-staff piano (from tokenize_duet.tokenize_single_part):
        [shared (key/time)] R [r_body] L [l_body]
    Single-staff piano (rare): no R/L markers; entire bar treated as R."""
    try:
        r_idx = bar_tokens.index('R')
    except ValueError:
        # Single-staff fallback: pull key/time as shared, rest as R, no L.
        shared = [t for t in bar_tokens if t.split('_', 1)[0] in ('time', 'key')]
        body   = [t for t in bar_tokens if t.split('_', 1)[0] not in ('time', 'key')]
        return shared, body, []
    shared = bar_tokens[:r_idx]
    try:
        l_idx = bar_tokens.index('L', r_idx + 1)
    except ValueError:
        return shared, bar_tokens[r_idx + 1:], []
    return shared, bar_tokens[r_idx + 1:l_idx], bar_tokens[l_idx + 1:]


# ──────────────────────── pseudo-solo synthesis ──────────────────────────────

def merge_violin_into_R(r_tokens, violin_bar_tokens):
    """Onset-aligned merge: for each piano-R event window, fold in any violin
    pitches sounding during that window as additional chord notes. Piano's
    rhythm is the master clock."""
    p_header, p_events = parse_events(r_tokens)
    _, v_events = parse_events(violin_bar_tokens)

    # Collect violin note intervals (skip rests; they contribute nothing).
    v_intervals = []
    v_onset = Fraction(0)
    for kind, pitches, length_tok, _ in v_events:
        v_dur = len_token_to_fraction(length_tok)
        if kind == 'note':
            v_intervals.append((v_onset, v_onset + v_dur, list(pitches)))
        v_onset += v_dur

    merged = []
    p_onset = Fraction(0)
    for kind, pitches, length_tok, attrs in p_events:
        p_dur = len_token_to_fraction(length_tok)
        win_start, win_end = p_onset, p_onset + p_dur
        added = []
        for vs, ve, vps in v_intervals:
            if vs < win_end and ve > win_start:
                for vp in vps:
                    if vp not in added and vp not in pitches:
                        added.append(vp)
        if added:
            new_kind = 'note'
            new_pitches = (list(pitches) if kind == 'note' else []) + added
        else:
            new_kind, new_pitches = kind, list(pitches)
        merged.append((new_kind, new_pitches, length_tok, attrs))
        p_onset = win_end

    return events_to_tokens(p_header, merged)


def make_pseudo_bar(piano_bar, violin_bar, stats):
    """Synthesize one pseudo-solo bar from one piano bar and one violin bar.

    Falls back to the original piano bar if the merge can't be done safely
    (multi-voice block in either hand, or parse failure)."""
    if '<voice>' in piano_bar or '<voice>' in violin_bar:
        stats['skipped_multivoice_bars'] += 1
        return list(piano_bar)
    try:
        shared, R, L = split_piano_bar(piano_bar)
        new_R = merge_violin_into_R(R, violin_bar)
    except Exception as e:
        stats['bar_merge_errors'] += 1
        stats[f'_err: {type(e).__name__}'] += 1
        return list(piano_bar)
    return list(shared) + ['R'] + new_R + ['L'] + list(L)


# ─────────────────────── segmentation & pair emission ────────────────────────

def generate_segment_spans(n_bars):
    """Same sliding-window logic as Phase 1.2 build_pairs.py:
    random segment length in [SEG_MIN, SEG_MAX], advance by SEG_STRIDE."""
    spans = []
    i = 0
    while i + SEG_MIN <= n_bars:
        seg_len = random.randint(SEG_MIN, min(SEG_MAX, n_bars - i))
        spans.append((i, i + seg_len))
        i += SEG_STRIDE
    return spans


def process_one(path, out_fh, stats):
    """Process one tokens_duet/*.json file; append segments to out_fh.
    Returns (n_A_emitted, n_B_emitted)."""
    try:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        piano_tokens  = d['piano']
        violin_tokens = d['violin']
    except Exception:
        stats['load_errors'] += 1
        return 0, 0

    piano_bars  = split_into_bars(piano_tokens)
    violin_bars = split_into_bars(violin_tokens)

    n_bars = min(len(piano_bars), len(violin_bars))
    if n_bars < SEG_MIN:
        stats['skipped_too_short'] += 1
        return 0, 0
    piano_bars  = piano_bars[:n_bars]
    violin_bars = violin_bars[:n_bars]

    pseudo_bars = [make_pseudo_bar(pb, vb, stats)
                   for pb, vb in zip(piano_bars, violin_bars)]

    song_rel = os.path.relpath(path, TOKENS_DIR).replace('\\', '/')

    n_A = n_B = 0
    for start, end in generate_segment_spans(n_bars):
        src_seg = bars_to_tokens(pseudo_bars[start:end])
        vio_seg = bars_to_tokens(violin_bars[start:end])
        pia_seg = bars_to_tokens(piano_bars[start:end])

        out_fh.write(json.dumps({
            'task':      'A',
            'src':       src_seg,
            'tgt':       vio_seg,
            'song':      song_rel,
            'bar_start': start,
            'bar_end':   end,
        }, ensure_ascii=False) + '\n')
        n_A += 1

        tgt_B = [TRACK_VIOLIN_TOKEN] + vio_seg + [TRACK_PIANO_TOKEN] + pia_seg
        out_fh.write(json.dumps({
            'task':      'B',
            'src':       src_seg,
            'tgt':       tgt_B,
            'song':      song_rel,
            'bar_start': start,
            'bar_end':   end,
        }, ensure_ascii=False) + '\n')
        n_B += 1

    return n_A, n_B


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only the first N token files (smoke test)')
    parser.add_argument('--out', type=str, default=OUTPUT_PATH,
                        help=f'Output JSONL path (default: {OUTPUT_PATH})')
    args = parser.parse_args()

    random.seed(RANDOM_SEED)

    print(f"Token source: {TOKENS_DIR}")
    print(f"Output:       {args.out}")
    if args.limit:
        print(f"LIMIT:        {args.limit} files (smoke test)")
    print()

    json_paths = sorted(glob(os.path.join(TOKENS_DIR, '**', '*.json'), recursive=True))
    if args.limit:
        json_paths = json_paths[:args.limit]
    print(f"Duet token files to process: {len(json_paths):,}\n")

    stats = Counter()
    n_A_total = n_B_total = 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as out_fh:
        for i, path in enumerate(json_paths):
            if i and i % 100 == 0:
                print(f"  [{i:>5}/{len(json_paths)}]  "
                      f"A={n_A_total:>7,}  B={n_B_total:>7,}  "
                      f"short={stats['skipped_too_short']}  "
                      f"voice_bars={stats['skipped_multivoice_bars']}  "
                      f"merge_err={stats['bar_merge_errors']}")
            n_A, n_B = process_one(path, out_fh, stats)
            n_A_total += n_A
            n_B_total += n_B

    print(f"\nDone.")
    print(f"  Dataset A segments:                 {n_A_total:,}")
    print(f"  Dataset B segments:                 {n_B_total:,}")
    print(f"  Total examples:                     {n_A_total + n_B_total:,}")
    print(f"  Files skipped (< {SEG_MIN} bars):           {stats['skipped_too_short']}")
    print(f"  Bars w/ multi-voice (kept as-is):   {stats['skipped_multivoice_bars']}")
    print(f"  Bar merge errors (fell back to R):  {stats['bar_merge_errors']}")
    print(f"  Load errors:                        {stats['load_errors']}")
    err_types = [(k, v) for k, v in stats.items() if k.startswith('_err: ')]
    if err_types:
        print(f"\n  Merge error breakdown:")
        for k, v in sorted(err_types, key=lambda x: -x[1]):
            print(f"    [{v:>4}] {k[6:]}")
    print(f"\n  Output: {args.out}")


if __name__ == '__main__':
    main()
