import os, json, csv, random
from itertools import combinations
from collections import defaultdict

CSV_PATH    = "/Users/yan/Documents/GitHub/score-rearrangement/PDMX.csv"
TOKENS_DIR  = "/Users/yan/Documents/GitHub/score-rearrangement/tokens"
OUTPUT_PATH = "/Users/yan/Documents/GitHub/score-rearrangement/data/pairs.jsonl"

SEG_MIN       = 4    # min bars per segment
SEG_MAX       = 8    # max bars per segment
SEG_STRIDE    = 2    # sliding window stride in bars
BAR_TOLERANCE = 0.1  # max allowed bar count difference ratio between paired scores

random.seed(42)


def token_path_from_mxl(mxl_rel):
    """Convert CSV mxl path (./mxl/1/11/Qm....mxl) to tokens/ path."""
    rel = mxl_rel.lstrip('./')              # mxl/1/11/Qm....mxl
    rel = rel.replace('mxl/', '', 1)        # 1/11/Qm....mxl
    rel = rel.replace('.mxl', '.json').replace('.xml', '.json')
    return os.path.join(TOKENS_DIR, rel)


def split_into_bars(tokens):
    """Split flat token list into list of per-bar token lists (excluding 'bar' separators)."""
    bars, current = [], []
    for t in tokens:
        if t == 'bar':
            if current:
                bars.append(current)
            current = []
        else:
            current.append(t)
    if current:
        bars.append(current)
    return bars


def bars_to_tokens(bars):
    """Reconstruct flat token list from list of bar token lists."""
    result = []
    for bar in bars:
        result.append('bar')
        result.extend(bar)
    return result


def get_hand_tokens(bar_tokens, hand):
    """
    Extract tokens belonging to one hand (R or L) from a single bar's tokens.
    Bar format: [shared_tokens] R [right_tokens] L [left_tokens]
    """
    try:
        idx = bar_tokens.index(hand)
    except ValueError:
        return []
    result, i = [], idx + 1
    while i < len(bar_tokens) and bar_tokens[i] not in ('R', 'L'):
        result.append(bar_tokens[i])
        i += 1
    return result


def max_chord_size(tokens):
    """
    Return max number of simultaneous notes (chord polyphony) in a token sequence.
    Consecutive note_* tokens before a len_* token form a chord.
    """
    max_poly, current = 0, 0
    for t in tokens:
        if t.startswith('note_'):
            current += 1
            max_poly = max(max_poly, current)
        elif t.startswith('len_') or t == 'rest':
            current = 0
    return max_poly


def assign_level(bars):
    """
    Assign difficulty level Lv.1-4 based on max polyphony per hand across all bars.
    Matches the paper's definition (Section 4.1):
      Lv.1 Beginner:      max poly <= 1 (one note per hand)
      Lv.2 Elementary:    max poly <= 2 (up to two simultaneous notes)
      Lv.3 Intermediate:  max poly <= 3 (up to three simultaneous notes)
      Lv.4 Advanced:      max poly >  3 (no restriction)
    """
    max_r, max_l = 0, 0
    for bar in bars:
        max_r = max(max_r, max_chord_size(get_hand_tokens(bar, 'R')))
        max_l = max(max_l, max_chord_size(get_hand_tokens(bar, 'L')))
    poly = max(max_r, max_l)
    if poly <= 1:
        return 'Lv.1'
    elif poly <= 2:
        return 'Lv.2'
    elif poly <= 3:
        return 'Lv.3'
    else:
        return 'Lv.4'


def generate_segments(src_bars, tgt_bars, src_level, tgt_level, song, src_path, tgt_path):
    """
    Generate overlapping segment pairs using a sliding window (stride = SEG_STRIDE).
    Segment length is randomly chosen between SEG_MIN and SEG_MAX bars.
    Source and target bars are sliced at the same positions to keep alignment.
    """
    n = min(len(src_bars), len(tgt_bars))
    segments = []
    i = 0
    while i + SEG_MIN <= n:
        seg_len = random.randint(SEG_MIN, min(SEG_MAX, n - i))
        segments.append({
            'src_tokens': bars_to_tokens(src_bars[i:i + seg_len]),
            'tgt_tokens': bars_to_tokens(tgt_bars[i:i + seg_len]),
            'src_level':  src_level,
            'tgt_level':  tgt_level,
            'song':       song,
            'src_path':   src_path,
            'tgt_path':   tgt_path,
        })
        i += SEG_STRIDE
    return segments


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Step 1: collect tokenized piano scores grouped by song name
    print("Loading CSV and matching to token files...")
    song_to_scores = defaultdict(list)  # song_name -> [token_path, ...]

    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            tracks = row['tracks'].strip()
            song   = row['song_name'].strip()
            if not song or song == 'NA':
                continue
            if not all(t == '0' for t in tracks.split('-')):
                continue
            token_path = token_path_from_mxl(row['mxl'].strip())
            if os.path.exists(token_path):
                song_to_scores[song].append(token_path)

    multi = {s: v for s, v in song_to_scores.items() if len(v) >= 2}
    print(f"Songs with 2+ tokenized piano arrangements: {len(multi)}")

    # Step 2: compute difficulty level for each arrangement, build pairs, segment
    total_segments  = 0
    total_pairs     = 0
    skipped_same_lv = 0
    skipped_bars    = 0
    level_counts    = defaultdict(int)

    with open(OUTPUT_PATH, 'w') as out:
        for i, (song, paths) in enumerate(multi.items()):
            if i % 1000 == 0:
                print(f"  [{i}/{len(multi)}]  segments so far: {total_segments}")

            # load tokens and assign difficulty level for each arrangement
            scored = []
            for path in paths:
                try:
                    with open(path) as f:
                        tokens = json.load(f)
                    bars  = split_into_bars(tokens)
                    if len(bars) < SEG_MIN:
                        continue  # too short to segment
                    level = assign_level(bars)
                    scored.append((path, level, bars))
                    level_counts[level] += 1
                except Exception:
                    continue

            if len(scored) < 2:
                continue

            # generate all nC2 pairs, trained bidirectionally (paper Section 3.1.1)
            for (path_a, lv_a, bars_a), (path_b, lv_b, bars_b) in combinations(scored, 2):

                if lv_a == lv_b:
                    skipped_same_lv += 1
                    continue  # no difficulty transformation — skip

                # require bar counts to be within tolerance for clean alignment
                na, nb = len(bars_a), len(bars_b)
                if abs(na - nb) / max(na, nb) > BAR_TOLERANCE:
                    skipped_bars += 1
                    continue

                # both directions: a->b and b->a
                for sp, tp, sl, tl, sb, tb in [
                    (path_a, path_b, lv_a, lv_b, bars_a, bars_b),
                    (path_b, path_a, lv_b, lv_a, bars_b, bars_a),
                ]:
                    segs = generate_segments(sb, tb, sl, tl, song, sp, tp)
                    for seg in segs:
                        out.write(json.dumps(seg) + '\n')
                    total_segments += len(segs)
                    total_pairs    += 1

    print(f"\nDone.")
    print(f"Total training segments:              {total_segments}")
    print(f"Total directional pairs:              {total_pairs}")
    print(f"Skipped (same difficulty level):      {skipped_same_lv}")
    print(f"Skipped (bar count mismatch >10%):    {skipped_bars}")
    print(f"Level distribution: {dict(sorted(level_counts.items()))}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
