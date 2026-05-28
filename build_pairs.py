import os, json, csv, random
from itertools import combinations
from collections import defaultdict

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH    = os.path.join(SCRIPT_DIR, "PDMX.csv")
TOKENS_DIR  = os.path.join(SCRIPT_DIR, "tokens")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "data", "pairs.jsonl")

SEG_MIN           = 4    # min bars per segment
SEG_MAX           = 8    # max bars per segment
SEG_STRIDE        = 2    # sliding window stride in bars
BAR_TOLERANCE     = 0.15 # max allowed bar count difference ratio (relaxed from 0.1 to recover pickup-bar pairs)
DENSITY_RATIO_MAX = 3.0  # max ratio of note densities between paired scores
NGRAM_OVERLAP_MIN = 0.25 # min trigram melodic overlap (filters different songs with same key/time)

random.seed(42)

# ---------------------------------------------------------------------------
# Song name quality filters
# ---------------------------------------------------------------------------

_BLACKLIST_EXACT = {
    'misc', 'untitled', 'new score', 'test', 'unknown', 'na', '',
    'new composition', 'untitled score', 'my score',
}

_BLACKLIST_SUBSTR = [
    'abcm2ps', 'sample tune', 'sample3', 'staff break',
    'template', 'default',
]

# These are book/collection titles — different users upload different individual
# pieces under the same name, so pairing them is almost always wrong.
_COLLECTION_EXACT = {
    'for children sz.42',
    'for children sz. 42',
    'mikrokosmos',
}

_COLLECTION_SUBSTR = [
    'sz.42', 'sz. 42',
    'mikrokosmos',
]


def is_bad_song_name(name: str) -> bool:
    """Return True if the song name is a garbage label or known collection title."""
    n = name.strip().lower()
    if n in _BLACKLIST_EXACT:
        return True
    if any(s in n for s in _BLACKLIST_SUBSTR):
        return True
    if n in _COLLECTION_EXACT:
        return True
    if any(s in n for s in _COLLECTION_SUBSTR):
        return True
    return False


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def token_path_from_mxl(mxl_rel):
    """Convert CSV mxl path (./mxl/1/11/Qm....mxl) to tokens/ path."""
    rel = mxl_rel.lstrip('./')
    rel = rel.replace('mxl/', '', 1)
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


def get_key(bars):
    """Return the key token from the first bar that contains one, or None."""
    for bar in bars:
        for tok in bar:
            if tok.startswith('key_'):
                return tok
    return None


def get_time(bars):
    """Return the time signature token from the first bar that contains one, or None."""
    for bar in bars:
        for tok in bar:
            if tok.startswith('time_'):
                return tok
    return None


def note_density(bars):
    """Return average number of note tokens per bar."""
    if not bars:
        return 0.0
    total = sum(t.startswith('note_') for bar in bars for t in bar)
    return total / len(bars)


def _pitch_ngrams(bars, n=3, n_bars=4):
    """
    Return a set of pitch-letter n-grams from the first n_bars.
    Uses the right-hand melody line (or all notes if R marker absent).
    Order-sensitive: 'C-E-G' and 'G-E-C' are different n-grams.
    """
    notes = []
    for bar in bars[:n_bars]:
        hand = get_hand_tokens(bar, 'R') or bar
        notes += [t[5] for t in hand if t.startswith('note_')]
    if len(notes) < n:
        return set()
    return set(zip(*[notes[i:] for i in range(n)]))


def melodic_ngram_overlap(bars_a, bars_b, n=3, n_bars=4) -> float:
    """
    Jaccard similarity of pitch-letter trigram sets over the first n_bars.
    Returns a value in [0, 1]; higher means more melodic similarity.
    More reliable than letter-set Jaccard because it captures note ORDER,
    preventing two different songs in the same key from scoring 1.0.
    """
    a = _pitch_ngrams(bars_a, n, n_bars)
    b = _pitch_ngrams(bars_b, n, n_bars)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pairs_are_compatible(bars_a, bars_b):
    """
    Return True if two arrangements are likely to share the same musical content.
    Checks (in order of cost, cheapest first):
      1. Same key signature
      2. Same time signature
      3. Note density ratio within DENSITY_RATIO_MAX
      4. Melodic trigram overlap >= NGRAM_OVERLAP_MIN
    """
    key_a, key_b = get_key(bars_a), get_key(bars_b)
    if key_a is not None and key_b is not None and key_a != key_b:
        return False

    time_a, time_b = get_time(bars_a), get_time(bars_b)
    if time_a is not None and time_b is not None and time_a != time_b:
        return False

    d_a, d_b = note_density(bars_a), note_density(bars_b)
    if d_a > 0 and d_b > 0:
        ratio = max(d_a, d_b) / min(d_a, d_b)
        if ratio > DENSITY_RATIO_MAX:
            return False

    if melodic_ngram_overlap(bars_a, bars_b) < NGRAM_OVERLAP_MIN:
        return False

    return True


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

    print("Loading CSV and matching to token files...")
    song_to_scores = defaultdict(list)

    with open(CSV_PATH, encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            tracks = row['tracks'].strip()
            song   = row['song_name'].strip()
            if not song or song == 'NA':
                continue
            if is_bad_song_name(song):
                continue
            if not all(t == '0' for t in tracks.split('-')):
                continue
            token_path = token_path_from_mxl(row['mxl'].strip())
            if os.path.exists(token_path):
                song_to_scores[song].append(token_path)

    multi = {s: v for s, v in song_to_scores.items() if len(v) >= 2}
    print(f"Songs with 2+ tokenized piano arrangements: {len(multi)}")

    total_segments    = 0
    total_pairs       = 0
    skipped_same_lv   = 0
    skipped_bars      = 0
    skipped_compat    = 0
    level_counts      = defaultdict(int)

    with open(OUTPUT_PATH, 'w') as out:
        for i, (song, paths) in enumerate(multi.items()):
            if i % 1000 == 0:
                print(f"  [{i}/{len(multi)}]  segments so far: {total_segments}")

            scored = []
            for path in paths:
                try:
                    with open(path) as f:
                        tokens = json.load(f)
                    bars  = split_into_bars(tokens)
                    if len(bars) < SEG_MIN:
                        continue
                    level = assign_level(bars)
                    scored.append((path, level, bars))
                    level_counts[level] += 1
                except Exception:
                    continue

            if len(scored) < 2:
                continue

            for (path_a, lv_a, bars_a), (path_b, lv_b, bars_b) in combinations(scored, 2):

                if lv_a == lv_b:
                    skipped_same_lv += 1
                    continue

                na, nb = len(bars_a), len(bars_b)
                if abs(na - nb) / max(na, nb) > BAR_TOLERANCE:
                    skipped_bars += 1
                    continue

                if not pairs_are_compatible(bars_a, bars_b):
                    skipped_compat += 1
                    continue

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
    print(f"Total training segments:                    {total_segments}")
    print(f"Total directional pairs:                    {total_pairs}")
    print(f"Skipped (same difficulty level):            {skipped_same_lv}")
    print(f"Skipped (bar count mismatch >{BAR_TOLERANCE*100:.0f}%): {skipped_bars}")
    print(f"Skipped (key/time/density/ngram mismatch):  {skipped_compat}")
    print(f"Level distribution: {dict(sorted(level_counts.items()))}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()