import os, json, csv, random
from itertools import combinations
from collections import defaultdict

CSV_PATH    = r".\PDMX.csv"
TOKENS_DIR  = r".\tokens"
OUTPUT_PATH = r".\data\pairs.jsonl"

SEG_MIN           = 4    # min bars per segment
SEG_MAX           = 8    # max bars per segment
SEG_STRIDE        = 2    # sliding window stride in bars
BAR_TOLERANCE     = 0.05  # tightened from 0.1 → 0.05 (same song should have similar length)
DENSITY_RATIO_MAX = 3.0   # max ratio of note densities between paired scores
SKYLINE_THRESHOLD = 0.6   # min melody skyline similarity to accept a pair as same song

random.seed(42)


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
    Assign difficulty level Lv.1-4 using three combined metrics.

    The original polyphony-only definition (paper Section 4.1) works well for
    commercial pop piano scores but breaks down on PDMX, where the vast majority
    of scores are simple arrangements with polyphony <= 1.  We therefore use
    three complementary metrics and take the MAXIMUM level across all three,
    so a score is rated hard if it is hard in ANY dimension.

    Metrics (same as paper Section 5.1.1 evaluation metrics):
      note_density  : average notes per bar (both hands combined)
      pitch_width   : semitone range across entire score
      polyphony     : max simultaneous notes per hand (original metric, retained)

    Thresholds calibrated for PDMX piano scores:
      Lv.1 : simple melody, small range, sparse
      Lv.2 : moderate complexity
      Lv.3 : denser, wider range
      Lv.4 : complex, wide range, many notes
    """
    from fractions import Fraction

    # ── polyphony (original metric) ───────────────────────────────────────
    max_r, max_l = 0, 0
    for bar in bars:
        max_r = max(max_r, max_chord_size(get_hand_tokens(bar, 'R')))
        max_l = max(max_l, max_chord_size(get_hand_tokens(bar, 'L')))
    poly = max(max_r, max_l)

    # ── note density ──────────────────────────────────────────────────────
    density = note_density(bars)

    # ── pitch width ───────────────────────────────────────────────────────
    pitches = []
    for bar in bars:
        for tok in bar:
            midi = pitch_token_to_midi(tok)
            if midi is not None:
                pitches.append(midi)
    width = (max(pitches) - min(pitches)) if len(pitches) >= 2 else 0

    # ── per-metric level ──────────────────────────────────────────────────
    def poly_level(p):
        if p <= 1: return 1
        if p <= 2: return 2
        if p <= 3: return 3
        return 4

    def density_level(d):
        # thresholds based on actual PDMX quartiles (p25=8.4, p50=10.9, p75=14.0)
        if d <= 8.4:  return 1
        if d <= 10.9: return 2
        if d <= 14.0: return 3
        return 4

    def width_level(w):
        # thresholds based on actual PDMX quartiles (p25=16, p50=19, p75=24)
        if w <= 16: return 1
        if w <= 19: return 2
        if w <= 24: return 3
        return 4

    # Use median of three metrics instead of max().
    # max() causes a single high-range score to dominate regardless of density/poly.
    # Median is more robust: a score must be hard in at least 2 of 3 dimensions
    # to be rated hard overall.
    levels = sorted([poly_level(poly), density_level(density), width_level(width)])
    level  = levels[1]   # median of three values
    return f'Lv.{level}'


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


# ── Melody Skyline helpers ────────────────────────────────────────────────────

_NOTE_ORDER = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_FLAT_MAP   = {'Bb':'A#','Eb':'D#','Ab':'G#','Db':'C#','Gb':'F#','Cb':'B','Fb':'E'}

def pitch_token_to_midi(token):
    """
    Convert a note_* token (e.g. 'note_C4', 'note_F#5', 'note_Bb3')
    to a MIDI number. Returns None if the token cannot be parsed.
    """
    if not token.startswith('note_'):
        return None
    name = token[5:]
    if len(name) < 2:
        return None
    try:
        octave = int(name[-1])
        pitch  = name[:-1]
        pitch  = _FLAT_MAP.get(pitch, pitch)
        if pitch not in _NOTE_ORDER:
            return None
        return octave * 12 + _NOTE_ORDER.index(pitch)
    except (ValueError, IndexError):
        return None


def melody_skyline(bars):
    """
    Compute the melody skyline: for each bar, take the highest MIDI pitch
    in the right-hand (R) staff. Returns a list of int|None (None = silent bar).

    The skyline captures the melody contour, which is the strongest indicator
    that two arrangements are versions of the same song.
    """
    skyline = []
    for bar in bars:
        r_tokens = get_hand_tokens(bar, 'R')
        pitches  = [pitch_token_to_midi(t) for t in r_tokens if t.startswith('note_')]
        pitches  = [p for p in pitches if p is not None]
        skyline.append(max(pitches) if pitches else None)
    return skyline


def skyline_similarity(sky_a, sky_b):
    """
    Compare two melody skylines and return a similarity score [0.0, 1.0].

    Alignment: zip the two skylines bar-by-bar (works because BAR_TOLERANCE
    already ensures the two scores have nearly the same number of bars).

    A bar pair is counted as 'matching' if both have a pitch and the pitch
    difference is within ±2 semitones (allows for minor transposition errors
    or enharmonic respellings between arrangements).

    Returns 0.0 if there are fewer than 4 comparable bar pairs (too short
    to judge reliably).
    """
    pairs = [(a, b) for a, b in zip(sky_a, sky_b) if a is not None and b is not None]
    if len(pairs) < 4:
        return 0.0
    matches = sum(1 for a, b in pairs if abs(a - b) <= 2)
    return matches / len(pairs)


# ── Compatibility filter ──────────────────────────────────────────────────────

def pairs_are_compatible(bars_a, bars_b, sky_a, sky_b):
    """
    Return True if two arrangements are likely to be versions of the same song.

    Checks (in order of cost, cheapest first):
      1. Same key signature    — different keys almost certainly mean different songs
      2. Same time signature   — different meters mean structurally incompatible
      3. Note density ratio    — one arrangement shouldn't have 3× more notes/bar
      4. Melody skyline        — most expensive but strongest indicator; at least
                                 60% of bars should share the same top-note contour
    """
    # 1. Key signature
    key_a, key_b = get_key(bars_a), get_key(bars_b)
    if key_a is not None and key_b is not None and key_a != key_b:
        return False

    # 2. Time signature
    time_a, time_b = get_time(bars_a), get_time(bars_b)
    if time_a is not None and time_b is not None and time_a != time_b:
        return False

    # 3. Note density ratio
    d_a, d_b = note_density(bars_a), note_density(bars_b)
    if d_a > 0 and d_b > 0:
        ratio = max(d_a, d_b) / min(d_a, d_b)
        if ratio > DENSITY_RATIO_MAX:
            return False

    # 4. Melody skyline similarity (most discriminative filter)
    sim = skyline_similarity(sky_a, sky_b)
    if sim < SKYLINE_THRESHOLD:
        return False

    return True


def ensure_segment_clefs(bars):
    """
    Ensure the first bar of a segment has clef tokens for R and L staves.

    When build_pairs.py cuts a score into 4-8 bar segments using a sliding
    window, most segments start mid-score where there are no clef tokens
    (MusicXML only writes clefs when they change, so only the first bar of
    the whole score has them).

    Without clef tokens, tokens_to_score() has no information about which
    staff each hand belongs to, causing the model to learn wrong clef
    associations.

    Strategy:
      1. Search the whole segment for any existing clef tokens.
      2. If the first bar is missing a clef, inject the found clef
         (or the default: clef_treble for R, clef_bass for L).

    This modifies bars in-place and returns the same list.
    """
    # Search entire segment for clef tokens (in case they appear later)
    found_treble = None
    found_bass   = None
    for bar in bars:
        for tok in bar:
            if tok == 'clef_treble' and found_treble is None:
                found_treble = tok
            if tok == 'clef_bass' and found_bass is None:
                found_bass = tok
        if found_treble and found_bass:
            break

    # Inject into first bar if missing
    first_bar = bars[0]

    if 'R' in first_bar:
        r_idx    = first_bar.index('R')
        l_idx    = first_bar.index('L') if 'L' in first_bar else len(first_bar)
        r_section = first_bar[r_idx + 1: l_idx]
        if not any(t == 'clef_treble' for t in r_section):
            first_bar.insert(r_idx + 1, found_treble or 'clef_treble')
            # recalculate l_idx after insertion
            if 'L' in first_bar:
                l_idx = first_bar.index('L')

    if 'L' in first_bar:
        l_idx    = first_bar.index('L')
        l_section = first_bar[l_idx + 1:]
        if not any(t == 'clef_bass' for t in l_section):
            first_bar.insert(l_idx + 1, found_bass or 'clef_bass')

    return bars


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
        seg_len  = random.randint(SEG_MIN, min(SEG_MAX, n - i))
        src_seg  = ensure_segment_clefs([bar[:] for bar in src_bars[i:i + seg_len]])
        tgt_seg  = ensure_segment_clefs([bar[:] for bar in tgt_bars[i:i + seg_len]])
        segments.append({
            'src_tokens': bars_to_tokens(src_seg),
            'tgt_tokens': bars_to_tokens(tgt_seg),
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
    song_to_scores = defaultdict(list)

    with open(CSV_PATH, encoding='utf-8') as f:
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
    total_segments    = 0
    total_pairs       = 0
    skipped_same_lv   = 0
    skipped_bars      = 0
    skipped_compat    = 0
    skipped_skyline   = 0
    level_counts      = defaultdict(int)

    with open(OUTPUT_PATH, 'w') as out:
        for i, (song, paths) in enumerate(multi.items()):
            if i % 1000 == 0:
                print(f"  [{i}/{len(multi)}]  segments so far: {total_segments}")

            # load tokens, compute difficulty level and melody skyline
            scored = []
            for path in paths:
                try:
                    with open(path) as f:
                        tokens = json.load(f)
                    bars  = split_into_bars(tokens)
                    if len(bars) < SEG_MIN:
                        continue
                    level = assign_level(bars)
                    sky   = melody_skyline(bars)
                    scored.append((path, level, bars, sky))
                    level_counts[level] += 1
                except Exception:
                    continue

            if len(scored) < 2:
                continue

            for (path_a, lv_a, bars_a, sky_a), (path_b, lv_b, bars_b, sky_b) \
                    in combinations(scored, 2):

                if lv_a == lv_b:
                    skipped_same_lv += 1
                    continue

                # bar count tolerance (tightened to 5%)
                na, nb = len(bars_a), len(bars_b)
                if abs(na - nb) / max(na, nb) > BAR_TOLERANCE:
                    skipped_bars += 1
                    continue

                # key / time / density / skyline
                if not pairs_are_compatible(bars_a, bars_b, sky_a, sky_b):
                    skipped_compat += 1
                    continue

                # both directions: a→b and b→a
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
    print(f"Total training segments              : {total_segments:,}")
    print(f"Total directional pairs              : {total_pairs:,}")
    print(f"Skipped (same difficulty level)      : {skipped_same_lv:,}")
    print(f"Skipped (bar count mismatch >5%)     : {skipped_bars:,}")
    print(f"Skipped (key/time/density/skyline)   : {skipped_compat:,}")
    print(f"Level distribution: {dict(sorted(level_counts.items()))}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()