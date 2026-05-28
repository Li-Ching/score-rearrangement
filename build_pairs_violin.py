import os, json, random
from itertools import combinations
from collections import defaultdict
from fractions import Fraction
import glob

VIOLIN_TOKENS_DIR = "../violin_tokens"
OUTPUT_PATH       = "../data/violin_pairs.jsonl"

SEG_MIN           = 4
SEG_MAX           = 8
SEG_STRIDE        = 2
BAR_TOLERANCE     = 0.1
DENSITY_RATIO_MAX = 3.0   # 跟學姊一樣，避免配對的兩首 note density 差太多

random.seed(42)


# ── Token helpers（跟 build_pairs.py 一樣）───────────────────────────────────

def split_into_bars(tokens):
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
    result = []
    for bar in bars:
        result.append('bar')
        result.extend(bar)
    return result


def note_density(bars):
    """Average number of note tokens per bar."""
    if not bars:
        return 0.0
    total = sum(t.startswith('note_') for bar in bars for t in bar)
    return total / len(bars)


def generate_segments(src_bars, tgt_bars, src_level, tgt_level, src_path, tgt_path):
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
            'src_path':   src_path,
            'tgt_path':   tgt_path,
        })
        i += SEG_STRIDE
    return segments


# ── Violin difficulty（跟 score_to_tokens.py 的 assign_violin_difficulty 一致）

NOTE_ORDER = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def pitch_token_to_midi(token):
    if not token.startswith('note_'):
        return None
    name = token[5:]
    if len(name) < 2:
        return None
    try:
        octave = int(name[-1])
        pitch  = name[:-1]
        flat_map = {'Bb':'A#','Eb':'D#','Ab':'G#','Db':'C#',
                    'Gb':'F#','Cb':'B','Fb':'E'}
        pitch = flat_map.get(pitch, pitch)
        if pitch not in NOTE_ORDER:
            return None
        return octave * 12 + NOTE_ORDER.index(pitch)
    except (ValueError, IndexError):
        return None


def assign_level_violin(bars):
    """
    Violin difficulty based on three metrics (paper Section 5.1.1 equivalents):
      note_density      : average notes per bar
      pitch_width       : semitone range across all bars
      rhythm_complexity : shortest note value present

    Uses max() across three metrics so a piece hard in ANY dimension
    is rated at the higher level overall.

    Thresholds calibrated to ABRSM violin grade descriptors:
      Lv.1 (Grade 1-2) : slow, small range, simple rhythms
      Lv.2 (Grade 3-4) : moderate tempo, wider range
      Lv.3 (Grade 5-6) : busier, third position range
      Lv.4 (Grade 7-8) : fast, wide range, advanced
    """
    bar_note_counts = []
    pitches         = []
    shortest        = Fraction(1)

    for bar in bars:
        count = 0
        for t in bar:
            if t.startswith('note_'):
                count += 1
                midi = pitch_token_to_midi(t)
                if midi is not None:
                    pitches.append(midi)
            elif t.startswith('len_'):
                try:
                    frac = Fraction(t[4:])
                    if frac < shortest:
                        shortest = frac
                except (ValueError, ZeroDivisionError):
                    pass
        if count > 0:
            bar_note_counts.append(count)

    note_dens  = sum(bar_note_counts) / max(len(bar_note_counts), 1)
    pitch_width = (max(pitches) - min(pitches)) if len(pitches) >= 2 else 0

    def density_level(d):
        if d <= 4:  return 1
        if d <= 7:  return 2
        if d <= 11: return 3
        return 4

    def width_level(w):
        if w <= 12: return 1
        if w <= 19: return 2
        if w <= 26: return 3
        return 4

    def rhythm_level(r):
        if r >= Fraction(1, 4):  return 1
        if r >= Fraction(1, 8):  return 2
        if r >= Fraction(1, 16): return 3
        return 4

    level = max(density_level(note_dens),
                width_level(pitch_width),
                rhythm_level(shortest))
    return f'Lv.{level}', note_dens   # 回傳 level 和 density，density 給 filter 用


# ── Compatibility filter（只用 note density，不用 key/time）─────────────────
# 學姊的三個過濾（key, time, density）是為了確認「同名歌曲真的是同一首」
# 小提琴是 non-parallel 配對，本來就是不同曲子，所以：
#   key signature  → 不過濾（不同曲子本來就可能不同調）
#   time signature → 不過濾（不同曲子本來就可能不同拍號）
#   note density   → 過濾（避免配對差異太極端，模型難以學習）

def density_compatible(density_a, density_b):
    if density_a > 0 and density_b > 0:
        ratio = max(density_a, density_b) / min(density_a, density_b)
        return ratio <= DENSITY_RATIO_MAX
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Step 1: 讀取所有 violin token 檔，計算難度，按難度分群
    print("Loading violin token files and assigning difficulty levels...")
    json_files = glob.glob(
        os.path.join(VIOLIN_TOKENS_DIR, '**', '*.json'), recursive=True
    )
    print(f"Found {len(json_files)} violin token files.")

    # level_groups: { 'Lv.1': [(path, bars, density), ...], ... }
    level_groups = defaultdict(list)
    level_counts = defaultdict(int)
    skip_short   = 0
    skip_error   = 0

    for path in json_files:
        try:
            with open(path) as f:
                tokens = json.load(f)
            bars = split_into_bars(tokens)
            if len(bars) < SEG_MIN:
                skip_short += 1
                continue
            level, density = assign_level_violin(bars)
            level_groups[level].append((path, bars, density))
            level_counts[level] += 1
        except Exception:
            skip_error += 1

    print(f"\nDifficulty distribution:")
    for lv in ['Lv.1', 'Lv.2', 'Lv.3', 'Lv.4']:
        print(f"  {lv} : {level_counts[lv]:,} scores")
    print(f"  Skipped (too short) : {skip_short}")
    print(f"  Skipped (error)     : {skip_error}")

    # Step 2: 配對不同難度的曲子（non-parallel）
    # 跟鋼琴的差異：鋼琴用 song_name 配對同曲，小提琴直接跨難度配對
    print("\nBuilding cross-difficulty pairs...")

    total_segments    = 0
    total_pairs       = 0
    skip_bar_mismatch = 0
    skip_density      = 0

    levels = ['Lv.1', 'Lv.2', 'Lv.3', 'Lv.4']

    with open(OUTPUT_PATH, 'w') as out:
        for lv_a, lv_b in combinations(levels, 2):
            group_a = level_groups[lv_a]
            group_b = level_groups[lv_b]

            if not group_a or not group_b:
                continue

            print(f"  Pairing {lv_a} ({len(group_a)}) x {lv_b} ({len(group_b)}) ...")

            n_pairs    = min(len(group_a), len(group_b))
            sampled_a  = random.sample(group_a, n_pairs)
            sampled_b  = random.sample(group_b, n_pairs)

            for (path_a, bars_a, dens_a), (path_b, bars_b, dens_b) in zip(sampled_a, sampled_b):

                # filter 1: bar count mismatch（跟學姊一樣）
                na, nb = len(bars_a), len(bars_b)
                if abs(na - nb) / max(na, nb) > BAR_TOLERANCE:
                    skip_bar_mismatch += 1
                    continue

                # filter 2: note density ratio（跟學姊一樣的邏輯，但不過濾 key/time）
                if not density_compatible(dens_a, dens_b):
                    skip_density += 1
                    continue

                # 雙向：a→b 和 b→a
                for sp, tp, sl, tl, sb, tb in [
                    (path_a, path_b, lv_a, lv_b, bars_a, bars_b),
                    (path_b, path_a, lv_b, lv_a, bars_b, bars_a),
                ]:
                    segs = generate_segments(sb, tb, sl, tl, sp, tp)
                    for seg in segs:
                        out.write(json.dumps(seg) + '\n')
                    total_segments += len(segs)
                    total_pairs    += 1

    print(f"\nDone.")
    print(f"Total training segments              : {total_segments:,}")
    print(f"Total directional pairs              : {total_pairs:,}")
    print(f"Skipped (bar count mismatch >10%)    : {skip_bar_mismatch:,}")
    print(f"Skipped (note density ratio >{DENSITY_RATIO_MAX}) : {skip_density:,}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()