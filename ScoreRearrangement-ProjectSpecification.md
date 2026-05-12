# Piano Score Rearrangement — Project Specification

---

## 1. Project Overview

This project implements an end-to-end piano score rearrangement system that transforms a piano score into a target difficulty level (Beginner / Elementary / Intermediate / Advanced).

The approach is based on the paper:
> "Piano Score Rearrangement into Multiple Difficulty Levels via Notation-to-Notation Approach"
> Masahiro Suzuki, EURASIP Journal on Audio, Speech, and Music Processing, 2023.

The system operates entirely at the notation level (musical symbols, articulations, beams, ties) rather than the note/MIDI level, preserving musical expressiveness. It uses the ST+ token representation and a sequence-to-sequence (seq2seq) Transformer model conditioned on difficulty level tokens.

---

## 2. Dataset

- **Source:** PDMX (Piano Data from MuseScore eXchange)
- **Total piano scores:** ~205,789 (filtered from 254,035 total by MIDI program = 0)
- **Songs with 2+ arrangements:** ~31,875 (used as training pairs)

Difficulty labels are **not pre-tagged** in PDMX and must be computed from token features:

| Level | Name | Definition |
|---|---|---|
| Level 1 | Beginner | Max 1 simultaneous note per hand |
| Level 2 | Elementary | Max 2 simultaneous notes per hand |
| Level 3 | Intermediate | Max 3 simultaneous notes per hand |
| Level 4 | Advanced | No restriction |

Supporting metrics (also computed from tokens):
- **Note density:** number of notes per measure
- **Pitch width:** semitone range (highest – lowest pitch) per measure
- **Polyphony:** max simultaneous notes per measure

---

## 2.1 Cross-Instrument Extension: Data Analysis

We investigated the feasibility of extending this system to cross-instrument translation (piano ↔ violin) with combined difficulty transformation. The following data was found in PDMX:

| | Count |
|---|---|
| Songs with 2+ piano arrangements | 31,875 |
| Songs with violin arrangements | 542 |
| Songs with BOTH piano AND violin | 141 |
| Total piano+violin cross pairs | 1,381 |

141 songs is far too little to train a cross-instrument model. For reference, the paper trained on 1,957 scores and got 130,930 segment pairs. With only 1,381 cross pairs, you would get approximately 10,000–20,000 segments after chunking — likely not enough for the model to generalize.

**Decision: implement piano-only first, for the following reasons:**

1. **Data is solid** — 31,875 multi-arrangement songs gives plenty of training pairs.
2. **Validates the pipeline** — confirms the full stack works before tackling a harder problem.
3. **Almost no rework** — when adding cross-instrument later, only `tokenize_all.py`, `build_pairs.py`, and conditioning tokens in `model.py` need changes. The rest stays the same.
4. **Cross-instrument needs more data** — additional violin+piano paired data from other sources (e.g., IMSLP) would be needed before expanding.

The cross-instrument extension is planned as **Phase 6** (see Section 5).

---

## 3. System Architecture

The full pipeline:

```
[Input MXL]
   » score_to_tokens.py   ——  tokenize to ST+ format
   » prepend {Dsrc, Dtgt}  ——  difficulty conditioning tokens
   » seq2seq model          ——  encoder-decoder Transformer
   » strip conditioning     ——  prepend Dtgt only
   » tokens_to_score.py   ——  detokenize to music21 Score
[Output MXL]
```

**Difficulty Conditioning (from paper Fig. 2b):**
- Source sequence: `Dsrc Dtgt bar key_flat_1 time_4/4 R ...`
- Target sequence: `Dtgt bar key_flat_1 time_4/4 R ...`

Score pairs are trained **bidirectionally** (easier→harder and harder→easier), and all nC2 combinations of available arrangements per song are used as training pairs.

---

## 4. Model

| Property | Value |
|---|---|
| Architecture | Encoder-Decoder Transformer |
| Model size | ~0.3M parameters (small, matching paper config) |
| Embedding dim | 48 |
| FFN dim | 96 |
| Layers | 3 encoder + 3 decoder |
| Seq length | 4–8 measure segments (overlapping) |
| Augmentation | Pitch transposition ±2 semitones (training only) |

**Vocabulary:** ST+ tokens (`bar`, `R`, `L`, `clef_*`, `key_*`, `time_*`, `note_*`, `len_*`, `stem_*`, `beam_*`, `tie_*`, `rest`, `accent`, `staccato`, `tenuto`, `slur_start`, `slur_stop`, `chord_*`, `bass_*`, `<voice>`, `</voice>`) + special tokens (`<pad>`, `<sos>`, `<eos>`, `Lv.1`, `Lv.2`, `Lv.3`, `Lv.4`)

---

## 5. Project Breakdown

### Phase 1 — Data Preparation

**[1.1] `tokenize_all.py`**
- Filter PDMX to piano-only scores (program=0) via PDMX.csv
- Run `MusicXML_to_tokens()` on all MXL files
- Save token sequences as JSON under `tokens/`
- **Status: DONE**

**[1.2] `build_pairs.py`**
- Compute difficulty metrics (polyphony, note density, pitch width) from token files
- Assign Lv.1–Lv.4 labels per score
- Match same-song scores using `song_name` column in PDMX.csv
- Generate all nC2 bidirectional pairs
- Segment pairs into 4–8-bar chunks with overlap:
  - **Breaks long songs into short chunks** — the model only processes 4–8 bars at a time, keeping sequences short enough for the small (0.3M) model to handle efficiently.
  - **Overlapping windows multiply training data** — one song pair generates ~28 segments instead of 1, helping the model generalize despite having limited songs.

  ```
  Input song (60 bars)
      ↓ split into segments
  [bar1-6]   → model → [bar1-6 at Lv.1]
  [bar7-12]  → model → [bar7-12 at Lv.1]
  [bar13-18] → model → [bar13-18 at Lv.1]
      ↓ stitch back together
  Output song (60 bars, Lv.1)
  ```

- Save as `pairs.jsonl`
- **Status: DONE**

**[1.3] `build_vocab.py`**
- Scan all token files to collect unique tokens
- Add special tokens: `<pad>`, `<sos>`, `<eos>`, `Lv.1`, `Lv.2`, `Lv.3`, `Lv.4`
- Save `vocab.json` (token → index mapping)
- Token strings need to encode to indexs for feeding to model
- **Status: DONE**

---

### Phase 2 — Model Implementation

**[2.1] `model.py`**
- Encoder-Decoder Transformer (PyTorch)
- Shared token embedding for source and target
- Difficulty conditioning via prepended `Lv.*` tokens
- **Status: TODO**

**[2.2] `dataset_seq2seq.py`**
- PyTorch Dataset that loads `pairs.jsonl`
- Encodes tokens using `vocab.json`
- Applies pitch augmentation (±2 semitones)
- Pads and batches source/target sequences
- **Status: TODO**

---

### Phase 3 — Training

**[3.1] `train_seq2seq.py`**
- Training loop with Adam optimizer + LR warmup/decay
- Teacher forcing on target sequence
- Validation loss tracking, early stopping
- Checkpoint saving (best model by validation loss)
- **Status: TODO**

---

### Phase 4 — Inference

**[4.1] `infer.py`**
- Load trained model checkpoint
- Accept input MXL + target difficulty level
- Tokenize – prepend conditioning – run model – detokenize
- Output rearranged MXL file
- **Status: TODO**

---

### Phase 5 — Evaluation

**[5.1] `evaluate.py`**
- Compute note density, pitch width, polyphony for generated vs. reference
- Jensen-Shannon divergence between generated and human-level distributions
- Syntax error rate and structure error rate
- **Status: TODO**

---

### Phase 6 — Cross-Instrument Extension (Piano ↔ Violin)

**[6.1] Expand `tokenize_all.py`**
- Also tokenize violin scores (MIDI program = 40) from PDMX
- Handle single-staff (R only) format for violin
- **Status: TODO** (pending Phase 1–5 completion)

**[6.2] Expand `build_pairs.py`**
- Match same-song piano+violin pairs (141 songs / 1,381 pairs in PDMX — insufficient alone)
- Supplement with external datasets (e.g., IMSLP) for more paired data
- **Status: TODO** (requires additional data sourcing)

**[6.3] Update `model.py` conditioning tokens**
- Change conditioning from `{Dsrc, Dtgt}` to `{Isrc_Dsrc, Itgt_Dtgt}`
- e.g., `piano_Lv3 violin_Lv1` as combined instrument+difficulty tokens
- **Status: TODO**

---

## 6. File Structure

```
score-rearrangement/
    mxl/                         raw MusicXML files (PDMX)
    tokens/                      tokenized JSON files (output of 1.1)
    data/
        pairs.jsonl              training pairs (output of 1.2)
        vocab.json               vocabulary (output of 1.3)
        checkpoints/             saved model weights
    score_to_tokens.py           MXL → ST+ tokens
    tokens_to_score.py           ST+ tokens → MXL
    tokenize_all.py              batch tokenization [Phase 1.1]
    build_pairs.py               pair generation + difficulty labeling [Phase 1.2]
    build_vocab.py               vocabulary builder [Phase 1.3]
    model.py                     seq2seq Transformer [Phase 2.1]
    dataset_seq2seq.py           PyTorch Dataset [Phase 2.2]
    train_seq2seq.py             training script [Phase 3.1]
    infer.py                     inference script [Phase 4.1]
    evaluate.py                  evaluation metrics [Phase 5.1]
    PDMX.csv                     PDMX metadata
    Score Rearrangement-Project Specification.md
```

---

## 7. Key References

1. Suzuki, M. (2023). Piano score rearrangement into multiple difficulty levels via notation-to-notation approach. *EURASIP Journal on Audio, Speech, and Music Processing.* https://doi.org/10.1186/s13636-023-00321-7

2. ScoreRearrangement GitHub (ST+ tokenization tools): https://github.com/suzuqn/ScoreRearrangement

3. PDMX Dataset: Piano Data from MuseScore eXchange
