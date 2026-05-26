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

### Data quality findings (after initial training)

PDMX "same song" pairs are user-uploaded arrangements that share a song title but may not share the same melody, key, or structure. This caused the first-round model to output piano music that sounded like a different piece entirely rather than a transformed version of the input.

**Compatibility filter added to `build_pairs.py`:**
- Same key signature required
- Same time signature required
- Note density ratio ≤ 3× (one arrangement cannot have 3× more notes/bar than the other)

This filter removes pairs that share only a title. Fewer but higher-quality pairs are expected to produce better melody-preserving transformations.

---

## 2.1 Cross-Instrument Extension: Data Analysis

We investigated the feasibility of extending this system to cross-instrument translation (piano ↔ violin) with combined difficulty transformation. The following data was found in PDMX:

| | Count |
|---|---|
| Songs with 2+ piano arrangements | 31,875 |
| Songs with purely violin-only arrangements | 434 |
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
   » strip conditioning     ——  remove Dtgt prefix from output
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

- Compatibility filters (key/time/density matching):
  - Same key signature — different keys almost certainly means different arrangements
  - Same time signature — structurally incompatible otherwise
  - Note density ratio ≤ 3× — if one arrangement has far more notes, they are likely unrelated pieces
- Save as `pairs.jsonl`
- **Status: DONE**

**[1.3] `build_vocab.py`**
- Scan all token files to collect unique tokens
- Add special tokens: `<pad>`, `<sos>`, `<eos>`, `Lv.1`, `Lv.2`, `Lv.3`, `Lv.4`
- Save `vocab.json` (token → index mapping)
- Token strings need to encode to indexes for feeding to model
- **Status: DONE**

---

### Phase 2 — Model Implementation

**[2.1] `model.py`**
- Encoder-Decoder Transformer (PyTorch)
- Shared token embedding for source and target
- Difficulty conditioning via prepended `Lv.*` tokens
- Key design decisions:
    - Shared embedding — single nn.Embedding for both encoder and decoder inputs
    - Difficulty conditioning — handled entirely by prepended Lv.* tokens in the sequence; no special architecture needed inside the model
    - `_bool_to_additive` — converts bool padding masks to float additive masks (-inf) so they're consistent with the float causal mask PyTorch generates, avoiding deprecation warnings
    - `forward()` — teacher-forced training path (src + tgt-shifted-right → logits)
    - `encode()` / `decode_step()` — separated for autoregressive inference
    - `greedy_decode()` — batched decoding used by `infer.py`:
        - `init_token_idx` — forces Dtgt as the first decoder output token (prevents the model from predicting the wrong difficulty level)
        - `temperature` — softmax temperature for sampling (>1 adds variety, <1 sharpens)
        - `top_k` — if >0, samples from top-k logits instead of argmax, breaking repetitive collapse
- **Status: DONE**

**[2.2] `dataset_seq2seq.py`**
- PyTorch Dataset that loads `pairs.jsonl`
- Encodes tokens using `vocab.json`
- Applies pitch augmentation (±2 semitones)
- Pads and batches source/target sequences
- `transpose_tokens(tokens, shift)`
    - Transposes all pitch-bearing tokens by shift semitones:
        - `note_*` — MIDI number ± shift, back to letter name
        - `key_*` — circle-of-fifths shift (e.g. G major +1 → Ab major)
        - `bass_*` / `chord_*` — pitch-class rotation, quality unchanged
- `ScorePairDataset`
    - Loads all pairs from pairs.jsonl
    - Builds encoder/decoder sequences per paper Fig. 2b: `src = [Dsrc, Dtgt, …, <eos>]`, `tgt = [<sos>, Dtgt, …, <eos>]`
    - On each `__getitem__` randomly samples a shift from {-2,-1,0,1,2}
- `make_collate_fn(pad_id)`
    - Returns a collate function that pads and splits the target into:
        - `tgt_in = tgt[:-1]` (decoder input, teacher-forced)
        - `tgt_out = tgt[1:]` (cross-entropy target)
- `make_splits(pairs_path, vocab_path)`
    - Song-level train/val split so no song leaks across splits (default 5% val)
- **Status: DONE**

---

### Phase 3 — Training

**[3.1] `train_seq2seq.py`**
- Training loop with Adam optimizer + LR warmup/decay
- Teacher forcing on target sequence
- Validation loss tracking, early stopping
- Checkpoint saving (best model by validation loss)
- LR schedule — `make_lr_lambda`: linear warmup over `--warmup_steps` steps, then cosine decay to `--min_lr`
- Training loop
    - Teacher forcing: `tgt_in = tgt[:-1]` → model → compared against `tgt_out = tgt[1:]`
    - `F.cross_entropy` with `ignore_index=pad_id` (pad positions don't contribute to loss)
    - `label_smoothing=0.1` (helps regularization on a small dataset)
    - Gradient clipping at `grad_clip=1.0`
    - Gradient accumulation (`--accum_steps`, default 4) — effective batch = batch_size × accum_steps without extra VRAM
    - Per-batch tqdm bar showing running loss + current LR
- Checkpointing
    - `best.pt` — saved whenever val loss improves (stores model, optimizer, scheduler state for resuming)
    - `epoch_NNNN.pt` — periodic snapshot every `--save_every` epochs (default 10)
    - `train_log.csv` — append-only CSV with epoch, train loss, val loss, lr, elapsed time
- Early stopping — stops after `--patience` consecutive epochs (default 10) with no val improvement
- Resuming — `--resume data/checkpoints/best.pt` restores full state and continues from next epoch

**Round 1 training results (noisy pairs):**
- Trained 60 epochs, val_loss = 1.24, LR decayed to minimum (1e-5)
- Output was musically valid in structure but did not sound like the same song as the input
- Root cause: training pairs matched by song title only, not musical content

**Round 2 plan (after reprocessing with compatibility filter):**
- Run `python build_pairs.py` to regenerate `pairs.jsonl` with key/time/density filter
- Train from scratch: `python train_seq2seq.py --epochs 100 --lr 1e-3`
- Training from scratch (not resuming) is preferred since the data distribution changes significantly

- **Status: Round 1 DONE, Round 2 pending dataset reprocessing**

---

### Phase 4 — Inference

**[4.1] `infer.py`**
- Load trained model checkpoint
- Accept input MXL + target difficulty level
- Tokenize – prepend conditioning – run model – detokenize
- Output rearranged MXL file

- Pipeline:
  ```
  Input MXL
     ↓ MusicXML_to_tokens()     tokenize to ST+ format
     ↓ split_into_bars()        split into per-bar lists
     ↓ assign_level()           detect source difficulty (Lv.1–4)
     ↓ non-overlapping chunks   --seg_len bars each (default 8)
     ↓ encode_segment()         prepend [Dsrc, Dtgt, ..., <eos>]
     ↓ model.greedy_decode()    autoregressive generation
     ↓ strip Dtgt token         remove conditioning prefix from output
     ↓ concatenate segments     stitch all bars back together
     ↓ tokens_to_score()        music21 Score
     ↓ score.write('musicxml')  output .mxl
  Output MXL
  ```

- Key arguments:
  ```
  --input       Input MXL or XML file (required)
  --output      Output MXL file (required)
  --level       Target difficulty: Lv.1 / Lv.2 / Lv.3 / Lv.4 (required)
  --checkpoint  Model checkpoint (default: data/checkpoints/best.pt)
  --seg_len     Bars per segment (default: 8, range: 4–8)
  --temperature Sampling temperature (default: 1.2; higher = more varied)
  --top_k       Top-k sampling (default: 10; 0 = greedy argmax)
  --device      Device override (default: auto-detect CUDA)
  ```

- Inference improvements over naive greedy decode:
    - **Forced Dtgt** (`init_token_idx`) — the target level token is injected directly into the decoder prefix, preventing the model from predicting the wrong difficulty level as its first token
    - **Top-k sampling** (`top_k=10, temperature=1.2`) — breaks the greedy repetition collapse where the model would output the same note hundreds of times

- Usage:
  ```bash
  python infer.py --input mxl/X/XX/Qm....mxl --output output.mxl --level Lv.1
  python infer.py --input mxl/X/XX/Qm....mxl --output output.mxl --level Lv.1 --temperature 0.8 --top_k 5
  ```

- **Status: DONE**

---

### Phase 5 — Evaluation

**[5.1] `evaluate.py`**
- Compute note density, pitch width, polyphony for generated vs. reference
- Jensen-Shannon divergence between generated and human-level distributions
- Syntax error rate and structure error rate
- **Status: TODO**

---

### Phase 6 — AI Auto-Orchestration & Cross-Instrument Extension (Piano ➔ Duet)

**Motivation & data-source change vs. Section 2.1.**
Section 2.1 concluded that cross-instrument training was infeasible because only **141 songs** in PDMX have both a piano-only arrangement and a violin-only arrangement (1,381 cross-pairs total). Phase 6 sidesteps that bottleneck by **changing the data source**: instead of pairing two separately-uploaded arrangements, we use PDMX's existing **piano + violin duet scores** (MIDI program 0 + 40 inside a single score) and **synthesize the source side ourselves** via reverse augmentation. This converts the problem from "find paired data" to "split data we already have", and unlocks a much larger pool of scores.

**[6.1] Duet Data Processing & Reverse Augmentation (`tokenize_duet.py`)**

- Filter PDMX to scores whose `tracks` column contains both `0` (piano) and `40` (violin). Verified against `PDMX.csv` (254,077 rows total): **1,890 scores match**.
- Composition of those 1,890 scores (important — they are mostly NOT pure duets):

  | Class | Count | Share |
  |---|---:|---:|
  | Contains other instruments (e.g. `0-40-41-42`, chamber / small orchestra) | 1,508 | 79.8% |
  | Pure piano + violin (`0-40` exactly) | 315 | 16.7% |
  | Piano / violin doublings only (e.g. `0-40-40`, `0-0-40`) | 67 | 3.5% |

  Top co-occurring programs alongside piano+violin: viola (41), cello (42), flute (73), trumpet (56) — meaning many of the 1,890 are actually string quartets or small-ensemble arrangements rather than literal duets.

- **Decision: use all 1,890 scores (Option B), discard non-0 / non-40 tracks at preprocessing time.**
  - For each score, pick the **first** program-0 part as the piano, and the **first** program-40 part as the violin melody.
  - For `0-40-40` (two violins): use Violin 1 only. Merging multiple violins into one melody line is left as a v2 extension — the first iteration keeps the task definition clean ("expand piano solo into piano + one violin melody").
  - For `0-0-40` and similar (multiple pianos): use the first piano part only.
  - **Drop** all other programs (41, 42, 73, …) entirely — do NOT fold viola/cello into the piano, since that would make the synthesized Pseudo Piano Solo unplayable and worsen the train/test distribution gap.
  - Trade-off acknowledged: dropping viola/cello means we lose harmonic information present in the original chamber score, but the goal here is to learn piano↔violin orchestration, not full chamber reduction.
- Run `MusicXML_to_tokens()` per track to isolate **[Violin Melody]** and **[Piano Accompaniment]** as separate token streams.
- **Reverse Data Augmentation** — synthesize a **[Pseudo Piano Solo]** by merging violin melody tokens into the piano accompaniment track:
  - Align by bar/onset, merge violin pitches into the piano right hand as additional chord notes (or as a separate voice, then flatten).
  - **Normalization step (critical to close the train/test distribution gap):** after merging, re-run hand assignment, re-collapse simultaneous notes into chords, and strip duet-only voice markers so the pseudo solo looks like a token sequence a real piano-solo score would produce. Without this, the model overfits to the synthetic "violin stacked on piano" pattern and fails on real piano-solo inputs at inference time.
- Output: per-score `{piano, violin, n_bars, tracks}` token JSONs under `Phase06/tokens_duet/`.
- Pseudo-solo synthesis is **deferred to Phase 6.2** (`build_pairs_duet.py`), so we can iterate on the merging strategy without re-tokenizing 1,890 MXLs.
- **Actual results (run 2026-05-18):** 1,735 / 1,890 scores tokenized successfully (91.8%). 65.1 MB total output. 151,475 bars across all duets; median 64 bars/score. 155 failures dominated by upstream `score_to_tokens.aggregate_notes()` raising `Cannot insert None into a tag` on edge-case notations — not worth chasing for first iteration.
- **Status:** DONE (selective extraction + per-track tokenization). Code: `Phase06/extract_duet_mxl.py`, `Phase06/tokenize_duet.py`.

**[6.2] Duet Pair Building (`build_pairs_duet.py`)**

Build training pairs from the synthesized data. Kept as a separate script from `build_pairs.py` to avoid breaking the Phase 1–5 piano-only pipeline.

- **Dataset A — Melody Extraction:** `[Pseudo Piano Solo] ➔ [Violin Melody]`
- **Dataset B — Auto-Orchestration:** `[Pseudo Piano Solo] ➔ [Original Duet (Violin + Piano)]`
- Reuse the 4–8-bar sliding-window segmentation from Phase 1.2 (the alignment is exact here, so no key/time/density compatibility filter is needed).
- Pseudo Piano Solo synthesis: onset-aligned chord merge — for each piano-R event window, fold any violin pitches sounding during that window in as additional chord notes (deduplicated). Piano's rhythm is the master clock; rests in piano R get upgraded to notes if violin is sounding. Bars containing `<voice>` markers (multi-voice within one hand) keep the piano R unchanged, since onset math is ambiguous across concurrent voices.
- Dataset B target encodes the duet as a single token stream delimited by `<track_violin>` and `<track_piano>` (the same delimiters Phase 6.3 will add to the vocabulary).
- Output: `Phase06/pairs_duet.jsonl` with each segment tagged by which task (A or B) it belongs to.
- **Actual results (run 2026-05-23):** 73,366 Dataset A + 73,366 Dataset B = 146,732 segments from 1,733 / 1,735 scores (99.9%). 13,910 bars (≈9.2% of total) had multi-voice blocks and kept original piano R unchanged — those bars' Dataset A loses some violin signal, Dataset B unaffected. 2 scores skipped (< 4 bars). Zero merge / load errors. Total examples exceed the paper's 130,930 reference.
- **Status:** DONE. Code: `Phase06/build_pairs_duet.py`. Output: `Phase06/pairs_duet.jsonl`.

**[6.3] Vocabulary & Model Update (`build_vocab.py`, `model.py`)**

- Extend `vocab.json` with:
  - Track tokens: `<track_piano>`, `<track_violin>` (delimit per-track regions inside the target sequence for multi-track output).
  - Task tokens: `<task_melody>`, `<task_duet>` (prepended to the source, analogous to existing `Lv.*` conditioning — lets one model handle both Dataset A and Dataset B).
- Single multi-task seq2seq Transformer (not two separate models): same architecture as Phase 2, just a larger vocab and longer max sequence length (Dataset B targets are ~2× the length of Dataset A's).
- Train from scratch with `train_seq2seq.py` on `pairs_duet.jsonl` (no fine-tuning from the Phase 3 checkpoint — the input distribution is too different).
- **Status:** TODO

**[6.4] Duet Inference (`infer_duet.py`)**

- Input: a real piano-solo MXL + a task flag (`--task melody` or `--task duet`).
- Pipeline mirrors Phase 4.1, with two changes:
  - Prepend `<task_*>` token instead of (or in addition to) `Lv.*`.
  - When detokenizing Dataset B output, split the token stream on `<track_*>` boundaries and emit a multi-staff MusicXML score (piano + violin) instead of a single piano part.
- **Status:** TODO

**[6.5] Duet Evaluation (`evaluate_duet.py`)**

- **Melody extraction (Dataset A):** note-level precision / recall / F1 of extracted violin against the held-out original violin part.
- **Auto-orchestration (Dataset B):** reuse Phase 5 distributional metrics (note density, pitch width, polyphony, JS divergence) on each track separately; also report inter-track interaction metrics (rhythmic alignment, harmonic consistency between violin & piano).
- **Status:** TODO

---

## 6. File Structure

```
score-rearrangement/
    mxl/                         raw MusicXML files (PDMX)
    tokens/                      tokenized JSON files (output of 1.1)
    data/
        pairs.jsonl              training pairs (output of 1.2)
        score_list.csv           per-score difficulty table (output of list_scores.py)
        vocab.json               vocabulary (output of 1.3)
        checkpoints/             saved model weights
            best.pt              best checkpoint by val loss
            epoch_NNNN.pt        periodic snapshots
            train_log.csv        epoch-by-epoch training log
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
    list_scores.py               generates score_list.csv for test score selection
    PDMX.csv                     PDMX metadata
    ScoreRearrangement-ProjectSpecification.md
```

---

## 7. Key References

1. Suzuki, M. (2023). Piano score rearrangement into multiple difficulty levels via notation-to-notation approach. *EURASIP Journal on Audio, Speech, and Music Processing.* https://doi.org/10.1186/s13636-023-00321-7

2. ScoreRearrangement GitHub (ST+ tokenization tools): https://github.com/suzuqn/ScoreRearrangement

3. PDMX Dataset: Piano Data from MuseScore eXchange
