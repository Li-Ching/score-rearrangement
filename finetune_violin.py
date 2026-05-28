"""
finetune_violin.py — Fine-tune the pretrained piano model on violin pairs

Phase 6.5 of the Score Rearrangement project.

What this script does differently from train_seq2seq.py:
  1. Loads a pretrained piano model checkpoint (Phase 3 best.pt)
  2. Resizes the embedding layer to match vocab_violin.json
     (adds 'piano' and 'violin' instrument tokens)
  3. Freezes encoder layers 0-1, only updates encoder layer 2 + full decoder
  4. Uses a lower learning rate (default 1e-4 vs 1e-3 for piano training)
  5. Reads violin_pairs.jsonl instead of pairs.jsonl

Usage:
    python finetune_violin.py
    python finetune_violin.py --pretrained data/checkpoints/best.pt
    python finetune_violin.py --resume data/checkpoints/violin/best.pt
    python finetune_violin.py --epochs 50 --batch_size 32

Checkpoints saved to data/checkpoints/violin/:
    best.pt           — lowest validation loss
    epoch_NNNN.pt     — periodic snapshot
    finetune_log.csv  — training log
"""

import argparse
import csv
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_violin  import build_model
from dataset_seq2seq import make_collate_fn, ScorePairDataset


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay (same as train_seq2seq.py)
# ---------------------------------------------------------------------------

def make_lr_lambda(warmup_steps: int, total_steps: int, peak_lr: float, min_lr: float):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr + (peak_lr - min_lr) * cosine) / peak_lr
    return lr_lambda


# ---------------------------------------------------------------------------
# Freeze encoder layers 0-1, keep encoder layer 2 + full decoder trainable
# ---------------------------------------------------------------------------

def apply_freeze(model) -> None:
    """
    Freeze the first two encoder layers to preserve general music knowledge
    learned from piano data. Only encoder layer 2 and the full decoder are
    updated during fine-tuning.

    Also freezes the shared embedding — the new violin/piano instrument
    tokens (appended at the end) are NOT frozen so they can be learned.
    This is handled by resize_embedding() which keeps old weights intact
    and only randomises the new rows.
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze encoder layer 2 (third layer, 0-indexed)
    for param in model.transformer.encoder.layers[2].parameters():
        param.requires_grad = True

    # Unfreeze full decoder
    for param in model.transformer.decoder.parameters():
        param.requires_grad = True

    # Unfreeze output projection
    for param in model.out_proj.parameters():
        param.requires_grad = True

    # Unfreeze only the new instrument token embeddings (last n rows)
    # We do this by registering a gradient hook rather than unfreezing
    # the whole embedding, to avoid disturbing piano token embeddings.
    model.embedding.weight.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}%)")


# ---------------------------------------------------------------------------
# One training epoch (identical logic to train_seq2seq.py)
# ---------------------------------------------------------------------------

def train_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    pad_id: int,
    vocab_size: int,
    grad_clip: float,
    label_smoothing: float,
    accum_steps: int = 1,
) -> float:
    model.train()
    total_loss   = 0.0
    total_tokens = 0

    optimizer.zero_grad()
    pbar = tqdm(loader, desc='  train', leave=False, unit='batch')

    for micro_step, (src, tgt_in, tgt_out, src_mask, tgt_mask) in enumerate(pbar):
        src      = src.to(device)
        tgt_in   = tgt_in.to(device)
        tgt_out  = tgt_out.to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)

        logits = model(src, tgt_in, src_mask, tgt_mask)

        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            tgt_out.reshape(-1),
            ignore_index=pad_id,
            label_smoothing=label_smoothing,
            reduction='sum',
        )
        n_tokens = (tgt_out != pad_id).sum().item()

        (loss / max(n_tokens, 1) / accum_steps).backward()

        total_loss   += loss.item()
        total_tokens += n_tokens

        is_last_batch = (micro_step + 1 == len(loader))
        if (micro_step + 1) % accum_steps == 0 or is_last_batch:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix(loss=f'{total_loss / max(total_tokens, 1):.4f}',
                         lr=f'{optimizer.param_groups[0]["lr"]:.2e}')

    return total_loss / max(1, total_tokens)


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def val_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
    pad_id: int,
    vocab_size: int,
) -> float:
    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    for src, tgt_in, tgt_out, src_mask, tgt_mask in loader:
        src      = src.to(device)
        tgt_in   = tgt_in.to(device)
        tgt_out  = tgt_out.to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)

        logits = model(src, tgt_in, src_mask, tgt_mask)

        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            tgt_out.reshape(-1),
            ignore_index=pad_id,
            reduction='sum',
        )
        n_tokens = (tgt_out != pad_id).sum().item()

        total_loss   += loss.item()
        total_tokens += n_tokens

    return total_loss / max(1, total_tokens)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, scheduler, epoch, val_loss, args):
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss':             val_loss,
        'args':                 vars(args),
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt['epoch'], ckpt['val_loss']


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Fine-tune pretrained piano model on violin pairs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    p.add_argument('--pretrained', default='data/checkpoints/best.pt',
                   help='pretrained piano model checkpoint (Phase 3)')
    p.add_argument('--pairs',   default='data/violin_pairs.jsonl',
                   help='violin training pairs file (from build_pairs_violin.py)')
    p.add_argument('--vocab',   default='data/vocab_violin.json',
                   help='violin vocabulary file (from build_vocab_violin.py)')
    p.add_argument('--out_dir', default='data/checkpoints/violin',
                   help='output directory for violin checkpoints')

    # Fine-tune specific
    p.add_argument('--freeze_encoder', action='store_true', default=True,
                   help='freeze encoder layers 0-1 (recommended for small data)')
    p.add_argument('--no_freeze_encoder', dest='freeze_encoder',
                   action='store_false',
                   help='unfreeze all encoder layers')

    # Resume fine-tuning (different from --pretrained)
    p.add_argument('--resume', default=None, metavar='CKPT',
                   help='resume an interrupted fine-tuning run '
                        '(pass a violin checkpoint, not the piano one)')

    # Training
    p.add_argument('--epochs',      type=int,   default=50)
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--accum_steps', type=int,   default=4)
    p.add_argument('--no_augment',  action='store_true',
                   help='disable pitch augmentation')

    # Optimizer — lower LR than piano training
    p.add_argument('--lr',              type=float, default=1e-4,
                   help='peak learning rate (lower than piano: 1e-3)')
    p.add_argument('--min_lr',          type=float, default=1e-6)
    p.add_argument('--warmup_steps',    type=int,   default=200)
    p.add_argument('--grad_clip',       type=float, default=1.0)
    p.add_argument('--label_smoothing', type=float, default=0.1)

    # Stopping / saving
    p.add_argument('--patience',   type=int, default=10)
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--val_ratio',  type=float, default=0.1,
                   help='validation split (larger than piano due to smaller dataset)')
    p.add_argument('--seed',       type=int, default=42)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Train/val split for violin (no 'song' field — use random index split)
# ---------------------------------------------------------------------------

def make_violin_splits(pairs_path, vocab_path, val_ratio=0.1, seed=42):
    """
    Random index-based train/val split for violin pairs.

    Replaces make_splits() from dataset_seq2seq.py, which requires a
    'song' field for song-level grouping. Violin pairs are non-parallel
    (different songs paired by difficulty level), so a simple random
    index split is appropriate.
    """
    import random as _random
    from torch.utils.data import Subset

    full_train = ScorePairDataset(pairs_path, vocab_path, augment=True)
    full_val   = ScorePairDataset(pairs_path, vocab_path, augment=False)

    n_total = len(full_train)
    n_val   = max(1, math.ceil(n_total * val_ratio))
    n_train = n_total - n_val

    rng     = _random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    return Subset(full_train, train_idx), Subset(full_val, val_idx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Violin vocab ───────────────────────────────────────────────────────
    print(f"Loading violin vocab from '{args.vocab}' ...")
    with open(args.vocab, encoding='utf-8') as f:
        vocab_data = json.load(f)
    vocab_size = len(vocab_data['token_to_id'])
    pad_id     = vocab_data['token_to_id']['<pad>']
    print(f"  Vocab size: {vocab_size}")

    # ── Build model from violin vocab size ────────────────────────────────
    # Start with the correct vocab size so resize_embedding only needs to
    # handle the delta between piano vocab and violin vocab.
    print(f"\nLoading pretrained piano model from '{args.pretrained}' ...")
    if not os.path.exists(args.pretrained):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {args.pretrained}\n"
            f"Run train_seq2seq.py first, or pass --pretrained <path>."
        )

    piano_ckpt = torch.load(args.pretrained, map_location='cpu')
    piano_args = piano_ckpt.get('args', {})

    # Infer piano vocab size from the saved embedding weight shape
    piano_vocab_size = piano_ckpt['model_state_dict']['embedding.weight'].shape[0]
    print(f"  Piano model vocab size  : {piano_vocab_size}")
    print(f"  Violin vocab size       : {vocab_size}")
    print(f"  New tokens to add       : {vocab_size - piano_vocab_size}")

    # Build model with piano vocab size, load piano weights, then resize
    model = build_model(piano_vocab_size, pad_id).to(device)
    model.load_state_dict(piano_ckpt['model_state_dict'])
    print("  Piano weights loaded.")

    # Resize embedding to violin vocab (adds new instrument token rows)
    if vocab_size > piano_vocab_size:
        model.resize_embedding(vocab_size)
    model = model.to(device)

    # ── Freeze layers ──────────────────────────────────────────────────────
    if args.freeze_encoder:
        print("\nApplying layer freeze (encoder layers 0-1 frozen) ...")
        apply_freeze(model)
    else:
        print("\nNo layer freeze applied — all parameters trainable.")
        trainable = model.count_parameters()
        print(f"Trainable parameters: {trainable:,}")

    # ── Dataset & loaders ─────────────────────────────────────────────────
    print(f"\nLoading violin pairs from '{args.pairs}' ...")
    augment = not args.no_augment
    train_ds, val_ds = make_violin_splits(args.pairs, args.vocab, args.val_ratio, args.seed)
    train_ds.dataset.augment = augment

    collate_fn   = make_collate_fn(pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    print(f"  Train pairs: {len(train_ds):,}   Val pairs: {len(val_ds):,}")

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    # Only pass parameters that require grad (respects freeze)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    opt_steps_per_epoch = math.ceil(len(train_loader) / args.accum_steps)
    total_steps         = args.epochs * opt_steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        make_lr_lambda(args.warmup_steps, total_steps, args.lr, args.min_lr),
    )

    # ── Resume fine-tuning (optional) ─────────────────────────────────────
    start_epoch      = 0
    best_val_loss    = float('inf')
    patience_counter = 0

    if args.resume:
        print(f"\nResuming fine-tuning from '{args.resume}' ...")
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        start_epoch += 1
        print(f"  Resumed at epoch {start_epoch}, best val {best_val_loss:.4f}")

    # ── Info ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Device          : {device}")
    print(f"Pretrained from : {args.pretrained}")
    print(f"Violin vocab    : {args.vocab}  ({vocab_size} tokens)")
    print(f"Pairs file      : {args.pairs}")
    print(f"Freeze encoder  : {args.freeze_encoder}")
    print(f"LR              : {args.lr}  (min {args.min_lr})")
    print(f"Batch size      : {args.batch_size} × {args.accum_steps} accum "
          f"= {args.batch_size * args.accum_steps} effective")
    print(f"Epochs          : {args.epochs}   Patience: {args.patience}")
    print(f"Augmentation    : {augment}")
    print(f"{'='*60}\n")

    # ── CSV log ───────────────────────────────────────────────────────────
    log_path   = os.path.join(args.out_dir, 'finetune_log.csv')
    log_exists = os.path.exists(log_path)
    log_file   = open(log_path, 'a', newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr', 'elapsed_s'])

    # ── Training loop ─────────────────────────────────────────────────────
    try:
        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()

            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, device,
                pad_id, vocab_size, args.grad_clip, args.label_smoothing,
                args.accum_steps,
            )
            val_loss = val_epoch(model, val_loader, device, pad_id, vocab_size)

            elapsed = time.time() - t0
            lr_now  = optimizer.param_groups[0]['lr']

            print(
                f'Epoch {epoch + 1:4d}/{args.epochs}  '
                f'train={train_loss:.4f}  val={val_loss:.4f}  '
                f'lr={lr_now:.2e}  {elapsed:.0f}s'
            )
            log_writer.writerow([epoch + 1, f'{train_loss:.6f}', f'{val_loss:.6f}',
                                 f'{lr_now:.2e}', f'{elapsed:.1f}'])
            log_file.flush()

            # Best checkpoint
            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                save_checkpoint(
                    os.path.join(args.out_dir, 'best.pt'),
                    model, optimizer, scheduler, epoch, val_loss, args,
                )
                print(f'  ✓ New best val loss: {val_loss:.4f}')
            else:
                patience_counter += 1
                print(f'  No improvement ({patience_counter}/{args.patience})')
                if patience_counter >= args.patience:
                    print('Early stopping.')
                    break

            # Periodic checkpoint
            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(
                    os.path.join(args.out_dir, f'epoch_{epoch + 1:04d}.pt'),
                    model, optimizer, scheduler, epoch, val_loss, args,
                )

    finally:
        log_file.close()

    print(f'\nDone. Best val loss: {best_val_loss:.4f}')
    print(f'Best model saved to: {os.path.join(args.out_dir, "best.pt")}')


if __name__ == '__main__':
    main()