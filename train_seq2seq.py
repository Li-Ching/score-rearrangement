"""
train_seq2seq.py — Training script for Piano Score Rearrangement

Usage (defaults are for the paper's ~0.3 M model):
    python train_seq2seq.py
    python train_seq2seq.py --batch_size 64 --epochs 50
    python train_seq2seq.py --resume data/checkpoints/best.pt

Checkpoints are saved to --out_dir (default: data/checkpoints/):
    best.pt            — lowest validation loss seen so far
    epoch_NNNN.pt      — periodic snapshot every --save_every epochs

A training log (train_log.csv) is written to --out_dir.
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

from model import build_model
from dataset_seq2seq import make_collate_fn, make_splits


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def make_lr_lambda(warmup_steps: int, total_steps: int, peak_lr: float, min_lr: float):
    """Returns a LambdaLR multiplier function (relative to peak_lr)."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr + (peak_lr - min_lr) * cosine) / peak_lr
    return lr_lambda


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    pad_id: int,
    vocab_size: int,
    grad_clip: float,
    label_smoothing: float,
) -> float:
    """Run one full training pass. Returns mean per-token loss."""
    model.train()
    total_loss   = 0.0
    total_tokens = 0

    pbar = tqdm(loader, desc='  train', leave=False, unit='batch')
    for src, tgt_in, tgt_out, src_mask, tgt_mask in pbar:
        src     = src.to(device)
        tgt_in  = tgt_in.to(device)
        tgt_out = tgt_out.to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)

        logits = model(src, tgt_in, src_mask, tgt_mask)   # (B, T, V)

        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            tgt_out.reshape(-1),
            ignore_index=pad_id,
            label_smoothing=label_smoothing,
            reduction='sum',
        )
        n_tokens = (tgt_out != pad_id).sum().item()

        optimizer.zero_grad()
        (loss / n_tokens).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss   += loss.item()
        total_tokens += n_tokens

        pbar.set_postfix(loss=f'{total_loss / total_tokens:.4f}',
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
    """Run one full validation pass. Returns mean per-token loss."""
    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    for src, tgt_in, tgt_out, src_mask, tgt_mask in loader:
        src     = src.to(device)
        tgt_in  = tgt_in.to(device)
        tgt_out = tgt_out.to(device)
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

def save_checkpoint(path: str, model, optimizer, scheduler, epoch: int,
                    val_loss: float, args: argparse.Namespace) -> None:
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss':             val_loss,
        'args':                 vars(args),
    }, path)


def load_checkpoint(path: str, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt['epoch'], ckpt['val_loss']


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Train the seq2seq piano rearrangement model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument('--pairs',      default='data/pairs.jsonl',    help='training pairs file')
    p.add_argument('--vocab',      default='data/vocab.json',     help='vocabulary file')
    p.add_argument('--out_dir',    default='data/checkpoints',    help='checkpoint output directory')
    p.add_argument('--val_ratio',  type=float, default=0.05,      help='fraction of songs held out for validation')
    p.add_argument('--seed',       type=int,   default=42)

    # Training
    p.add_argument('--epochs',     type=int,   default=100)
    p.add_argument('--batch_size', type=int,   default=128)
    p.add_argument('--no_augment', action='store_true',           help='disable pitch augmentation')

    # Optimizer / LR
    p.add_argument('--lr',              type=float, default=1e-3,  help='peak learning rate')
    p.add_argument('--min_lr',          type=float, default=1e-5,  help='minimum LR after cosine decay')
    p.add_argument('--warmup_steps',    type=int,   default=1000,  help='linear warmup steps')
    p.add_argument('--grad_clip',       type=float, default=1.0,   help='gradient norm clip')
    p.add_argument('--label_smoothing', type=float, default=0.1,   help='cross-entropy label smoothing')

    # Stopping / saving
    p.add_argument('--patience',    type=int, default=10,  help='early stopping patience (epochs)')
    p.add_argument('--save_every',  type=int, default=10,  help='save periodic checkpoint every N epochs')

    # Resume
    p.add_argument('--resume', default=None, metavar='CKPT',
                   help='path to a checkpoint to resume training from')

    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.out_dir, exist_ok=True)

    # ── vocab ──────────────────────────────────────────────────────────────
    with open(args.vocab, encoding='utf-8') as f:
        vocab_data = json.load(f)
    vocab_size = len(vocab_data['token_to_id'])
    pad_id     = vocab_data['token_to_id']['<pad>']

    # ── datasets & loaders ────────────────────────────────────────────────
    augment = not args.no_augment
    train_ds, val_ds = make_splits(args.pairs, args.vocab, args.val_ratio, args.seed)

    # Patch augment flag on the underlying ScorePairDataset
    # (make_splits wraps them in Subset, so we reach .dataset)
    train_ds.dataset.augment = augment

    collate_fn  = make_collate_fn(pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=(device.type == 'cuda'),
    )

    # ── model ─────────────────────────────────────────────────────────────
    model = build_model(vocab_size, pad_id).to(device)

    # ── optimizer & scheduler ─────────────────────────────────────────────
    optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    scheduler   = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        make_lr_lambda(args.warmup_steps, total_steps, args.lr, args.min_lr),
    )

    # ── resume ────────────────────────────────────────────────────────────
    start_epoch      = 0
    best_val_loss    = float('inf')
    patience_counter = 0

    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        start_epoch += 1
        print(f'Resumed from {args.resume} (epoch {start_epoch}, best val {best_val_loss:.4f})')

    # ── info ──────────────────────────────────────────────────────────────
    print(f'Device          : {device}')
    print(f'Parameters      : {model.count_parameters():,}')
    print(f'Vocab size      : {vocab_size}')
    print(f'Train pairs     : {len(train_ds):,}   Val pairs: {len(val_ds):,}')
    print(f'Steps per epoch : {len(train_loader)}   Total steps: {total_steps}')
    print(f'Augmentation    : {augment}')
    print()

    # ── CSV log ───────────────────────────────────────────────────────────
    log_path = os.path.join(args.out_dir, 'train_log.csv')
    log_exists = os.path.exists(log_path)
    log_file   = open(log_path, 'a', newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr', 'elapsed_s'])

    # ── training loop ─────────────────────────────────────────────────────
    try:
        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()

            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, device,
                pad_id, vocab_size, args.grad_clip, args.label_smoothing,
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
                    print(f'Early stopping.')
                    break

            # Periodic checkpoint
            if (epoch + 1) % args.save_every == 0:
                periodic_path = os.path.join(args.out_dir, f'epoch_{epoch + 1:04d}.pt')
                save_checkpoint(periodic_path, model, optimizer, scheduler, epoch, val_loss, args)

    finally:
        log_file.close()

    print(f'\nDone. Best val loss: {best_val_loss:.4f}')
    print(f'Best model: {os.path.join(args.out_dir, "best.pt")}')


if __name__ == '__main__':
    main()
