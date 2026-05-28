"""
model.py — Encoder-Decoder Transformer for Piano Score Rearrangement

Architecture based on:
  "Piano Score Rearrangement into Multiple Difficulty Levels via
   Notation-to-Notation Approach" (Suzuki, 2023)

Difficulty conditioning is done by prepending Lv.* tokens to the input
sequences before passing them to the model — no special conditioning
mechanism is needed inside the architecture itself.

Piano-only (Phase 1-5):
  Source sequence:  [Dsrc, Dtgt, bar, key_*, time_*, R/L, ...]
  Target sequence:  [Dtgt, bar, key_*, time_*, R/L, ...]

Violin fine-tune (Phase 6):
  Source sequence:  [Isrc, Dsrc, Itgt, Dtgt, bar, key_*, time_*, R, ...]
  Target sequence:  [Itgt, Dtgt, bar, key_*, time_*, R, ...]
  e.g.  violin Lv.3 violin Lv.1 bar key_flat_1 time_4/4 R ...
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bool_to_additive(mask: torch.Tensor | None) -> torch.Tensor | None:
    """Convert a bool padding mask (True = ignore) to a float additive mask
    (0.0 = attend, -inf = ignore). PyTorch Transformer internals expect both
    attn_mask and key_padding_mask to share the same dtype."""
    if mask is None:
        return None
    return mask.float().masked_fill(mask, float("-inf"))


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class ScoreRearrangementModel(nn.Module):
    """
    Encoder-Decoder Transformer for piano score rearrangement.

    Parameters
    ----------
    vocab_size : int
        Total vocabulary size (including special tokens).
    d_model : int
        Embedding / hidden dimension (paper: 48).
    nhead : int
        Number of attention heads. Must divide d_model evenly (paper uses 4).
    num_encoder_layers : int
        Number of encoder layers (paper: 3).
    num_decoder_layers : int
        Number of decoder layers (paper: 3).
    dim_feedforward : int
        Inner dimension of the position-wise FFN (paper: 96).
    dropout : float
        Dropout probability.
    max_seq_len : int
        Maximum sequence length for positional encoding.
    pad_idx : int
        Padding token index — embeddings at this index are zeroed out.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 48,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 96,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        pad_idx: int = 0,
    ):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.d_model = d_model
        self.pad_idx = pad_idx

        # Shared embedding used for both encoder input and decoder input
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection: maps decoder hidden states → vocabulary logits
        self.out_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.d_model ** -0.5)
        if self.pad_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.pad_idx].zero_()
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    # ------------------------------------------------------------------
    # Core forward pass (used during training with teacher forcing)
    # ------------------------------------------------------------------

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        src : (batch, src_len)  — encoder input token indices
        tgt : (batch, tgt_len)  — decoder input token indices (teacher-forced)
        src_key_padding_mask : (batch, src_len) bool, True = ignore position
        tgt_key_padding_mask : (batch, tgt_len) bool, True = ignore position

        Returns
        -------
        logits : (batch, tgt_len, vocab_size)
        """
        scale = math.sqrt(self.d_model)

        src_emb = self.pos_encoding(self.embedding(src) * scale)
        tgt_emb = self.pos_encoding(self.embedding(tgt) * scale)

        # Causal mask: float additive mask (-inf at future positions)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1), device=tgt.device
        )
        # Padding masks must be float (additive) to match the causal mask type
        src_pad_mask = _bool_to_additive(src_key_padding_mask)
        tgt_pad_mask = _bool_to_additive(tgt_key_padding_mask)

        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.out_proj(out)  # (batch, tgt_len, vocab_size)

    # ------------------------------------------------------------------
    # Separated encode / decode for inference
    # ------------------------------------------------------------------

    def encode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the encoder and return memory. (batch, src_len, d_model)"""
        scale = math.sqrt(self.d_model)
        src_emb = self.pos_encoding(self.embedding(src) * scale)
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    def decode_step(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run the decoder for the current tgt prefix and return logits
        for the next token.

        tgt    : (batch, cur_len)
        memory : (batch, src_len, d_model)

        Returns logits : (batch, vocab_size)  — logits at the last position
        """
        scale = math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(self.embedding(tgt) * scale)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1), device=tgt.device
        )
        out = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=_bool_to_additive(memory_key_padding_mask),
        )
        return self.out_proj(out[:, -1, :])  # (batch, vocab_size)

    # ------------------------------------------------------------------
    # Greedy decoding helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 1024,
        src_key_padding_mask: torch.Tensor | None = None,
        # ── Phase 1-5 (piano only) ──────────────────────────────────────
        init_token_idx: int | None = None,
        # ── Phase 6 (violin fine-tune) ──────────────────────────────────
        # Pass a list of token indices to force as a prefix after <sos>.
        # For violin: [Itgt_idx, Dtgt_idx]
        # e.g. [vocab["violin"], vocab["Lv.1"]]
        # Takes priority over init_token_idx when provided.
        init_tokens: list[int] | None = None,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> list[list[int]]:
        """
        Autoregressive decoding for a batch.

        Parameters
        ----------
        src : (batch, src_len)
        sos_idx : <sos> token index
        eos_idx : <eos> token index
        max_len : maximum number of generated tokens
        src_key_padding_mask : (batch, src_len) bool mask
        init_token_idx : (Phase 1-5) force a single token after <sos>
            e.g. the target difficulty Dtgt for piano-only mode.
            Kept for backward compatibility with existing piano inference code.
        init_tokens : (Phase 6) force a list of tokens after <sos>
            e.g. [violin_idx, Lv1_idx] for violin → violin Lv.1.
            Takes priority over init_token_idx when both are provided.
        temperature : softmax temperature (1.0 = greedy when top_k=0).
        top_k : if >0, sample from top-k logits.

        Returns
        -------
        List of token-index lists (one per batch item, without <sos>, <eos>,
        and without any forced init tokens).
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device

        memory = self.encode(src, _bool_to_additive(src_key_padding_mask))

        # Build the forced prefix: [<sos>] + forced tokens
        # init_tokens takes priority over init_token_idx
        tgt = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)

        if init_tokens is not None:
            # Phase 6: multiple forced tokens (e.g. [violin_idx, Lv1_idx])
            for tok_idx in init_tokens:
                forced = torch.full(
                    (batch_size, 1), tok_idx, dtype=torch.long, device=device
                )
                tgt = torch.cat([tgt, forced], dim=1)
        elif init_token_idx is not None:
            # Phase 1-5: single forced token (backward compatible)
            forced = torch.full(
                (batch_size, 1), init_token_idx, dtype=torch.long, device=device
            )
            tgt = torch.cat([tgt, forced], dim=1)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        outputs = [[] for _ in range(batch_size)]

        for _ in range(max_len):
            logits = self.decode_step(tgt, memory, memory_key_padding_mask=src_key_padding_mask)

            if top_k > 0:
                scaled = logits / max(temperature, 1e-8)
                topk_vals, _ = torch.topk(scaled, top_k, dim=-1)
                threshold = topk_vals[:, -1].unsqueeze(-1)
                scaled = scaled.masked_fill(scaled < threshold, float('-inf'))
                probs = torch.softmax(scaled, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            elif temperature != 1.0:
                probs = torch.softmax(logits / max(temperature, 1e-8), dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = logits.argmax(dim=-1)

            for i in range(batch_size):
                if not finished[i]:
                    tok = next_tokens[i].item()
                    if tok == eos_idx:
                        finished[i] = True
                    else:
                        outputs[i].append(tok)

            if finished.all():
                break

            tgt = torch.cat([tgt, next_tokens.unsqueeze(1)], dim=1)

        return outputs

    # ------------------------------------------------------------------
    # Fine-tune helper: resize embedding for new vocab
    # ------------------------------------------------------------------

    def resize_embedding(self, new_vocab_size: int) -> None:
        """
        Resize the embedding layer and output projection to support a larger
        vocabulary. Used when fine-tuning the piano model on violin data with
        vocab_violin.json, which adds 'piano' and 'violin' instrument tokens.

        New token embeddings are initialised with small random values.
        Existing token embeddings (piano vocab) are preserved exactly.
        """
        old_vocab_size = self.embedding.num_embeddings
        if new_vocab_size <= old_vocab_size:
            return  # nothing to do

        d_model = self.d_model
        n_new   = new_vocab_size - old_vocab_size

        # Embedding
        old_weight = self.embedding.weight.data
        new_weight = torch.empty(n_new, d_model, device=old_weight.device)
        nn.init.normal_(new_weight, mean=0.0, std=d_model ** -0.5)
        self.embedding = nn.Embedding(
            new_vocab_size, d_model, padding_idx=self.pad_idx
        )
        self.embedding.weight.data = torch.cat([old_weight, new_weight], dim=0)

        # Output projection
        old_out_w = self.out_proj.weight.data   # (old_vocab, d_model)
        old_out_b = self.out_proj.bias.data     # (old_vocab,)
        new_out_w = torch.empty(n_new, d_model, device=old_out_w.device)
        nn.init.xavier_uniform_(new_out_w)
        new_out_b = torch.zeros(n_new, device=old_out_b.device)
        self.out_proj = nn.Linear(new_vocab_size, d_model)
        self.out_proj.weight.data = torch.cat([old_out_w, new_out_w], dim=0)
        self.out_proj.bias.data   = torch.cat([old_out_b, new_out_b], dim=0)

        print(f"Embedding resized: {old_vocab_size} → {new_vocab_size} "
              f"(+{n_new} new tokens)")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(vocab_size: int, pad_idx: int = 0) -> ScoreRearrangementModel:
    """
    Build the default model matching the paper's ~0.3M parameter configuration.

    vocab_size : total vocabulary size (from vocab.json or vocab_violin.json)
    pad_idx    : index of the <pad> token
    """
    return ScoreRearrangementModel(
        vocab_size=vocab_size,
        d_model=48,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=96,
        dropout=0.1,
        max_seq_len=1024,
        pad_idx=pad_idx,
    )


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    # Change to vocab_violin.json when testing violin fine-tune
    vocab_path = "data/vocab_violin.json"
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)

    vocab_size = len(vocab["token_to_id"])
    pad_idx = vocab["token_to_id"]["<pad>"]

    model = build_model(vocab_size, pad_idx)
    n_params = model.count_parameters()
    print(f"Vocab size      : {vocab_size}")
    print(f"Total parameters: {n_params:,}  (~{n_params/1e6:.2f}M)")

    # Dummy forward pass
    batch, src_len, tgt_len = 4, 200, 180
    src = torch.randint(0, vocab_size, (batch, src_len))
    tgt = torch.randint(0, vocab_size, (batch, tgt_len))
    src_pad = (src == pad_idx)
    tgt_pad = (tgt == pad_idx)

    logits = model(src, tgt, src_pad, tgt_pad)
    print(f"Forward output  : {logits.shape}  (expected {batch}×{tgt_len}×{vocab_size})")

    # Greedy decode — piano mode (single init token, backward compatible)
    sos = vocab["token_to_id"]["<sos>"]
    eos = vocab["token_to_id"]["<eos>"]
    decoded = model.greedy_decode(src[:2], sos, eos, max_len=50,
                                  init_token_idx=vocab["token_to_id"].get("Lv.1"))
    print(f"Greedy (piano)  : {len(decoded)} sequences, "
          f"lengths {[len(s) for s in decoded]}")

    # Greedy decode — violin mode (multi-token init, Phase 6)
    if "violin" in vocab["token_to_id"]:
        violin_idx = vocab["token_to_id"]["violin"]
        lv1_idx = vocab["token_to_id"]["Lv.1"]
        decoded_v  = model.greedy_decode(src[:2], sos, eos, max_len=50,
                                         init_tokens=[violin_idx, lv1_idx])
        print(f"Greedy (violin) : {len(decoded_v)} sequences, "
              f"lengths {[len(s) for s in decoded_v]}")