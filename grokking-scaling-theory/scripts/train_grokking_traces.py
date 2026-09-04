"""
train_grokking_traces.py
========================
Activation-logging harness for the Phase 2 confirmatory sweep.

This is the last missing piece between the integrated analysis code and a real
grokking test of H1-H4: it wraps a modular-addition training loop and emits,
for every run, an analysis-ready per-example hidden-state trace in the exact
.npz format that ``grokking_scaling_theory.sheaf_order_parameter`` consumes:

    epochs  : int   [T]                checkpoint epochs
    hidden  : float [T, N, d]          per-example post-nonlinearity hidden state
    classes : int   [N]                rule class c(x) = (a + b) mod p

It also writes a run-table row with the extended schema pre-registered in
``experiments/BETA_DISCRIMINATOR_PROTOCOL.md`` (train_fraction, batch_size,
steps_per_epoch, budget_epochs) so censoring is structurally detectable and the
p^-2-vs-N^-1 degeneracy is breakable downstream.

Design choices (all local / CPU-friendly, matching the July integration):
  * Two matched architectures: ``mlp`` (SoftRose) and ``residual``
    (ResidualRose: identical embedding + width, hidden block h = h + ReLU(Wh)).
    The residual model is the matched-width/params variant the Phase 2
    pre-registration assumes; no residual arch survived in the repo, so it is
    (re)introduced here, conventionally.
  * Full-batch training (the existing loop is full-batch); batch_size and
    steps_per_epoch are recorded so tau stays convertible to optimizer steps.
  * Checkpointing runs to grok + ``post_grok_frac`` of the grok epoch (default
    +50%), NOT stopping at grokking, because the order-parameter transition and
    the val-accuracy transition need to be compared on the same trace.
  * Per-example hidden states are subsampled to ``max_examples`` (default 2000,
    per PHASE2 4.4) with a fixed, seed-derived index set so the same examples
    are logged at every checkpoint (cross-time comparison stays valid). Raw
    hidden width is stored; PCA-to-64 is an analysis-time step, not done here.
  * Resumable: if the target .npz already exists and ``--resume`` is set, the
    run is skipped, so an interrupted local sweep can be restarted cheaply.

Usage
=====
Single run::

    python scripts/train_grokking_traces.py --modulus 59 --arch mlp --seed 0

Full pre-registered sweep (3 p x 2 arch x 3 seeds = 18 runs)::

    python scripts/train_grokking_traces.py --sweep

Outputs (per run), under ``--output_dir`` (default data/phase2_traces/):
    trace_<arch>_p<p>_s<seed>.npz     per-example hidden-state trace
    trace_<arch>_p<p>_s<seed>.json    run metadata (extended schema)
and an appended row in ``<output_dir>/phase2_run_table.csv``.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim


# ---------------------------------------------------------------------------
# Models  (mlp = SoftRose, residual = ResidualRose; matched width/params)
# ---------------------------------------------------------------------------


class SoftRose(nn.Module):
    """The repository's canonical MLP for modular addition (from
    scripts/test_alignment_trigger.py). The logged hidden state is the
    post-ReLU activation of the single hidden layer."""

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(x).view(x.shape[0], -1)
        return self.relu(self.fc1(embeds))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.hidden(x))


class ResidualRose(nn.Module):
    """Matched-width residual variant of SoftRose.

    Same embedding and hidden width as SoftRose; the hidden representation is
    formed by a residual block ``h = h0 + ReLU(W h0)`` where h0 is the input
    projection. This is the minimal, conventional residual analogue the Phase 2
    pre-registration calls for ("residual, matched width/params"). The logged
    hidden state is the block output h (post-residual), which is the natural
    analogue of SoftRose's logged post-ReLU activation."""

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim * 2, hidden_dim)
        self.block = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.readout = nn.Linear(hidden_dim, vocab_size)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(x).view(x.shape[0], -1)
        h0 = self.proj(embeds)
        return h0 + self.relu(self.block(h0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.hidden(x))


class NoSkipRose(nn.Module):
    """Bypass/parameter control (PHASE2 amendment 3, item 5).

    Identical to ResidualRose with the skip connection removed:
    hidden = ReLU(W2 W1 e). Parameter-identical to ResidualRose
    (same proj + block + readout stack), but function-class-identical to
    SoftRose, since W2 W1 collapses to a single linear map. The contrast it
    isolates: if the sheaf signal follows ResidualRose, the cause is
    parametrization/optimization; if it follows SoftRose (unclassifiable),
    the linear bypass is load-bearing."""

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim * 2, hidden_dim)
        self.block = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.readout = nn.Linear(hidden_dim, vocab_size)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(x).view(x.shape[0], -1)
        return self.relu(self.block(self.proj(embeds)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.hidden(x))


class DeepRose(nn.Module):
    """Depth arms (PHASE2 amendments 5-6): n_hidden stacked ReLU layers.

    embedding -> [Linear+ReLU] x n_hidden -> readout. n_hidden=3 is the
    amendment-5 arm (crossing the *Logical Information Cells* ~3-layer
    bifurcation from below); n_hidden=5 is the amendment-6 Part B arm.
    The canonical logged hidden state is the final hidden layer (feeding
    the readout), matching prior arms; earlier layers are logged at a
    coarser cadence via ``hidden_all`` (false-null guard: cells may form
    in any layer)."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        n_hidden: int = 3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        dims = [embed_dim * 2] + [hidden_dim] * n_hidden
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(n_hidden)
        )
        self.relu = nn.ReLU()
        self.readout = nn.Linear(hidden_dim, vocab_size)

    def hidden_all(self, x: torch.Tensor) -> List[torch.Tensor]:
        h = self.embedding(x).view(x.shape[0], -1)
        outs: List[torch.Tensor] = []
        for layer in self.layers:
            h = self.relu(layer(h))
            outs.append(h)
        return outs

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.hidden_all(x)[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.hidden(x))


def build_model(arch: str, modulus: int, embed_dim: int, hidden_dim: int) -> nn.Module:
    if arch == "mlp":
        return SoftRose(modulus, embed_dim, hidden_dim)
    if arch == "residual":
        return ResidualRose(modulus, embed_dim, hidden_dim)
    if arch == "noskip":
        return NoSkipRose(modulus, embed_dim, hidden_dim)
    if arch == "widemlp":
        # Optional secondary arm (amendment 3, item 5): SoftRose widened to
        # approximately parameter-match ResidualRose. Run only after the
        # primary noskip arm is analyzed.
        return SoftRose(modulus, embed_dim, 448)
    if arch == "deep":
        return DeepRose(modulus, embed_dim, hidden_dim, n_hidden=3)
    if arch == "deep5":
        return DeepRose(modulus, embed_dim, hidden_dim, n_hidden=5)
    if arch == "deepnarrow":
        # Amendment 7: the pinned narrow arm: LIC-faithful depth (3) at
        # width 64; isolates width at fixed depth vs the amendment-5 arm.
        return DeepRose(modulus, embed_dim, 64, n_hidden=3)
    raise ValueError(
        f"Unknown arch '{arch}' (expected 'mlp', 'residual', 'noskip', "
        f"'widemlp', 'deep', 'deep5', 'deepnarrow')"
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def make_dataset(
    modulus: int, train_fraction: float, seed: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full pair table (a, b) -> (a + b) mod p, split by train_fraction.

    The split uses a numpy Generator seeded from ``seed`` so the same seed
    reproduces the same split, independently of global torch/random state.
    """
    rng = np.random.default_rng(seed)
    all_pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
    perm = rng.permutation(len(all_pairs))
    split_idx = int(len(all_pairs) * train_fraction)
    train_idx, val_idx = perm[:split_idx], perm[split_idx:]

    pairs = np.array(all_pairs)
    targets = (pairs[:, 0] + pairs[:, 1]) % modulus

    def to_ten(idx):
        x = torch.tensor(pairs[idx], dtype=torch.long)
        y = torch.tensor(targets[idx], dtype=torch.long)
        return x, y

    x_train, y_train = to_ten(train_idx)
    x_val, y_val = to_ten(val_idx)
    return x_train, y_train, x_val, y_val


# ---------------------------------------------------------------------------
# Run configuration and metadata (extended schema)
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    modulus: int
    arch: str
    seed: int
    weight_decay: float = 1.0
    learning_rate: float = 1e-3
    embed_dim: int = 128
    hidden_dim: int = 256
    train_fraction: float = 0.45
    budget_epochs: int = 40000
    checkpoint_every: int = 20
    post_grok_frac: float = 0.5
    grok_threshold: float = 90.0  # val-acc % defining the grok epoch
    max_examples: int = 2000
    optimizer: str = "adamw"
    # Amendment 5: cadence for auxiliary (non-final) hidden layers of
    # multi-layer architectures (coarser than checkpoint_every for storage).
    aux_every: int = 100


@dataclass
class RunResult:
    # extended run-table schema (BETA_DISCRIMINATOR_PROTOCOL.md)
    modulus: int
    arch: str
    seed: int
    weight_decay: float
    learning_rate: float
    train_fraction: float
    batch_size: int
    steps_per_epoch: int
    budget_epochs: int
    grokking_threshold: float  # val-acc %% defining tau (amendment 3, item 1)
    grokking_epoch: Optional[int]
    is_censored: bool
    n_train: int
    n_val: int
    n_logged_examples: int
    checkpoint_every: int
    n_checkpoints: int
    final_val_acc: float
    trace_path: str
    notes: str


# ---------------------------------------------------------------------------
# Training + logging
# ---------------------------------------------------------------------------


def _logged_indices(n_train: int, max_examples: int, seed: int) -> np.ndarray:
    """Fixed subsample of training-example rows to log (same at every
    checkpoint). If the train set already fits, log all of it."""
    if n_train <= max_examples:
        return np.arange(n_train)
    rng = np.random.default_rng(seed + 10_000)
    return np.sort(rng.choice(n_train, size=max_examples, replace=False))


def train_and_log(cfg: RunConfig, out_dir: Path, verbose: bool = True) -> RunResult:
    """Train one run; checkpoint per-example hidden states; write .npz + json."""
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cpu")  # local, CPU-only
    x_train, y_train, x_val, y_val = make_dataset(
        cfg.modulus, cfg.train_fraction, cfg.seed
    )
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    model = build_model(cfg.arch, cfg.modulus, cfg.embed_dim, cfg.hidden_dim).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    log_idx = _logged_indices(len(x_train), cfg.max_examples, cfg.seed)
    x_logged = x_train[log_idx]
    classes = y_train[log_idx].cpu().numpy().astype(int)  # c(x) = (a+b) mod p

    n_train = int(len(x_train))
    batch_size = n_train  # full-batch, matching the existing loop
    steps_per_epoch = 1

    epochs_logged: List[int] = []
    hidden_logged: List[np.ndarray] = []
    val_acc_logged: List[float] = []
    train_acc_logged: List[float] = []

    grok_epoch: Optional[int] = None
    stop_epoch = cfg.budget_epochs
    final_val_acc = 0.0

    epochs_aux_logged: List[int] = []
    hidden_aux_logged: List[List[np.ndarray]] = []  # per aux ckpt: [h1, h2, ...]

    def checkpoint(epoch: int) -> None:
        model.eval()
        with torch.no_grad():
            # Stored as float16 to keep the raw states while preserving the
            # full representation; the analysis pipeline upcasts to float64.
            if hasattr(model, "hidden_all"):
                # Amendment 5 (multi-layer arch): canonical `hidden` is the
                # final hidden layer; earlier layers are logged at the
                # coarser aux cadence (false-null guard for cell formation).
                hs = [
                    t.cpu().numpy().astype(np.float16)
                    for t in model.hidden_all(x_logged)
                ]
                h = hs[-1]
                if epoch % cfg.aux_every == 0 or epoch == 1:
                    epochs_aux_logged.append(epoch)
                    hidden_aux_logged.append(hs[:-1])
            else:
                h = model.hidden(x_logged).cpu().numpy().astype(np.float16)
        epochs_logged.append(epoch)
        hidden_logged.append(h)
        model.train()

    for epoch in range(1, cfg.budget_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        do_checkpoint = (epoch % cfg.checkpoint_every == 0) or (epoch == 1)
        if do_checkpoint:
            checkpoint(epoch)

        # Evaluate val/train accuracy on the same cadence: locates the grok
        # epoch AND stores the series in the trace (amendment 3, item 2), so
        # tau is re-derivable at any threshold without re-running. These
        # no_grad forward passes consume no RNG, so the parameter trajectory
        # is bit-identical to runs without them (determinism check vs v1).
        if do_checkpoint:
            model.eval()
            with torch.no_grad():
                val_acc = (
                    (model(x_val).argmax(1) == y_val).float().mean().item() * 100
                )
                train_acc = (
                    (model(x_train).argmax(1) == y_train).float().mean().item() * 100
                )
            model.train()
            val_acc_logged.append(val_acc)
            train_acc_logged.append(train_acc)
            final_val_acc = val_acc
            if grok_epoch is None and val_acc >= cfg.grok_threshold:
                grok_epoch = epoch
                # extend logging to grok + post_grok_frac, capped at budget
                stop_epoch = min(
                    cfg.budget_epochs,
                    int(round(grok_epoch * (1.0 + cfg.post_grok_frac))),
                )
            if verbose and (epoch % (cfg.checkpoint_every * 25) == 0 or epoch == 1):
                tag = f"grok@{grok_epoch}" if grok_epoch else "pre-grok"
                print(
                    f"  [{cfg.arch} p={cfg.modulus} s={cfg.seed}] "
                    f"ep {epoch:5d}  val {val_acc:5.1f}%  ({tag})"
                )

        # Stop once we've logged past grok + post_grok_frac.
        if grok_epoch is not None and epoch >= stop_epoch:
            break

    is_censored = grok_epoch is None
    hidden_arr = np.stack(hidden_logged, axis=0)  # [T, N, d]
    epochs_arr = np.array(epochs_logged, dtype=int)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"trace_{cfg.arch}_p{cfg.modulus}_s{cfg.seed}"
    npz_path = out_dir / f"{stem}.npz"
    extra: Dict[str, np.ndarray] = {}
    if hidden_aux_logged:
        for li in range(len(hidden_aux_logged[0])):
            extra[f"hidden_l{li + 1}"] = np.stack(
                [hs[li] for hs in hidden_aux_logged], axis=0
            )
        extra["epochs_aux"] = np.array(epochs_aux_logged, dtype=int)
    np.savez_compressed(
        npz_path,
        epochs=epochs_arr,
        hidden=hidden_arr,
        classes=classes,
        val_acc=np.array(val_acc_logged, dtype=np.float32),
        train_acc=np.array(train_acc_logged, dtype=np.float32),
        **extra,
    )

    notes = (
        "CENSORED: reached budget_epochs without grokking"
        if is_censored
        else f"grokked at epoch {grok_epoch}; logged to {epochs_arr[-1]}"
    )

    result = RunResult(
        modulus=cfg.modulus,
        arch=cfg.arch,
        seed=cfg.seed,
        weight_decay=cfg.weight_decay,
        learning_rate=cfg.learning_rate,
        train_fraction=cfg.train_fraction,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        budget_epochs=cfg.budget_epochs,
        grokking_threshold=cfg.grok_threshold,
        grokking_epoch=grok_epoch,
        is_censored=is_censored,
        n_train=n_train,
        n_val=int(len(x_val)),
        n_logged_examples=int(len(log_idx)),
        checkpoint_every=cfg.checkpoint_every,
        n_checkpoints=int(len(epochs_arr)),
        final_val_acc=float(final_val_acc),
        trace_path=str(npz_path),
        notes=notes,
    )

    (out_dir / f"{stem}.json").write_text(
        json.dumps({"config": asdict(cfg), "result": asdict(result)}, indent=2)
    )
    return result


# ---------------------------------------------------------------------------
# Run-table append
# ---------------------------------------------------------------------------


def append_run_table(result: RunResult, out_dir: Path) -> None:
    import csv

    table = out_dir / "phase2_run_table.csv"
    row = asdict(result)
    write_header = not table.exists()
    with open(table, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


SWEEP_MODULI = (59, 97, 113)
SWEEP_ARCHS = ("mlp", "residual")
SWEEP_SEEDS = (0, 1, 2)


def build_configs(args: argparse.Namespace) -> List[RunConfig]:
    if args.sweep:
        archs = (
            tuple(args.sweep_archs.split(",")) if args.sweep_archs else SWEEP_ARCHS
        )
        return [
            RunConfig(
                modulus=p,
                arch=a,
                seed=s,
                weight_decay=args.weight_decay,
                learning_rate=args.lr,
                train_fraction=args.train_fraction,
                budget_epochs=args.budget_epochs,
                checkpoint_every=args.checkpoint_every,
                post_grok_frac=args.post_grok_frac,
                max_examples=args.max_examples,
            )
            for p in SWEEP_MODULI
            for a in archs
            for s in SWEEP_SEEDS
        ]
    return [
        RunConfig(
            modulus=args.modulus,
            arch=args.arch,
            seed=args.seed,
            weight_decay=args.weight_decay,
            learning_rate=args.lr,
            train_fraction=args.train_fraction,
            budget_epochs=args.budget_epochs,
            checkpoint_every=args.checkpoint_every,
            post_grok_frac=args.post_grok_frac,
            max_examples=args.max_examples,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true", help="run the full 18-run sweep")
    parser.add_argument(
        "--sweep_archs", type=str, default=None,
        help="comma-separated arch override for --sweep (e.g. 'noskip' for the control arm)",
    )
    parser.add_argument("--modulus", type=int, default=59)
    parser.add_argument(
        "--arch", type=str, default="mlp",
        choices=["mlp", "residual", "noskip", "widemlp", "deep", "deep5", "deepnarrow"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--train_fraction", type=float, default=0.45)
    parser.add_argument("--budget_epochs", type=int, default=40000)
    parser.add_argument("--checkpoint_every", type=int, default=20)
    parser.add_argument("--post_grok_frac", type=float, default=0.5)
    parser.add_argument("--max_examples", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="data/phase2_traces")
    parser.add_argument("--resume", action="store_true", help="skip runs whose .npz exists")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    configs = build_configs(args)
    print(f"Planned runs: {len(configs)}  ->  {out_dir}")

    for i, cfg in enumerate(configs, 1):
        stem = f"trace_{cfg.arch}_p{cfg.modulus}_s{cfg.seed}"
        npz_path = out_dir / f"{stem}.npz"
        if args.resume and npz_path.exists():
            print(f"[{i}/{len(configs)}] skip (exists): {stem}")
            continue
        print(f"[{i}/{len(configs)}] train: {stem}")
        result = train_and_log(cfg, out_dir)
        append_run_table(result, out_dir)
        status = "CENSORED" if result.is_censored else f"grok@{result.grokking_epoch}"
        print(
            f"    done: {status}  val={result.final_val_acc:.1f}%  "
            f"checkpoints={result.n_checkpoints}  ex={result.n_logged_examples}"
        )


if __name__ == "__main__":
    main()
