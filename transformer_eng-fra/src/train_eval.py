import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data_utils import (
    Vocabulary,
    build_vocabulary,
    detokenize,
    make_collate_fn,
    read_parallel_data,
    tokenize,
    TranslationDataset,
)
from .model import ModelConfig, Seq2SeqTransformer


@dataclass
class TrainConfig:
    train_file: str = "dataset/eng-fra_train_data.txt"
    test_file: str = "dataset/eng-fra_test_data.txt"
    output_dir: str = "outputs"
    batch_size: int = 64
    epochs: int = 100
    lr: float = 3e-4
    max_vocab_size: int = 12000
    max_len: int = 40
    d_model: int = 128
    num_heads: int = 4
    ff_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    seed: int = 42
    attention_types: Tuple[str, ...] = ("dot", "additive")
    sample_size: int = 8
    train_limit: int = -1
    test_limit: int = -1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_data(cfg: TrainConfig):
    train_pairs = read_parallel_data(cfg.train_file)
    test_pairs = read_parallel_data(cfg.test_file)

    if cfg.train_limit > 0:
        train_pairs = train_pairs[: cfg.train_limit]
    if cfg.test_limit > 0:
        test_pairs = test_pairs[: cfg.test_limit]

    src_vocab = build_vocabulary([s for s, _ in train_pairs], max_vocab_size=cfg.max_vocab_size)
    tgt_vocab = build_vocabulary([t for _, t in train_pairs], max_vocab_size=cfg.max_vocab_size)

    train_ds = TranslationDataset(train_pairs, src_vocab, tgt_vocab, cfg.max_len, cfg.max_len)
    test_ds = TranslationDataset(test_pairs, src_vocab, tgt_vocab, cfg.max_len, cfg.max_len)

    collate_fn = make_collate_fn(src_vocab.pad_id, tgt_vocab.pad_id)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader, train_pairs, test_pairs, src_vocab, tgt_vocab


def count_unigram_overlap(pred: Sequence[str], ref: Sequence[str]) -> int:
    ref_counts: Dict[str, int] = {}
    for tok in ref:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1

    overlap = 0
    for tok in pred:
        if ref_counts.get(tok, 0) > 0:
            overlap += 1
            ref_counts[tok] -= 1
    return overlap


def compute_simple_bleu(predictions: Sequence[Sequence[str]], references: Sequence[Sequence[str]]) -> float:
    clipped = 0
    total = 0
    pred_len = 0
    ref_len = 0

    for pred, ref in zip(predictions, references):
        clipped += count_unigram_overlap(pred, ref)
        total += max(len(pred), 1)
        pred_len += len(pred)
        ref_len += len(ref)

    precision = clipped / total if total > 0 else 0.0
    if pred_len == 0:
        return 0.0

    bp = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / max(pred_len, 1))
    return 100.0 * bp * precision


def run_epoch(
    model: Seq2SeqTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        src_ids = batch["src_ids"].to(device)
        tgt_ids = batch["tgt_ids"].to(device)

        decoder_in = tgt_ids[:, :-1]
        labels = tgt_ids[:, 1:]

        logits = model(src_ids, decoder_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(
    model: Seq2SeqTransformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tgt_vocab: Vocabulary,
    max_len: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0

    all_preds: List[List[str]] = []
    all_refs: List[List[str]] = []
    token_hits = 0
    token_total = 0

    for batch in dataloader:
        src_ids = batch["src_ids"].to(device)
        tgt_ids = batch["tgt_ids"].to(device)

        decoder_in = tgt_ids[:, :-1]
        labels = tgt_ids[:, 1:]

        logits = model(src_ids, decoder_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        total_loss += loss.item()

        preds = model.greedy_decode(src_ids, bos_id=tgt_vocab.bos_id, eos_id=tgt_vocab.eos_id, max_new_tokens=max_len)
        pred_tokens_batch = [tgt_vocab.decode(seq.tolist()[1:], stop_at_eos=True) for seq in preds]
        ref_tokens_batch = [tgt_vocab.decode(seq.tolist()[1:], stop_at_eos=True) for seq in tgt_ids]

        all_preds.extend(pred_tokens_batch)
        all_refs.extend(ref_tokens_batch)

        pred_for_acc = logits.argmax(dim=-1)
        non_pad = labels != tgt_vocab.pad_id
        token_hits += ((pred_for_acc == labels) & non_pad).sum().item()
        token_total += non_pad.sum().item()

    bleu = compute_simple_bleu(all_preds, all_refs)
    token_acc = token_hits / max(token_total, 1)

    return {
        "eval_loss": total_loss / max(len(dataloader), 1),
        "token_acc": token_acc,
        "bleu": bleu,
    }


@torch.no_grad()
def sample_translations(
    model: Seq2SeqTransformer,
    pairs: Sequence[Tuple[str, str]],
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    device: torch.device,
    sample_size: int,
    max_len: int,
) -> List[Dict[str, str]]:
    chosen = list(pairs[:sample_size])
    samples: List[Dict[str, str]] = []

    for src_text, tgt_text in chosen:
        src_tokens = tokenize(src_text)[: max_len - 2]
        src_ids = [src_vocab.bos_id] + src_vocab.encode(src_tokens) + [src_vocab.eos_id]
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

        pred_ids = model.greedy_decode(src_tensor, bos_id=tgt_vocab.bos_id, eos_id=tgt_vocab.eos_id, max_new_tokens=max_len)
        pred_tokens = tgt_vocab.decode(pred_ids[0].tolist()[1:], stop_at_eos=True)
        pred_text = detokenize(pred_tokens)

        samples.append({
            "src": src_text,
            "ref": tgt_text,
            "pred": pred_text,
        })

    return samples


def train_one_attention(
    attn_type: str,
    cfg: TrainConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    train_pairs: Sequence[Tuple[str, str]],
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    device: torch.device,
    output_dir: Path,
) -> Dict:
    model_cfg = ModelConfig(
        src_vocab_size=len(src_vocab.id_to_token),
        tgt_vocab_size=len(tgt_vocab.id_to_token),
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        num_encoder_layers=cfg.num_layers,
        num_decoder_layers=cfg.num_layers,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
        attention_type=attn_type,
    )

    model = Seq2SeqTransformer(model_cfg, src_vocab.pad_id, tgt_vocab.pad_id).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id)

    history = []
    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
        eval_metrics = evaluate(model, test_loader, criterion, device, tgt_vocab, cfg.max_len)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **eval_metrics,
        }
        history.append(row)
        print(
            f"[{attn_type}] epoch={epoch} train_loss={train_loss:.4f} "
            f"eval_loss={eval_metrics['eval_loss']:.4f} token_acc={eval_metrics['token_acc']:.4f} bleu={eval_metrics['bleu']:.2f}"
        )

    samples = sample_translations(
        model=model,
        pairs=train_pairs,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device,
        sample_size=cfg.sample_size,
        max_len=cfg.max_len,
    )

    out = {
        "attention_type": attn_type,
        "model_config": asdict(model_cfg),
        "history": history,
        "final_metrics": history[-1],
        "samples": samples,
    }

    with open(output_dir / f"results_{attn_type}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    torch.save(model.state_dict(), output_dir / f"model_{attn_type}.pt")
    return out


def write_comparison_md(results: List[Dict], out_path: Path) -> None:
    lines = []
    lines.append("# Attention Comparison")
    lines.append("")
    lines.append("| Attention | Train Loss | Eval Loss | Token Acc | BLEU |")
    lines.append("|---|---:|---:|---:|---:|")

    for item in results:
        fm = item["final_metrics"]
        lines.append(
            f"| {item['attention_type']} | {fm['train_loss']:.4f} | {fm['eval_loss']:.4f} | {fm['token_acc']:.4f} | {fm['bleu']:.2f} |"
        )

    lines.append("")
    lines.append("## Translation Samples")
    lines.append("")

    for item in results:
        lines.append(f"### {item['attention_type']}")
        for s in item["samples"][:5]:
            lines.append(f"- EN: {s['src']}")
            lines.append(f"- FR(ref): {s['ref']}")
            lines.append(f"- FR(pred): {s['pred']}")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_loss_curves(results: List[Dict], out_path: Path) -> None:
    plt.figure(figsize=(10, 6))

    for item in results:
        name = item["attention_type"]
        epochs = [row["epoch"] for row in item["history"]]
        train_losses = [row["train_loss"] for row in item["history"]]
        eval_losses = [row["eval_loss"] for row in item["history"]]

        plt.plot(epochs, train_losses, label=f"{name}-train", linewidth=2)
        plt.plot(epochs, eval_losses, label=f"{name}-eval", linewidth=2, linestyle="--")

    plt.title("Loss Curves: Dot-Product vs Additive Attention")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Transformer EN->FR with dot/additive attention.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--max-vocab-size", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=-1)
    parser.add_argument("--test-limit", type=int, default=-1)
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        max_len=args.max_len,
        max_vocab_size=args.max_vocab_size,
        seed=args.seed,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )

    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, train_pairs, _, src_vocab, tgt_vocab = build_data(cfg)

    results = []
    for attn_type in cfg.attention_types:
        result = train_one_attention(
            attn_type=attn_type,
            cfg=cfg,
            train_loader=train_loader,
            test_loader=test_loader,
            train_pairs=train_pairs,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=device,
            output_dir=output_dir,
        )
        results.append(result)

    write_comparison_md(results, output_dir / "comparison.md")
    plot_loss_curves(results, output_dir / "loss_comparison.png")

    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "results": results}, f, ensure_ascii=False, indent=2)

    print("Done. Comparison written to outputs/comparison.md and outputs/loss_comparison.png")


if __name__ == "__main__":
    main()
