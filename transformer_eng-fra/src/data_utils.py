import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


@dataclass
class Vocabulary:
    token_to_id: Dict[str, int]
    id_to_token: List[str]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.token_to_id.get(tok, self.unk_id) for tok in tokens]

    def decode(self, ids: Sequence[int], stop_at_eos: bool = True) -> List[str]:
        tokens: List[str] = []
        for idx in ids:
            tok = self.id_to_token[idx]
            if stop_at_eos and tok == EOS_TOKEN:
                break
            if tok in {PAD_TOKEN, BOS_TOKEN}:
                continue
            tokens.append(tok)
        return tokens


_TOKENIZER = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize(text: str) -> List[str]:
    return _TOKENIZER.findall(text.lower().strip())


def detokenize(tokens: Sequence[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([?.!,;:])", r"\1", text)
    return text


def read_parallel_data(file_path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            src, tgt = line.split("\t", maxsplit=1)
            pairs.append((src.strip(), tgt.strip()))
    return pairs


def build_vocabulary(texts: Sequence[str], max_vocab_size: int = 12000, min_freq: int = 1) -> Vocabulary:
    counter: Counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    special_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    id_to_token = special_tokens[:]
    for token, freq in sorted_tokens:
        if freq < min_freq:
            continue
        if token in special_tokens:
            continue
        if len(id_to_token) >= max_vocab_size:
            break
        id_to_token.append(token)

    token_to_id = {tok: i for i, tok in enumerate(id_to_token)}
    return Vocabulary(token_to_id=token_to_id, id_to_token=id_to_token)


class TranslationDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        max_src_len: int = 40,
        max_tgt_len: int = 40,
    ) -> None:
        self.pairs = list(pairs)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        src_text, tgt_text = self.pairs[idx]

        src_tokens = tokenize(src_text)[: self.max_src_len - 2]
        tgt_tokens = tokenize(tgt_text)[: self.max_tgt_len - 2]

        src_ids = [self.src_vocab.bos_id] + self.src_vocab.encode(src_tokens) + [self.src_vocab.eos_id]
        tgt_ids = [self.tgt_vocab.bos_id] + self.tgt_vocab.encode(tgt_tokens) + [self.tgt_vocab.eos_id]

        return {"src_ids": src_ids, "tgt_ids": tgt_ids, "src_text": src_text, "tgt_text": tgt_text}


def make_collate_fn(src_pad_id: int, tgt_pad_id: int) -> Callable:
    def collate_fn(batch: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        src_max = max(len(item["src_ids"]) for item in batch)
        tgt_max = max(len(item["tgt_ids"]) for item in batch)

        src_batch = []
        tgt_batch = []

        for item in batch:
            src = item["src_ids"] + [src_pad_id] * (src_max - len(item["src_ids"]))
            tgt = item["tgt_ids"] + [tgt_pad_id] * (tgt_max - len(item["tgt_ids"]))
            src_batch.append(src)
            tgt_batch.append(tgt)

        return {
            "src_ids": torch.tensor(src_batch, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_batch, dtype=torch.long),
        }

    return collate_fn
