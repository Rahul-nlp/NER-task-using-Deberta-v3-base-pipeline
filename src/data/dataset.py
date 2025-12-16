# src/data/dataset.py

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class NERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, label2id, max_len):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        self.data = [
            self.encode_sentence(toks, labs)
            for toks, labs in zip(sentences, labels)
        ]

    def encode_sentence(self, tokens, labs):
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids()
        label_ids = []
        prev_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                label_ids.append(self.label2id[labs[word_idx]])
            else:
                label_ids.append(self.label2id[labs[word_idx]])
            prev_word_idx = word_idx

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_ids),
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, tokenizer):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    pad_id = tokenizer.pad_token_id or 0

    return {
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=pad_id),
        "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
        "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
    }
