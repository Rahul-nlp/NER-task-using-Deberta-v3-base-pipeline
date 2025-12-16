# src/data/processor.py

from typing import List, Tuple

def read_conll(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    sentences = []
    labels = []
    cur_tokens = []
    cur_labels = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                if cur_tokens:
                    sentences.append(cur_tokens)
                    labels.append(cur_labels)
                    cur_tokens, cur_labels = [], []
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            token = parts[0]
            label = parts[-1]

            if token == "-DOCSTART-":
                continue

            cur_tokens.append(token)
            cur_labels.append(label)

    if cur_tokens:
        sentences.append(cur_tokens)
        labels.append(cur_labels)

    return sentences, labels


def build_label_maps(labels):
    unique_labels = sorted(set(l for seq in labels for l in seq))
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}
    return unique_labels, label2id, id2label
