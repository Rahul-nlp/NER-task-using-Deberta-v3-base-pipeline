# scripts/train.py

import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForTokenClassification

from src.data.processor import read_conll, build_label_maps
from src.data.dataset import NERDataset, collate_fn
from src.models.trainer import NERTrainer

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 7
LR = 5e-5

def main():
    sentences, labels = read_conll("data/train.conll")
    unique_labels, label2id, id2label = build_label_maps(labels)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = NERDataset(
        sentences, labels, tokenizer, label2id, MAX_LEN
    )

    n_val = int(0.2 * len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    trainer = NERTrainer(model, optimizer, device, id2label)

    for epoch in range(1, EPOCHS + 1):
        trainer.train_epoch(train_loader, epoch)
        val_loss, val_f1, y_true, y_pred = trainer.evaluate(val_loader)
        print(f"Epoch {epoch} | Val loss: {val_loss:.4f} | F1: {val_f1:.4f}")

    trainer.plot_metrics()
    trainer.report(y_true, y_pred)

if __name__ == "__main__":
    main()
