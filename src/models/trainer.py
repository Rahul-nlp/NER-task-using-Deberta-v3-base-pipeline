# src/models/trainer.py

import torch
from seqeval.metrics import f1_score, classification_report
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


class NERTrainer:
    def __init__(self, model, optimizer, device, id2label):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.id2label = id2label

        self.train_losses = []
        self.val_losses = []
        self.val_f1s = []

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            self.optimizer.zero_grad()

            outputs = self.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                labels=batch["labels"].to(self.device),
            )

            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"train_loss": loss.item()})

        avg_loss = running_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        all_true, all_pred = [], []

        with torch.no_grad():
            for batch in val_loader:
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["labels"].to(self.device),
                )

                total_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)

                for p_seq, l_seq in zip(preds, batch["labels"]):
                    p_labels, l_labels = [], []
                    for p_i, l_i in zip(p_seq, l_seq):
                        if l_i == -100:
                            continue
                        p_labels.append(self.id2label[p_i.item()])
                        l_labels.append(self.id2label[l_i.item()])
                    all_pred.append(p_labels)
                    all_true.append(l_labels)

        avg_loss = total_loss / len(val_loader)
        f1 = f1_score(all_true, all_pred)

        self.val_losses.append(avg_loss)
        self.val_f1s.append(f1)

        return avg_loss, f1, all_true, all_pred

    def plot_metrics(self):
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure()
        plt.plot(epochs, self.train_losses, label="Train loss")
        plt.plot(epochs, self.val_losses, label="Val loss")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(epochs, self.val_f1s, marker="o")
        plt.show()

    def report(self, all_true, all_pred):
        print(classification_report(all_true, all_pred))
