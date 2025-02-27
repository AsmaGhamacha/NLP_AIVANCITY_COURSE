import os
import torch
import numpy as np
from tqdm import tqdm
from config import CONFIG
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_scheduler
from torch.utils.data import Dataset, DataLoader

MODEL_CKPT = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {  
            "input_ids": self.encodings["input_ids"][idx],  
            "attention_mask": self.encodings["attention_mask"][idx],  
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),  
        }

class BertClassifier(torch.nn.Module):
    def __init__(self, num_labels=2):
        super(BertClassifier, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT, num_labels=num_labels)
        
        for param in self.bert.base_model.parameters():
            param.requires_grad = False
        for param in self.bert.base_model.encoder.layer[-2:].parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, labels=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

def loaders(gen_dir, hybrid_dir):
    texts, labels = [], []
    for filename in tqdm(os.listdir(gen_dir), desc='Loading Synthetic Data'):
        if filename.endswith('.txt'):
            with open(os.path.join(gen_dir, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(0)

    for filename in tqdm(os.listdir(hybrid_dir), desc='Loading Hybrid Data'):
        if filename.endswith('.txt'):
            with open(os.path.join(hybrid_dir, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(1)

    encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")    
    return encodings, np.array(labels)

def train_model():
    X, y = loaders(gen_dir=CONFIG.gen_dir, hybrid_dir=CONFIG.hybrid_dir)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    accuracies = []
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X["input_ids"], y)):
        print(f"Fold {fold+1} / 5")
        train_dataset = TextDataset({key: X[key][train_idx] for key in X}, y[train_idx])
        valid_dataset = TextDataset({key: X[key][valid_idx] for key in X}, y[valid_idx])
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

        model = BertClassifier().to(CONFIG.device)
        optimizer = AdamW(model.parameters(), lr=5e-5)
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 3)
        loss_fn = torch.nn.CrossEntropyLoss()

        model.train()
        for epoch in range(3):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()

                batch = {k: v.to(CONFIG.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = loss_fn(outputs.logits, batch["labels"])
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}: Training Loss = {total_loss / len(train_loader):.4f}")

        model.eval()
        y_preds, y_true = [], []
        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(CONFIG.device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                y_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                y_true.extend(batch["labels"].cpu().numpy())

        acc = accuracy_score(y_true, y_preds)
        accuracies.append(acc)
        print(f"Fold {fold+1} - Accuracy: {acc:.4f}")

    print(f"\nFinal Mean Accuracy: {np.mean(accuracies) * 100:.2f}%")

if __name__ == "__main__":
    train_model()