import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from loaders import loaders2
from config import CONFIG
import xgboost as xgb
from bert_nn_transformers import BertClassifier, tokenizer, TextDataset
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

X, y = loaders2(CONFIG.gen_dir, CONFIG.hybrid_dir)

# XGBoost Benchmark
def train_xgboost(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies, train_times, infer_times = [], [], []
    
    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        
        model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
        
        start_train = time.time()
        model.fit(X_train, y_train)
        train_times.append(time.time() - start_train)
        
        start_infer = time.time()
        y_pred = model.predict(X_valid)
        infer_times.append(time.time() - start_infer)
        
        acc = accuracy_score(y_valid, y_pred)
        accuracies.append(acc)
    
    return accuracies, train_times, infer_times

# BERT Benchmark
def train_bert(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies, train_times, infer_times = [], [], []
    device = CONFIG.device
    
    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        
        train_encodings = tokenizer(list(map(str, X_train)), truncation=True, padding=True, max_length=512, return_tensors='pt')        
        valid_encodings = tokenizer(list(map(str, X_valid)), truncation=True, padding=True, max_length=512, return_tensors='pt')
        
        train_dataset = TextDataset(train_encodings, y_train.tolist())
        valid_dataset = TextDataset(valid_encodings, y_valid.tolist())
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=8)
        
        model = BertClassifier(num_labels=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        criterion = torch.nn.CrossEntropyLoss()
        
        start_train = time.time()
        model.train()
        for epoch in range(3):
            for batch in train_loader:
                optimizer.zero_grad()
                inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
                labels = batch['labels'].to(device)
                outputs = model(**inputs)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()
        train_times.append(time.time() - start_train)
        
        start_infer = time.time()
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in valid_loader:
                inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
                outputs = model(**inputs)
                preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
        infer_times.append(time.time() - start_infer)
        
        acc = accuracy_score(y_valid, preds)
        accuracies.append(acc)
    
    return accuracies, train_times, infer_times

xgb_acc, xgb_train, xgb_infer = train_xgboost(X, y)
bert_acc, bert_train, bert_infer = train_bert(X, y)
plt.figure(figsize=(12, 5))

# Barplot for accuracy
plt.subplot(1, 2, 1)
sns.barplot(x=['XGBoost', 'BERT'], y=[np.mean(xgb_acc), np.mean(bert_acc)], palette='viridis')
plt.ylabel('Accuracy')
plt.title('Performance Comparison')

# Lineplot for times (Training & Inference)
plt.subplot(1, 2, 2)
sns.lineplot(x=['XGBoost', 'BERT'], y=[np.mean(xgb_train), np.mean(bert_train)], label='Train Time', marker='o', color='blue')
sns.lineplot(x=['XGBoost', 'BERT'], y=[np.mean(xgb_infer), np.mean(bert_infer)], label='Inference Time', marker='o', color='orange')
plt.ylabel('Time (s)')
plt.title('Time Comparison')
plt.legend()
plt.show()

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init = 'random')
X_embedded = tsne.fit_transform(X)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y, palette='coolwarm')
plt.title('Visual T-SNE')
plt.show()
