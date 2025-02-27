import os 
from tqdm import tqdm
from preprocess import ProcessText4Classification
from sklearn.model_selection import StratifiedKFold
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from config import CONFIG

def loaders(gen_dir, hybrid_dir):
    text_dset = []
    labels = []
    
    for filename in tqdm(os.listdir(gen_dir), desc='Synthetic data'):
        if filename.endswith('.txt'):
            with open(os.path.join(gen_dir, filename), 'r', encoding='utf-8') as f:
                text_dset.append(f.read())
                labels.append(0)

    for filename in tqdm(os.listdir(hybrid_dir), desc='Hybrid data'):
        if filename.endswith('.txt'):
            with open(os.path.join(hybrid_dir, filename), 'r', encoding='utf-8') as f:
                text_dset.append(f.read())
                labels.append(1)
    
    model_process = ProcessText4Classification()
    process_test_dset = model_process._tokenizer_and_vectorization(text_dset)
    return process_test_dset, np.array(labels)

def baseline(n_splits = 5):
    X, y = loaders(gen_dir=CONFIG.gen_dir, hybrid_dir=CONFIG.hybrid_dir) 
    skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 42)
    accuracies = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"🔄 Fold {fold+1} / 5")
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_valid)
        acc = accuracy_score(y_valid, y_hat)
        accuracies.append(acc)
        print(f"✅ Fold {fold+1} - Accuracy: {acc:.4f}")

    print(f"\n📊 Mean average of the {n_splits} Folds: {np.mean(accuracies) * 100:.2f}%")

if __name__ == "__main__":
    baseline()