import torch
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, AdamW, get_scheduler
import pickle
from tqdm import tqdm
from preprocess_data import TextDataset
from torch.nn import CrossEntropyLoss
import numpy as np 
from sklearn.utils.class_weight import compute_class_weight

# Load Datasets
def load_datasets(train_path, test_path):
    with open(train_path, "rb") as f:
        train_dataset = pickle.load(f)
    with open(test_path, "rb") as f:
        test_dataset = pickle.load(f)
    return train_dataset, test_dataset

# Create DataLoaders
def create_dataloaders(train_dataset, test_dataset, batch_size=16):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

# Setup Model, Optimizer, Scheduler
def setup_model(learning_rate=1e-5):
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2, hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    return model, optimizer

# Training Loop with Gradient Clipping & Scheduler
def train(model, train_loader, optimizer, scheduler, criterion, device, epochs=10):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            loss = criterion(outputs.logits, batch['labels'])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f" Epoch {epoch + 1} - Average Loss: {total_loss / len(train_loader):.4f}")

if __name__ == "__main__":
    train_path = "C:/Users/asmag/Documents/NLP_Aivancity_03022025/NLP_AIVANCITY_COURSE/data/processed/train_dataset.pkl"
    test_path = "C:/Users/asmag/Documents/NLP_Aivancity_03022025/NLP_AIVANCITY_COURSE/data/processed/test_dataset.pkl"

    train_dataset, test_dataset = load_datasets(train_path, test_path)
    print(f" Train dataset: {len(train_dataset)} samples")
    print(f" Test dataset: {len(test_dataset)} samples")

    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, batch_size=16)
    model, optimizer = setup_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Training on: {device}")

    # Class weights to handle imbalance (adjust if needed)
    class_weights = torch.tensor([1.0, 1.5], device=device)
    criterion = CrossEntropyLoss(weight=class_weights)

    # Scheduler for learning rate decay
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * 20
    )

    train(model, train_loader, optimizer, scheduler, criterion, device, epochs=20)

    # Save Model
    model_save_path = "C:/Users/asmag/Documents/NLP_Aivancity_03022025/NLP_AIVANCITY_COURSE/models/roberta_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
