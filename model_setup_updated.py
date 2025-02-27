import torch
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, AdamW, get_scheduler
import pickle
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from preprocess_data import TextDataset

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
def setup_model(learning_rate=3e-5):
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-large", num_labels=2, hidden_dropout_prob=0.4, attention_probs_dropout_prob=0.4
    )

    # Freeze lower layers (freeze first 18 layers of 24)
    for layer in model.roberta.encoder.layer[:18]:
        for param in layer.parameters():
            param.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    return model, optimizer

# Training Loop with Early Stopping & Class Weights
def train(model, train_loader, test_loader, optimizer, scheduler, criterion, device, epochs=20, patience=3):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            loss = criterion(outputs.logits, batch['labels'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation using test_loader
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                val_loss = criterion(outputs.logits, batch['labels'])
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(test_loader)
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val (Test) Loss: {avg_val_loss:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_roberta_large.pth")  # Save best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered 🚀")
                break

if __name__ == "__main__":
    train_path = "C:/Users/asmag/Documents/NLP_Aivancity_03022025/NLP_AIVANCITY_COURSE/data/processed/train_dataset.pkl"
    test_path = "C:/Users/asmag/Documents/NLP_Aivancity_03022025/NLP_AIVANCITY_COURSE/data/processed/test_dataset.pkl"

    train_dataset, test_dataset = load_datasets(train_path, test_path)
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")

    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, batch_size=16)
    model, optimizer = setup_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Compute class weights dynamically
    train_labels = [label for label in train_dataset.labels]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = CrossEntropyLoss(weight=class_weights_tensor)

    # Scheduler with warmup steps
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_loader) * 20
    )

    # Train with early stopping
    train(model, train_loader, test_loader, optimizer, scheduler, criterion, device, epochs=20, patience=3)

    # Save final model
    model_save_path = "C:/Users/asmag/Documents/NLP_Aivancity_03022025/NLP_AIVANCITY_COURSE/models/roberta_large_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Final model saved to {model_save_path}")
