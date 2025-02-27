import torch
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification
import pickle
from preprocess_data import TextDataset
from sklearn.metrics import accuracy_score, classification_report

# Charger le dataset tokenisé
def load_dataset(path):
    if path is None:
        raise ValueError("Le chemin du dataset ne peut pas être None. Assurez-vous de fournir un chemin valide.")
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    return dataset

# Créer le DataLoader
def create_dataloader(dataset, batch_size=16):
    return DataLoader(dataset, batch_size=batch_size)

# Fonction d'évaluation
def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["Original", "Generated"])
    return accuracy, report

if __name__ == "__main__":
    # Chemins du dataset de test et du modèle sauvegardé
    test_path = "C:/Users/asmag/Documents/NLP_Aivancity_03022025/NLP_AIVANCITY_COURSE/data/processed/test_dataset.pkl"
    model_path = "C:/Users/asmag/Documents/NLP_Aivancity_03022025/NLP_AIVANCITY_COURSE/models/roberta_model.pth"

    try:
        # Charger le dataset de test
        test_dataset = load_dataset(test_path)
        test_loader = create_dataloader(test_dataset, batch_size=16)

        # Charger le modèle
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
        model.load_state_dict(torch.load(model_path, map_location=device))

        # Évaluer le modèle
        accuracy, report = evaluate(model, test_loader, device)

        print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
        print("Classification Report:")
        print(report)

    except Exception as e:
        print(f"Une erreur est survenue : {e}")
