
from transformers import RobertaTokenizer
import torch
from torch.utils.data import Dataset
import pandas as pd
#from pathlib import Path


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

if __name__ == "__main__":
    # Load data from CSV files created earlier
    train_df = pd.read_csv("C:/Users/asmag/Documents/NLP_Aivancity_03022025/NLP_AIVANCITY_COURSE/data/train.csv")
    test_df = pd.read_csv("C:/Users/asmag/Documents/NLP_Aivancity_03022025/NLP_AIVANCITY_COURSE/data/test.csv")

    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Create dataset objects
    train_dataset = TextDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
    test_dataset = TextDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
