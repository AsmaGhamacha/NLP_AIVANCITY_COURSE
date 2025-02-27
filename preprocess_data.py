from transformers import RobertaTokenizer
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import pickle
import random
import nltk
from nltk.corpus import wordnet

# 📥 Download required NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')

def synonym_replacement(text, n=4):
    """
    Replace 'n' random words in the text with their synonyms.
    """
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)

    num_replaced = 0
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        if not synonyms:
            continue
        synonym_words = [lemma.name() for syn in synonyms for lemma in syn.lemmas() if lemma.name() != random_word]
        if not synonym_words:
            continue
        synonym = random.choice(synonym_words)
        new_words = [synonym if word == random_word else word for word in new_words]
        num_replaced += 1
        if num_replaced >= n:
            break

    return ' '.join(new_words)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def create_datasets(augment=False):
    """
    Loads CSV data, applies augmentation if enabled, and returns tokenized datasets.
    """
    # 📂 File paths
    train_csv_path = "C:/Users/asmag/Documents/NLP_Aivancity_03022025/NLP_AIVANCITY_COURSE/data/train.csv"
    test_csv_path = "C:/Users/asmag/Documents/NLP_Aivancity_03022025/NLP_AIVANCITY_COURSE/data/test.csv"

    # 🗂️ Load CSVs
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # 🚀 Data augmentation (applies only to training set)
    if augment:
        augmented_texts = []
        augmented_labels = []

        for text, label in zip(train_df['text'], train_df['label']):
            augmented_text = synonym_replacement(text, n=2)  # Change 'n' for stronger/weaker augmentation
            augmented_texts.append(augmented_text)
            augmented_labels.append(label)

        # 📈 Add augmented samples to the original training data
        augmented_df = pd.DataFrame({'text': augmented_texts, 'label': augmented_labels})
        train_df = pd.concat([train_df, augmented_df], ignore_index=True)

    # 📝 Tokenization
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    train_dataset = TextDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
    test_dataset = TextDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer)

    return train_dataset, test_dataset


if __name__ == "__main__":
    # ✅ Create datasets with data augmentation enabled (set augment=False to disable)
    train_dataset, test_dataset = create_datasets(augment=True)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # 💾 Save datasets
    processed_dir = "C:/Users/asmag/Documents/NLP_Aivancity_03022025/NLP_AIVANCITY_COURSE/data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    train_dataset_path = os.path.join(processed_dir, "train_dataset.pkl")
    test_dataset_path = os.path.join(processed_dir, "test_dataset.pkl")

    with open(train_dataset_path, "wb") as f:
        pickle.dump(train_dataset, f)

    with open(test_dataset_path, "wb") as f:
        pickle.dump(test_dataset, f)

    print(f"Tokenized datasets saved to {processed_dir}/")
