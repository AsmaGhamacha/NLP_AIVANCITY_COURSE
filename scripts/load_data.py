from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split

#define data path
DATA_DIR = Path("C:/Users/asmag/Documents/NLP_Aivancity_03022025/NLP_AIVANCITY_COURSE/data/GeneratedTextDetection-main/Dataset/Hybrid_AbstractDataset/")
SAVE_DIR = Path("C:/Users/asmag/Documents/NLP_Aivancity_03022025/NLP_AIVANCITY_COURSE/data/")

#lists inizializing
train_texts, train_labels, test_texts, test_labels = [], [], [], []

# Read files from the directory
texts, labels = [], []
files = list(DATA_DIR.glob("*.txt"))

for file_path in files:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read().strip()
        label = 1 if "generatedAbstract" in file_path.name else 0  # 1: generated, 0: original
        texts.append(text)
        labels.append(label)

# Perform an 80/20 train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Create DataFrames
train_df = pd.DataFrame({"text": train_texts, "label": train_labels})
test_df = pd.DataFrame({"text": test_texts, "label": test_labels})

# Save DataFrames to CSV files
train_df.to_csv(SAVE_DIR / "train.csv", index=False)
test_df.to_csv(SAVE_DIR / "test.csv", index=False)

# Print lengths of the lists
print(f"len(train_texts): {len(train_texts)}")
print(f"len(train_labels): {len(train_labels)}")
print(f"len(test_texts): {len(test_texts)}")
print(f"len(test_labels): {len(test_labels)}")