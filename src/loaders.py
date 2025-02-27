import os 
import numpy as np 
from tqdm import tqdm
from preprocess import ProcessText4Classification
from sklearn.model_selection import train_test_split

def loaders(gen_dir, hybrid_dir):
    text_dset = []
    labels = []
    for filename in tqdm(os.listdir(gen_dir), desc = 'Synthethic data'):
        if filename.endswith('.txt'):
            with open(os.path.join(gen_dir, filename), 'r', encoding = 'utf-8') as f:
                text = f.read()
                text_dset.append(text)
                labels.append(0)
    
    for filename in tqdm(os.listdir(hybrid_dir), desc = 'Synthethic data'):
        if filename.endswith('.txt'):
            with open(os.path.join(hybrid_dir, filename), 'r', encoding = 'utf-8') as f:
                text = f.read()
                text_dset.append(text)
                labels.append(1)
    
    model_process = ProcessText4Classification()
    process_test_dset = model_process._tokenizer_and_vectorization(text_dset)
    x_train, x_valid, y_train, y_valid = train_test_split(process_test_dset, labels, test_size = .2, shuffle = False, random_state = 42)
    return x_train, x_valid, y_train, y_valid

def loaders2(gen_dir, hybrid_dir):
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
    
