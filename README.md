# AI-Generated Text Detection

This repository contains an implementation of AI-generated text detection using two different approaches: **XGBoost (Extreme Gradient Boosting)** and **RoBERTa (a Transformer-based model)**. The goal is to evaluate whether traditional machine learning techniques can outperform large language models (LLMs) in detecting AI-generated text.

## Features
- **XGBoost Model:** A gradient boosting classifier trained on TF-IDF feature representations.
- **RoBERTa Model:** A fine-tuned Transformer-based neural network.
- **Cross-validation Setup:** Uses k-fold cross-validation for robust evaluation.
- **Preprocessing & Data Loading:** Efficient text preprocessing with tokenization and vectorization.
- **Visualization:** Performance comparison with plots and t-SNE visualizations.

## Installation

### Prerequisites
Ensure you have Python 3.7+ installed, along with the necessary dependencies. You can install them using:

```sh
pip install -r requirements.txt
```

### Required Libraries
- `torch`
- `transformers`
- `xgboost`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## Usage

### 1. Running XGBoost Baseline
To train and evaluate the XGBoost model, run:

```sh
python baseline.py
```

### 2. Training the RoBERTa Model
To fine-tune RoBERTa for text classification, use:

```sh
python bert_nn_transformers.py
```

### 3. Running the K-Fold Validation Experiment
For cross-validation with XGBoost:

```sh
python xgb_with_kfold.py
```

### 4. Visualizing Performance
To generate accuracy and time comparison plots, run:

```sh
python visual_plot.py
```

## Project Structure

```
├── baseline.py             # XGBoost baseline model
├── bert_nn_transformers.py # RoBERTa fine-tuning model
├── config.py               # Configuration settings
├── loaders.py              # Data loading and preprocessing
├── preprocess.py           # Text processing utilities
├── visual_plot.py          # Performance visualization
├── xgb_with_kfold.py       # XGBoost with K-Fold cross-validation
├── requirements.txt        # Dependencies
```

## Experimental Setup
All experiments were conducted on a personal machine with the following specifications:
- **RAM:** 16GB
- **Processor:** Intel Core i7 (14th generation)
- **GPU:** NVIDIA GeForce RTX 4060

## Results & Findings
- **XGBoost outperformed RoBERTa**, achieving higher accuracy and requiring less computational power.
- RoBERTa struggled to surpass 60% accuracy, even with hyperparameter tuning.
- Simpler machine learning models with well-engineered features can be **more effective than large language models** in AI text detection.

## Contributions
- **Bryan**: XGBoost model development and implementation.
- **Ketia**: Team organization, project management, and scientific paper research.
- **Asma**: RoBERTa model implementation (work under backup branch) and report generation.

## References
- Liu et al. (2019) on RoBERTa fine-tuning challenges: [arXiv.org](https://arxiv.org)
- XGBoost Documentation: [KDD.org](https://www.kdd.org)

