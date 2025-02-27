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
- https://medium.com/@nghihuynh_37300/understanding-loss-functions-for-classification-81c19ee72c2a#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6Ijc2M2Y3YzRjZDI2YTFlYjJiMWIzOWE4OGY0NDM0ZDFmNGQ5YTM2OGIiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJhdWQiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJzdWIiOiIxMDAyNTMyNTUzNzU4MTk2OTQ4MDkiLCJlbWFpbCI6ImFzbWEuZ2hhbWFjaGFAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsIm5iZiI6MTc0MDQzMzY0MywibmFtZSI6IkFzbWEgR2hhbWFjaGEiLCJwaWN0dXJlIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EvQUNnOG9jSml5SWJ5NURTcWtFaUdFb0pxWHJLcEw5anA3OGt2XzVVUUpncTg5ZFplU0N5OHV5YUw9czk2LWMiLCJnaXZlbl9uYW1lIjoiQXNtYSIsImZhbWlseV9uYW1lIjoiR2hhbWFjaGEiLCJpYXQiOjE3NDA0MzM5NDMsImV4cCI6MTc0MDQzNzU0MywianRpIjoiNWZmNGU3NjBhY2E1NWM4YzQ5Y2I4NzRjMWRjYTEzNTA3NjgyYmUxMiJ9.us_PI21ewWXX3THHmXGorDdRap945VUik7RGobnS7c0PG279uD1UcPNgvqepYyZ9NB9ax_pLipCPb22Sp4_X3o55qs2TM3vBQoL_LGMTjWcdSG1b0E5wMAKSsz8pkMDX40xomr24dJzIDI1PN1eS8CSrW5qLYgKRe4CUmMJlfKiWZy-LktCgUR3Yg4X-Z8D7KxqmYadUxxpEx9IGWMywWNRw7zFm_9WQqCNjH7IUyBYfh_daYjCYiII3m3fniA2snrqD_DIvaXHWuvr1ragQbNLdGaXp1XWgUjdPxen7XSCOYKs3Oin24Cvt5SdfJr0qzwm8FpfA_Mquep_QTqL6FQ
- https://www.analyticsvidhya.com/blog/2021/03/binary-cross-entropy-log-loss-for-binary-classification/ 
