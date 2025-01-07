# Qwen2.5-Coder Continual Pre-Training

This repository provides a framework for the continual pre-training of the Qwen2.5-Coder model. The goal is to enhance the model's performance by continually training it on new data.

## Setup

### Environment Setup

1. Clone the repository:

2. Create a virtual environment and activate it:

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Data Preprocessing

1. Place your raw data in the `data` directory.

2. Run the data preprocessing script:
    ```bash
    python scripts/preprocess_data.py
    ```

## Model Training

1. Run the model training script:
    ```bash
    python scripts/train_model.py
    ```

## Model Evaluation

1. Run the model evaluation script:
    ```bash
    python scripts/evaluate_model.py
    ```
