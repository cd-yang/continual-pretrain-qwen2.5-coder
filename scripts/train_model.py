import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def load_data(data_path):
    """
    Load preprocessed data from a CSV file.
    
    Args:
        data_path (str): Path to the preprocessed data file.
        
    Returns:
        Dataset: Loaded preprocessed data.
    """
    return load_dataset('csv', data_files=data_path)

def define_model(model_name):
    """
    Define the model architecture.
    
    Args:
        model_name (str): Name of the pre-trained model.
        
    Returns:
        model: Defined model.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def train_model(model, tokenizer, train_dataset, output_dir):
    """
    Train the model.
    
    Args:
        model: Model to be trained.
        tokenizer: Tokenizer for the model.
        train_dataset: Dataset for training.
        output_dir (str): Directory to save the trained model.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    trainer.save_model(output_dir)

def main():
    data_path = os.path.join('data', 'preprocessed_data.csv')
    model_name = 'qwen2.5-coder'
    output_dir = os.path.join('models', 'qwen2.5-coder')
    
    # Load preprocessed data
    train_dataset = load_data(data_path)
    
    # Define model
    model, tokenizer = define_model(model_name)
    
    # Train model
    train_model(model, tokenizer, train_dataset['train'], output_dir)

if __name__ == "__main__":
    main()
