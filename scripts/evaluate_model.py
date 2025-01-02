import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def load_trained_model(model_path):
    """
    Load the trained model from the specified path.
    
    Args:
        model_path (str): Path to the trained model.
        
    Returns:
        model: Loaded trained model.
        tokenizer: Tokenizer for the model.
    """
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def load_data(data_path):
    """
    Load evaluation data from a CSV file.
    
    Args:
        data_path (str): Path to the evaluation data file.
        
    Returns:
        Dataset: Loaded evaluation data.
    """
    return load_dataset('csv', data_files=data_path)

def evaluate_model(model, tokenizer, eval_dataset):
    """
    Evaluate the model on the evaluation dataset.
    
    Args:
        model: Trained model to be evaluated.
        tokenizer: Tokenizer for the model.
        eval_dataset: Dataset for evaluation.
        
    Returns:
        dict: Evaluation results.
    """
    model.eval()
    results = {}
    for example in eval_dataset['train']:
        inputs = tokenizer(example['text'], return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        # Example evaluation metric: perplexity
        loss = outputs.loss
        perplexity = torch.exp(loss)
        results[example['id']] = perplexity.item()
    return results

def main():
    model_path = os.path.join('models', 'qwen2.5-coder')
    data_path = os.path.join('data', 'evaluation_data.csv')
    
    # Load trained model
    model, tokenizer = load_trained_model(model_path)
    
    # Load evaluation data
    eval_dataset = load_data(data_path)
    
    # Evaluate model
    results = evaluate_model(model, tokenizer, eval_dataset)
    
    # Print evaluation results
    for example_id, perplexity in results.items():
        print(f"Example ID: {example_id}, Perplexity: {perplexity}")

if __name__ == "__main__":
    main()
