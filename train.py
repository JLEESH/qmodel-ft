import torch
import wandb

import torch.optim as optim
from torch.utils.data import DataLoader

from model.model import QModel
from src.utils import train, evaluate, load_data

import argparse

def main():
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate a causal language model.")
    parser.add_argument('--save_model', action='store_true', default=False, help="Flag to save the trained model.")
    parser.add_argument('--model_name', type=str, default="bert-base-uncased", help="Name of the model to use.")
    parser.add_argument('--wandb', action='store_true', default=True, help="Flag to enable Weights & Biases logging.")
    
    args = parser.parse_args()
    save_model = args.save_model
    print(f"Save model flag is set to: {save_model}")
    model_name = args.model_name
    print(f"Model name is set to: {model_name}")
    use_wandb = args.wandb
    print(f"Use Weights & Biases logging: {use_wandb}")
    
    
    # Model and tokenizer names
    # ~~Uncomment the model you want to use~~
    # (Use arguments to specify the model and tokenizer names)
    
    #model_name = "gpt2-medium"  # Example model name
    #tokenizer_name = "gpt2-medium"  # Example tokenizer name
    #model_name = "gpt2"
    #tokenizer_name = "gpt2"
    #model_name = "TheBloke/Mistral-7B-v0.1-Q4_0"  # Example model name
    
    #model_name = "TheBloke/Mistral-7B-v0.1-AWQ"
    #model_name = "bert-base-uncased"
    tokenizer_name = model_name  # Assuming tokenizer name is the same as model name
    
    epochs = 3
    batch_size = 16
    learning_rate = 2e-5
    
    # Initialize Weights & Biases
    if use_wandb:
        wandb.init(project="qmodel-ft", entity="your_entity_name", name="qmodel-training")
        wandb.config.update({
            "model_name": model_name,
            "tokenizer_name": tokenizer_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        })
    
    # Load model and tokenizer
    print(f"Loading model: {model_name} and tokenizer: {tokenizer_name}")
    
    # Initialize the model
    model = QModel(model_name, tokenizer_name)
    
    # Check for available device (GPU/CPU)
    print("Checking for available device...")
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'    
    model.to(device)
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    #train_dataset = load_data("wikitext", split='train')
    #val_dataset = load_data("wikitext", split='validation')
    dataset_name = "glue"
    if dataset_name == "glue":
        from datasets import load_dataset
        train_dataset = load_dataset("glue", "mrpc", split='train')
        val_dataset = load_dataset("glue", "mrpc", split='validation')
        test_dataset = load_dataset("glue", "mrpc", split='test')
    else:
        train_dataset = load_data(dataset_name, split='train')
        val_dataset = load_data(dataset_name, split='validation')
        test_dataset = load_data(dataset_name, split='test')
    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    
    # Preprocess the dataset
    print("Preprocessing the dataset...")
    train_dataset = train_dataset.map(lambda x: model.tokenizer(x['text'], truncation=True, padding='max_length', max_length=512), batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    val_dataset = val_dataset.map(lambda x: model.tokenizer(x['text'], truncation=True, padding='max_length', max_length=512), batched=True)
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    print("Training dataset format:", train_dataset.column_names)
    print("Validation dataset format:", val_dataset.column_names)
    
    # Train the model
    print("Training the model...")
    model.train()
    trained_model = train(model, train_dataset, val_dataset, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    print("Training completed.")
    
    # Evaluate the model
    print("Starting evaluation...")
    model.eval()
    evaluate(trained_model, val_dataset, batch_size=batch_size)
    evaluate(trained_model, test_dataset, batch_size=batch_size)
    print("Evaluation completed.")
    
    # Save the trained model
    if save_model:
        print("Saving the trained model...")
        model_save_path = "trained_qmodel"
        model.model.save_pretrained(model_save_path)
        model.tokenizer.save_pretrained(model_save_path)
        print(f"Model saved to {model_save_path}")
    
    # Finish Weights & Biases run
    if use_wandb:
        print("Finishing Weights & Biases run...")
        wandb.finish()


#     print("Training and evaluation completed successfully.")
#     print("You can now use the trained model for inference or further fine-tuning.")
#     print("For inference, load the model using:")
#     print(f"model = QModel('{model_name}', '{tokenizer_name}')")
#     print("For further fine-tuning, use the train function with your new dataset.")
#     print("For evaluation, use the evaluate function with your validation dataset.")
#     print("Thank you for using the QModel training script!")
#     print("For any issues or questions, please refer to the documentation or raise an issue on the GitHub repository.")
#     print("Happy coding!")
#     print("For more information, visit the Hugging Face documentation at https://huggingface.co/docs/transformers/index")
    
if __name__ == "__main__":
    main()


# End of code snippet
# This code defines a simple training and evaluation loop for a causal language model using PyTorch and the Hugging Face Transformers library. It includes model definition, training, evaluation, and data loading functions
# for a specified dataset. The model is trained on the training dataset and evaluated on the validation dataset, with progress displayed using tqdm.
# The model is designed to be flexible, allowing for different model and tokenizer names, and can be easily adapted for other datasets or configurations. The training and evaluation processes are encapsulated in functions for
# modularity and reusability. The model is trained on a GPU if available, and the training and validation losses are printed to the console.
# The script is structured to be run as a standalone program, with a main function that orchestrates the training and evaluation process. The model can be further extended or modified to include additional features such as
# logging, saving checkpoints, or hyperparameter tuning.