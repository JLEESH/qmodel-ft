import argparse
import dotenv
import torch
import wandb
import evaluate
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer, get_scheduler
from datasets import load_dataset
from tqdm import tqdm
#from model.model import QModel
from src.utils import preprocess_function, train_model, evaluate_model

dotenv.load_dotenv()

def main():
    argparser = argparse.ArgumentParser(description="Fine-tune a model.")
    argparser.add_argument('--model_name', type=str, default="bert-base-uncased", help="Name of the model to use.")
    argparser.add_argument('--save_model', action='store_true', default=False, help="Flag to save the trained model.")
    argparser.add_argument('--wandb', action='store_true', default=False, help="Flag to enable Weights & Biases logging.")
    args = argparser.parse_args()
    
    model_name = args.model_name
    save_model = args.save_model
    use_wandb = args.wandb
    
    epochs = 3
    batch_size = 16
    learning_rate = 2e-5
    entity_name = dotenv.get_key('.env', 'ENTITY_NAME')
    
    run = wandb.init(
        #project="qmodel-ft",
        project="qmodel-ft-check",
        name="qmodel-ft-training",
        entity=entity_name,
        config={
            "model_name": model_name,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size
        }
    ) if use_wandb else None

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Model: {model_name}, Tokenizer: {tokenizer.name_or_path}")

    #device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    #elif torch.backends.mps.is_available():
    else:
        device = "mps"
    model.to(device)
    print(f"Using device: {device}")

    task_name = "mrpc"
    if task_name in ["mrpc", "sst2", "cola", "qnli", "qqp", "rte", "stsb"]:
        dataset = load_dataset("glue", task_name)
        metric = evaluate.load("glue", task_name)
    else:
        dataset = load_dataset(task_name)
        metric = evaluate.load(task_name)
    print(f"Loaded dataset: {task_name}")
    
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    
    # Torch format
    encoded_dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "label"]
    )

    train_dataloader = DataLoader(
        encoded_dataset["train"], shuffle=True, batch_size=batch_size
    )
    eval_dataloader = DataLoader(
        encoded_dataset["validation"], batch_size=batch_size
    )
    test_dataloader = DataLoader(
        encoded_dataset["test"], batch_size=batch_size
    )
    
    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_training_steps = epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    print("Starting training...")
    train_model(model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                num_training_steps=num_training_steps,
                num_epochs=epochs,
                run=run,
                n_iter=10)
    trained_model = model
    print("Training complete.")

    evaluate_model(model,
                   eval_dataloader=eval_dataloader,
                   metric=metric)
    print("Evaluation complete.")
    print()

if __name__ == "__main__":
    main()
