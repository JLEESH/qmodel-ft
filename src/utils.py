   
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from datasets import load_dataset
from tqdm import tqdm

# --- Tokenize the Dataset ---
def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


def evaluate_model(model, dataloader, metric):
    all_preds = []
    all_labels = []

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    #device = torch.device("mps")
    model.to(device)
    model.eval()

    for batch in dataloader:
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device)
            )

            predictions = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())

    # Compute metric
    final_score = metric.compute(predictions=all_preds, references=all_labels)
    #print(final_score)
    return final_score


# ---- Training Loop ----
def train_model(model, optimizer, lr_scheduler, train_dataloader, eval_dataloader, num_training_steps, num_epochs, run, n_iter=100):
    model.train()
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                labels=batch["label"]
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            
            # Log the loss to wandb
            run.log({"loss": loss.item()})
            if progress_bar.n % n_iter == 0:
                score = evaluate_model(model, eval_dataloader)
                run.log({"eval_accuracy": score["accuracy"], "f1": score["f1"]})


def train(model, train_dataset, val_dataset, epochs=3, batch_size=8, learning_rate=5e-5):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    model.train()
    for epoch in range(epochs):
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch.get('attention_mask', None)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        print(f"Epoch {epoch+1} Loss: {loss.item()}")

    return model

def evaluate(model, val_dataset, batch_size=8):
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch.get('attention_mask', None)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss}")
    return avg_loss < 0.1

def load_data(dataset_name, split='train'):
    dataset = load_dataset(dataset_name, split=split)
    return dataset
