   
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from datasets import load_dataset
from tqdm import tqdm

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
