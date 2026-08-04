import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split

# BEFORE RUNNING install the ff:
# python -m pip install transformers peft accelerate

# --- 1. Custom Joint Loss Function ---
class OrthogonalJuryLoss(nn.Module):
    def __init__(self, lambda_penalty=3.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_penalty = lambda_penalty

    def forward(self, logits1, logits2, targets):
        # Standard Cross-Entropy for individual accuracy
        loss1 = self.ce_loss(logits1, targets)
        loss2 = self.ce_loss(logits2, targets)
        
        # Get probabilities for class 1 (True)
        probs1 = F.softmax(logits1, dim=1)[:, 1]
        probs2 = F.softmax(logits2, dim=1)[:, 1]
        
        # Calculate Error Residuals
        float_targets = targets.float()
        err1 = probs1 - float_targets
        err2 = probs2 - float_targets
        
        # Orthogonality Penalty: Dot product of errors
        shared_error_penalty = torch.mean(err1 * err2)
        
        # Total Loss
        total_loss = loss1 + loss2 + (self.lambda_penalty * shared_error_penalty)
        return total_loss, loss1, loss2, shared_error_penalty

# --- 2. Dataset Loader ---
class FuzzyTrapDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=256):
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        self.encodings = tokenizer(
            [item['scenario_text'] for item in data], 
            truncation=True, 
            padding=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
        self.labels = torch.tensor([1 if item['ground_truth'] else 0 for item in data])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

# --- 3. Evaluation Function ---
@torch.no_grad()
def evaluate_split(model1, model2, dataloader, device, dtype):
    model1.eval()
    model2.eval()
    
    all_err1, all_err2 = [], []
    correct1, correct2, joint_correct = 0, 0, 0
    total = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.cuda.amp.autocast(dtype=dtype):
            outputs1 = model1(input_ids, attention_mask=attention_mask)
            outputs2 = model2(input_ids, attention_mask=attention_mask)

        probs1 = F.softmax(outputs1.logits, dim=1)[:, 1].cpu().numpy()
        probs2 = F.softmax(outputs2.logits, dim=1)[:, 1].cpu().numpy()
        targets = labels.cpu().numpy()

        all_err1.extend(probs1 - targets)
        all_err2.extend(probs2 - targets)

        preds1 = (probs1 > 0.5).astype(int)
        preds2 = (probs2 > 0.5).astype(int)
        
        correct1 += np.sum(preds1 == targets)
        correct2 += np.sum(preds2 == targets)
        joint_correct += np.sum((preds1 == targets) | (preds2 == targets))
        total += len(targets)

    acc1 = correct1 / total
    acc2 = correct2 / total
    joint_acc = joint_correct / total
    
    # Calculate Error Correlation
    rho, _ = pearsonr(all_err1, all_err2) if len(all_err1) > 1 else (0.0, 0.0)
    
    return acc1, acc2, joint_acc, float(rho)

# --- 4. Main Training Pipeline ---
def train_orthogonal_models(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    dataset_path="datasets/tom_fuzzy_dataset.json",
    batch_size=4,
    epochs=5,
    lr=2e-4,
    lambda_penalty=3.0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[*] Training on device: {device} | Precision: {dtype}")

    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load Full Dataset & Perform 80 / 10 / 10 Split
    full_dataset = FuzzyTrapDataset(dataset_path, tokenizer)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"[*] Dataset split: {train_size} Train | {val_size} Val | {test_size} Test")

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    print(f"[*] Loading base model '{model_name}' for Model 1 & Model 2...")
    
    def load_peft_model_instance():
        m = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            torch_dtype=dtype,
            device_map="auto"
        )
        m.config.pad_token_id = tokenizer.pad_token_id
        m.gradient_checkpointing_enable()
        m = get_peft_model(m, peft_config)
        return m

    model1 = load_peft_model_instance()
    model2 = load_peft_model_instance()

    print("[*] LoRA adapters injected successfully.")
    model1.print_trainable_parameters()

    # Optimizer setup (only updates LoRA parameters)
    trainable_params = list(model1.parameters()) + list(model2.parameters())
    optimizer = AdamW(trainable_params, lr=lr)
    criterion = OrthogonalJuryLoss(lambda_penalty=lambda_penalty)

    metrics_history = []

    print("\n" + "="*70)
    print(" STARTING MISTRAL-7B LORA JOINT NEGATIVE CORRELATION TRAINING")
    print("="*70)

    for epoch in range(epochs):
        model1.train()
        model2.train()
        epoch_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=dtype):
                outputs1 = model1(input_ids, attention_mask=attention_mask)
                outputs2 = model2(input_ids, attention_mask=attention_mask)
                loss, l1, l2, penalty = criterion(outputs1.logits, outputs2.logits, labels)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # Evaluate on Validation Set
        val_acc1, val_acc2, val_joint_acc, val_rho = evaluate_split(
            model1, model2, val_loader, device, dtype
        )

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")
        print(f"  [Val] Acc M1: {val_acc1:.1%} | Acc M2: {val_acc2:.1%} | Joint Acc: {val_joint_acc:.1%}")
        print(f"  [Val] Error Correlation (ρ): {val_rho:.3f}")
        print("-" * 70)

        metrics_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "acc1": val_acc1,
            "acc2": val_acc2,
            "joint_acc": val_joint_acc,
            "rho": val_rho
        })

    # Save training metrics for plots.py
    with open("training_curves.json", "w") as f:
        json.dump(metrics_history, f, indent=4)
    print("[+] Training curves saved to training_curves.json")

    # Final Test Set Evaluation
    print("\n" + "="*70)
    print(" RUNNING FINAL EVALUATION ON HELD-OUT TEST SET")
    print("="*70)
    test_acc1, test_acc2, test_joint_acc, test_rho = evaluate_split(
        model1, model2, test_loader, device, dtype
    )

    test_results = {
        "test_acc_m1": test_acc1,
        "test_acc_m2": test_acc2,
        "test_joint_acc": test_joint_acc,
        "test_error_correlation_rho": test_rho
    }

    print(f"Test Set Results:")
    print(f"  -> Model 1 Accuracy:      {test_acc1:.1%}")
    print(f"  -> Model 2 Accuracy:      {test_acc2:.1%}")
    print(f"  -> Joint Jury Accuracy:   {test_joint_acc:.1%}")
    print(f"  -> Final Error Correlation: {test_rho:.3f}")

    with open("test_results.json", "w") as f:
        json.dump(test_results, f, indent=4)
    print("[+] Final test results saved to test_results.json")

    # Save LoRA Adapters
    os.makedirs("checkpoints/model1_lora", exist_ok=True)
    os.makedirs("checkpoints/model2_lora", exist_ok=True)
    model1.save_pretrained("checkpoints/model1_lora")
    model2.save_pretrained("checkpoints/model2_lora")
    print("[+] LoRA adapters saved to ./checkpoints/")

if __name__ == "__main__":
    train_orthogonal_models()