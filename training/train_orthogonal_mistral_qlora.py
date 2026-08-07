import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup # Added Scheduler
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split

# --- 1. Custom Joint Loss Function ---
class OrthogonalJuryLoss(nn.Module):
    def __init__(self, lambda_penalty=0.5): # Lowered to 0.5 to prevent mode collapse
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_penalty = lambda_penalty

    def forward(self, logits1, logits2, targets):
        loss1 = self.ce_loss(logits1, targets)
        loss2 = self.ce_loss(logits2, targets)
        
        probs1 = F.softmax(logits1, dim=1)[:, 1]
        probs2 = F.softmax(logits2, dim=1)[:, 1]
        
        float_targets = targets.float()
        err1 = probs1 - float_targets
        err2 = probs2 - float_targets
        
        shared_error_penalty = torch.mean(err1 * err2)
        
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

# --- 3. Main Training Pipeline ---
def train_orthogonal_models(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    train_dataset_path="datasets/tom_inverted_dataset_1000.json", 
    ood_dataset_path="datasets/ood_test_dataset_150.json",
    batch_size=2, 
    accumulation_steps=4, # Added for speed
    epochs=12,            # Increased for slower, deeper learning
    lr=2e-5,              # Lowered 10x to prevent mode collapse
    lambda_penalty=0.5    # Lowered 
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[*] Training on device: {device} | Precision: {dtype}")

    # OPTIMIZATION: Enable cuDNN benchmarking for faster convolutions/operations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 1. Load Training Dataset and split 90/10 for Train/Validation
    full_train_dataset = FuzzyTrapDataset(train_dataset_path, tokenizer)
    total_train_size = len(full_train_dataset)
    train_size = int(0.9 * total_train_size)
    val_size = total_train_size - train_size

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    # OPTIMIZATION: Added num_workers and pin_memory for faster data loading
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 2. Load the Explicit OOD Test Dataset
    print(f"[*] Loading OOD Test Dataset from {ood_dataset_path}...")
    ood_dataset = FuzzyTrapDataset(ood_dataset_path, tokenizer)
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"[*] Dataset sizes: {train_size} Train | {val_size} Val | {len(ood_dataset)} OOD Test")

    # Configure 4-bit Quantization (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    print(f"[*] Loading 4-bit quantized base model '{model_name}' for Model 1 (GPU 0) & Model 2 (GPU 1)...")
    
    num_gpus = torch.cuda.device_count()
    device1 = 'cuda:0' if num_gpus > 0 else 'cpu'
    device2 = 'cuda:1' if num_gpus > 1 else device1

    def load_peft_model_instance(device_map_target):
        m = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            quantization_config=bnb_config,
            device_map={'': device_map_target},
            attn_implementation="sdpa" # OPTIMIZATION: Use Scaled Dot Product Attention 
        )
        m.config.pad_token_id = tokenizer.pad_token_id
        
        # Prepare for QLoRA training
        m = prepare_model_for_kbit_training(m)
        m.gradient_checkpointing_enable()
        m = get_peft_model(m, peft_config)
        return m

    model1 = load_peft_model_instance(0 if num_gpus > 0 else "auto")
    model2 = load_peft_model_instance(1 if num_gpus > 1 else (0 if num_gpus > 0 else "auto"))

    print("[*] LoRA adapters injected successfully.")
    
    trainable_params = list(model1.parameters()) + list(model2.parameters())
    optimizer = AdamW(trainable_params, lr=lr)
    criterion = OrthogonalJuryLoss(lambda_penalty=lambda_penalty)

    # --- SCHEDULER SETUP ---
    # Total steps = (number of batches / accumulation steps) * epochs
    total_training_steps = (len(train_loader) // accumulation_steps) * epochs
    warmup_steps = int(0.1 * total_training_steps) # 10% warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_training_steps
    )
    print(f"[*] Configured Scheduler: {total_training_steps} total steps, {warmup_steps} warmup steps.")

    metrics_history = []

    print("\n" + "="*70)
    print(" STARTING MISTRAL-7B QLORA JOINT NEGATIVE CORRELATION TRAINING")
    print("="*70)

    for epoch in range(epochs):
        model1.train()
        model2.train()
        epoch_loss = 0
        
        # OPTIMIZATION: Zero gradients before the epoch starts
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids1 = batch['input_ids'].to(model1.device, non_blocking=True)
            attention_mask1 = batch['attention_mask'].to(model1.device, non_blocking=True)
            labels1 = batch['labels'].to(model1.device, non_blocking=True)
            
            input_ids2 = batch['input_ids'].to(model2.device, non_blocking=True)
            attention_mask2 = batch['attention_mask'].to(model2.device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=dtype):
                outputs1 = model1(input_ids1, attention_mask=attention_mask1)
                outputs2 = model2(input_ids2, attention_mask=attention_mask2)
                
                logits2_on_device1 = outputs2.logits.to(model1.device)
                
                loss, l1, l2, penalty = criterion(outputs1.logits, logits2_on_device1, labels1)
                
                # OPTIMIZATION: Scale the loss for gradient accumulation
                loss = loss / accumulation_steps

            loss.backward()
            
            # OPTIMIZATION: Step the optimizer only after 'accumulation_steps' batches
            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                scheduler.step() # Step the scheduler here
                optimizer.zero_grad()
                
            epoch_loss += (loss.item() * accumulation_steps) # Re-scale for reporting

        avg_loss = epoch_loss / len(train_loader)

        # Evaluate on Validation Set
        def device_aware_eval(m1, m2, loader):
            m1.eval()
            m2.eval()
            
            all_err1, all_err2 = [], []
            correct1, correct2, joint_correct = 0, 0, 0
            total = 0

            with torch.no_grad():
                for batch in loader:
                    input_ids1 = batch['input_ids'].to(m1.device, non_blocking=True)
                    attention_mask1 = batch['attention_mask'].to(m1.device, non_blocking=True)
                    labels1 = batch['labels'].to(m1.device, non_blocking=True)
                    
                    input_ids2 = batch['input_ids'].to(m2.device, non_blocking=True)
                    attention_mask2 = batch['attention_mask'].to(m2.device, non_blocking=True)

                    with torch.amp.autocast('cuda', dtype=dtype):
                        outputs1 = m1(input_ids1, attention_mask=attention_mask1)
                        outputs2 = m2(input_ids2, attention_mask=attention_mask2)

                    probs1 = F.softmax(outputs1.logits, dim=1)[:, 1].float().cpu().numpy()
                    probs2 = F.softmax(outputs2.logits, dim=1)[:, 1].float().cpu().numpy()
                    
                    targets = batch['labels'].numpy()

                    all_err1.extend(probs1 - targets)
                    all_err2.extend(probs2 - targets)

                    preds1 = (probs1 > 0.5).astype(int)
                    preds2 = (probs2 > 0.5).astype(int)
                    
                    correct1 += np.sum(preds1 == targets)
                    correct2 += np.sum(preds2 == targets)
                    joint_correct += np.sum((preds1 == targets) | (preds2 == targets))
                    total += len(targets)

            acc1 = correct1 / total if total > 0 else 0
            acc2 = correct2 / total if total > 0 else 0
            joint_acc = joint_correct / total if total > 0 else 0
            rho, _ = pearsonr(all_err1, all_err2) if len(all_err1) > 1 else (0.0, 0.0)
            return acc1, acc2, joint_acc, float(rho)

        val_acc1, val_acc2, val_joint_acc, val_rho = device_aware_eval(model1, model2, val_loader)

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

    with open("training_curves.json", "w") as f:
        json.dump(metrics_history, f, indent=4)
    print("[+] Training curves saved to training_curves.json")

    print("\n" + "="*70)
    print(" RUNNING FINAL EVALUATION ON OUT-OF-DISTRIBUTION (OOD) TEST SET")
    print("="*70)
    
    ood_acc1, ood_acc2, ood_joint_acc, ood_rho = device_aware_eval(model1, model2, ood_loader)

    ood_results = {
        "ood_acc_m1": ood_acc1,
        "ood_acc_m2": ood_acc2,
        "ood_joint_acc": ood_joint_acc,
        "ood_error_correlation_rho": ood_rho
    }

    print(f"OOD Test Set Results:")
    print(f"  -> Model 1 Accuracy:      {ood_acc1:.1%}")
    print(f"  -> Model 2 Accuracy:      {ood_acc2:.1%}")
    print(f"  -> Joint Jury Accuracy:   {ood_joint_acc:.1%}")
    print(f"  -> Final Error Correlation: {ood_rho:.3f}")

    with open("ood_test_results.json", "w") as f:
        json.dump(ood_results, f, indent=4)
    print("[+] Final OOD test results saved to ood_test_results.json")

    os.makedirs("checkpoints/model1_lora", exist_ok=True)
    os.makedirs("checkpoints/model2_lora", exist_ok=True)
    model1.save_pretrained("checkpoints/model1_lora")
    model2.save_pretrained("checkpoints/model2_lora")
    print("[+] LoRA adapters saved to ./checkpoints/")

if __name__ == "__main__":
    train_orthogonal_models()
