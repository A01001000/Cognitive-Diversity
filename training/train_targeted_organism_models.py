import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
import random
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

def set_seed(seed=42):
    """Locks down all random operations for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

set_seed(42)

# --- 1. Dataset Loader ---
class FuzzyTrapDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=256, persona_prefix=""):
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Append the steering prefix directly to the text
        modified_texts = [persona_prefix + item['scenario_text'] for item in data]
        
        self.encodings = tokenizer(
            modified_texts, 
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

# --- 2. Evaluation Function ---
@torch.no_grad()
def evaluate_jury(m1, m2, loader, dtype):
    m1.eval()
    m2.eval()
    
    all_err1, all_err2 = [], []
    correct1, correct2, joint_correct = 0, 0, 0
    total = 0

    for batch in loader:
        input_ids1 = batch['input_ids'].to(m1.device, non_blocking=True)
        attention_mask1 = batch['attention_mask'].to(m1.device, non_blocking=True)
        
        input_ids2 = batch['input_ids'].to(m2.device, non_blocking=True)
        attention_mask2 = batch['attention_mask'].to(m2.device, non_blocking=True)
        labels = batch['labels'].numpy()

        with torch.amp.autocast('cuda', dtype=dtype):
            outputs1 = m1(input_ids1, attention_mask=attention_mask1)
            outputs2 = m2(input_ids2, attention_mask=attention_mask2)

        probs1 = F.softmax(outputs1.logits, dim=1)[:, 1].float().cpu().numpy()
        probs2 = F.softmax(outputs2.logits, dim=1)[:, 1].float().cpu().numpy()

        all_err1.extend(probs1 - labels)
        all_err2.extend(probs2 - labels)

        preds1 = (probs1 > 0.5).astype(int)
        preds2 = (probs2 > 0.5).astype(int)
        
        correct1 += np.sum(preds1 == labels)
        correct2 += np.sum(preds2 == labels)
        joint_correct += np.sum((preds1 == labels) | (preds2 == labels))
        total += len(labels)

    acc1 = correct1 / total if total > 0 else 0
    acc2 = correct2 / total if total > 0 else 0
    joint_acc = joint_correct / total if total > 0 else 0
    rho, _ = pearsonr(all_err1, all_err2) if len(all_err1) > 1 else (0.0, 0.0)
    return acc1, acc2, joint_acc, float(rho)

# Passing zipped loader logic dynamically for eval
@torch.no_grad()
def dynamic_eval(m1, m2, loader1, loader2, dtype):
    m1.eval()
    m2.eval()
    all_err1, all_err2 = [], []
    correct1, correct2, joint_correct = 0, 0, 0
    total = 0
    
    # Zip allows us to pull the M1-prefixed batch and M2-prefixed batch at the same time
    for b1, b2 in zip(loader1, loader2):
        i1, a1, lbl = b1['input_ids'].to(m1.device), b1['attention_mask'].to(m1.device), b1['labels'].numpy()
        i2, a2 = b2['input_ids'].to(m2.device), b2['attention_mask'].to(m2.device)
        
        with torch.amp.autocast('cuda', dtype=dtype):
            o1 = m1(i1, attention_mask=a1)
            o2 = m2(i2, attention_mask=a2)

        p1 = F.softmax(o1.logits, dim=1)[:, 1].float().cpu().numpy()
        p2 = F.softmax(o2.logits, dim=1)[:, 1].float().cpu().numpy()

        all_err1.extend(p1 - lbl)
        all_err2.extend(p2 - lbl)

        pr1, pr2 = (p1 > 0.5).astype(int), (p2 > 0.5).astype(int)
        correct1 += np.sum(pr1 == lbl)
        correct2 += np.sum(pr2 == lbl)
        joint_correct += np.sum((pr1 == lbl) | (pr2 == lbl))
        total += len(lbl)

    ac1 = correct1 / total if total > 0 else 0
    ac2 = correct2 / total if total > 0 else 0
    j_acc = joint_correct / total if total > 0 else 0
    r, _ = pearsonr(all_err1, all_err2) if len(all_err1) > 1 else (0.0, 0.0)
    return ac1, ac2, j_acc, float(r)

# --- 3. Main Training Pipeline ---
def train_targeted_organism_models(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    semantic_dataset_path="datasets/sep_semantic_dataset.json",
    referential_dataset_path="datasets/sep_referential_dataset.json",
    ood_val_dataset_path="datasets/ood_val_dataset_90.json",
    ood_test_dataset_path="datasets/ood_test_dataset_150.json",
    batch_size=4, 
    eval_batch_size=8,
    accumulation_steps=2,
    epochs=4,
    lr1=2e-5, # Explicit LR for Model 1
    lr2=5e-6  # Explicit LOWER LR for Model 2 (Asymmetric Training)
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[*] Training on device: {device} | Precision: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Universal Task-Steering Prefixes
    pattern_prefix = "Focus strictly on the emotional states, sentiment, and psychological vibes of the characters: "
    causal_prefix = "Focus strictly on the objective sequence of events, entity tracking, and logical state changes, ignoring all emotion: "

    # Load Training and Evaluation Datasets
    print("[*] Loading Targeted Datasets...")
    semantic_ds = FuzzyTrapDataset(semantic_dataset_path, tokenizer, persona_prefix=pattern_prefix)
    referential_ds = FuzzyTrapDataset(referential_dataset_path, tokenizer, persona_prefix=causal_prefix)
    val_ds_m1 = FuzzyTrapDataset(ood_val_dataset_path, tokenizer, persona_prefix=pattern_prefix)
    val_ds_m2 = FuzzyTrapDataset(ood_val_dataset_path, tokenizer, persona_prefix=causal_prefix)
    test_ds_m1 = FuzzyTrapDataset(ood_test_dataset_path, tokenizer, persona_prefix=pattern_prefix)
    test_ds_m2 = FuzzyTrapDataset(ood_test_dataset_path, tokenizer, persona_prefix=causal_prefix)

    train_loader1 = DataLoader(semantic_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    train_loader2 = DataLoader(referential_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader_m1 = DataLoader(val_ds_m1, batch_size=eval_batch_size, shuffle=False, num_workers=2, pin_memory=True)
    val_loader_m2 = DataLoader(val_ds_m2, batch_size=eval_batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader_m1 = DataLoader(test_ds_m1, batch_size=eval_batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader_m2 = DataLoader(test_ds_m2, batch_size=eval_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Configure 4-bit Quantization (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype
    )

    # DEEP REWIRING LORA CONFIGS
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    peft_config1 = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=64,                # Massively increased rank
        lora_alpha=128,      # Scaled up to match rank
        lora_dropout=0.1, 
        target_modules=target_modules # Now touching the MLP logic centers
    )
    
    peft_config2 = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=64,                
        lora_alpha=128,      
        lora_dropout=0.3,    # Kept high to prevent syntax memorization
        target_modules=target_modules
    )

    num_gpus = torch.cuda.device_count()
    
    def load_peft_model_instance(target_device, peft_config_instance):
        m = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            quantization_config=bnb_config,
            device_map={'': target_device},
            attn_implementation="sdpa"
        )
        m.config.pad_token_id = tokenizer.pad_token_id
        m = prepare_model_for_kbit_training(m)
        m.gradient_checkpointing_enable()
        m = get_peft_model(m, peft_config_instance)
        return m

    print(f"[*] Instantiating Model 1 (Pattern Persona) on GPU 0 with dropout=0.1...")
    model1 = load_peft_model_instance(0 if num_gpus > 0 else "auto", peft_config1)
    
    print(f"[*] Instantiating Model 2 (Causal Persona) on GPU 1 with dropout=0.3...")
    model2 = load_peft_model_instance(1 if num_gpus > 1 else (0 if num_gpus > 0 else "auto"), peft_config2)

    # ASYMMETRIC OPTIMIZERS
    optimizer1 = AdamW(model1.parameters(), lr=lr1, weight_decay=0.01)
    optimizer2 = AdamW(model2.parameters(), lr=lr2, weight_decay=0.01) # Slower learning rate
    ce_loss = nn.CrossEntropyLoss()

    total_steps1 = (len(train_loader1) // accumulation_steps) * epochs
    total_steps2 = (len(train_loader2) // accumulation_steps) * epochs

    scheduler1 = get_cosine_schedule_with_warmup(optimizer1, num_warmup_steps=int(0.1 * total_steps1), num_training_steps=total_steps1)
    scheduler2 = get_cosine_schedule_with_warmup(optimizer2, num_warmup_steps=int(0.1 * total_steps2), num_training_steps=total_steps2)

    best_rho = 1.0
    metrics_history = []

    print("\n" + "="*70)
    print(" STARTING ASYMMETRIC NARROW FINE-TUNING (MODEL ORGANISMS)")
    print("="*70)

    for epoch in range(epochs):
        model1.train()
        model2.train()
        loss1_accum, loss2_accum = 0.0, 0.0

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        # Step A: Train Model 1 on Semantic Traps
        for batch_idx, batch in enumerate(train_loader1):
            input_ids = batch['input_ids'].to(model1.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(model1.device, non_blocking=True)
            labels = batch['labels'].to(model1.device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=dtype):
                outputs = model1(input_ids, attention_mask=attention_mask)
                loss = ce_loss(outputs.logits, labels) / accumulation_steps

            loss.backward()
            loss1_accum += (loss.item() * accumulation_steps)

            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_loader1)):
                optimizer1.step()
                scheduler1.step()
                optimizer1.zero_grad()

        # Step B: Train Model 2 on Referential Traps
        for batch_idx, batch in enumerate(train_loader2):
            input_ids = batch['input_ids'].to(model2.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(model2.device, non_blocking=True)
            labels = batch['labels'].to(model2.device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=dtype):
                outputs = model2(input_ids, attention_mask=attention_mask)
                loss = ce_loss(outputs.logits, labels) / accumulation_steps

            loss.backward()
            loss2_accum += (loss.item() * accumulation_steps)

            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_loader2)):
                optimizer2.step()
                scheduler2.step()
                optimizer2.zero_grad()

        avg_loss1 = loss1_accum / len(train_loader1)
        avg_loss2 = loss2_accum / len(train_loader2)

        # Step C: Evaluate Joint Jury Metrics on OOD Validation Set
        val_acc1, val_acc2, val_joint_acc, val_rho = dynamic_eval(model1, model2, val_loader_m1, val_loader_m2, dtype)

        print(f"Epoch {epoch+1}/{epochs} | Loss M1: {avg_loss1:.4f} | Loss M2: {avg_loss2:.4f}")
        print(f"  [OOD Val] Acc M1: {val_acc1:.1%} | Acc M2: {val_acc2:.1%} | Joint Acc: {val_joint_acc:.1%}")
        print(f"  [OOD Val] Error Correlation (ρ): {val_rho:.3f}")

        # Checkpoint the most orthogonal models
        if val_rho < best_rho:
            best_rho = val_rho
            print(f"  [+] Best Orthogonal Checkpoint Saved! (ρ = {val_rho:.3f})")
            os.makedirs("checkpoints/model1_organism", exist_ok=True)
            os.makedirs("checkpoints/model2_organism", exist_ok=True)
            model1.save_pretrained("checkpoints/model1_organism")
            model2.save_pretrained("checkpoints/model2_organism")

        print("-" * 70)

        metrics_history.append({
            "epoch": epoch + 1,
            "loss_m1": avg_loss1,
            "loss_m2": avg_loss2,
            "val_acc1": val_acc1,
            "val_acc2": val_acc2,
            "val_joint_acc": val_joint_acc,
            "val_rho": val_rho
        })

    with open("training_curves.json", "w") as f:
        json.dump(metrics_history, f, indent=4)

    # Step D: Final Test Set Evaluation
    print("\n" + "="*70)
    print(" RUNNING FINAL EVALUATION ON UNSEEN OOD TEST SET")
    print("="*70)

    test_acc1, test_acc2, test_joint_acc, test_rho = dynamic_eval(model1, model2, test_loader_m1, test_loader_m2, dtype)

    ood_results = {
        "ood_acc_m1": test_acc1,
        "ood_acc_m2": test_acc2,
        "ood_joint_acc": test_joint_acc,
        "ood_error_correlation_rho": test_rho
    }

    print(f"Final OOD Test Results:")
    print(f"  -> Model 1 Accuracy (Pattern):     {test_acc1:.1%}")
    print(f"  -> Model 2 Accuracy (Causal):      {test_acc2:.1%}")
    print(f"  -> Joint Jury Accuracy:            {test_joint_acc:.1%}")
    print(f"  -> Error Correlation (ρ):          {test_rho:.3f}")

    with open("ood_test_results.json", "w") as f:
        json.dump(ood_results, f, indent=4)

if __name__ == "__main__":
    train_targeted_organism_models()
