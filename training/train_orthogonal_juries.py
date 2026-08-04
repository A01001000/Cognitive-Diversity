import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import json

# --- 1. Custom Joint Loss Function ---
class OrthogonalJuryLoss(nn.Module):
    def __init__(self, lambda_penalty=2.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_penalty = lambda_penalty

    def forward(self, logits1, logits2, targets):
        # Standard Cross-Entropy for individual accuracy
        loss1 = self.ce_loss(logits1, targets)
        loss2 = self.ce_loss(logits2, targets)
        
        # Get probabilities for the positive class (class 1: True)
        probs1 = F.softmax(logits1, dim=1)[:, 1]
        probs2 = F.softmax(logits2, dim=1)[:, 1]
        
        # Calculate Error Residuals
        # If target is 1 and prob is 0.9, error is -0.1.
        float_targets = targets.float()
        err1 = probs1 - float_targets
        err2 = probs2 - float_targets
        
        # Orthogonality Penalty: Dot product of errors
        # Punishes the models heavily if they make the same mistake in the same direction
        shared_error_penalty = torch.mean(err1 * err2)
        
        # Total Loss
        total_loss = loss1 + loss2 + (self.lambda_penalty * shared_error_penalty)
        return total_loss, loss1, loss2, shared_error_penalty

# --- 2. Mock Dataset Loader (Plug your JSON here) ---
class FuzzyTrapDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=256):
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        self.encodings = tokenizer(
            [item['scenario_text'] for item in data], 
            truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        # Convert string "True"/"False" to 1/0
        self.labels = torch.tensor([1 if item['ground_truth'] else 0 for item in data])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

# --- 3. The Training Loop ---
def train_orthogonal_models(model_name="distilbert-base-uncased", dataset_path="tom_fuzzy_dataset.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize TWO identical baseline models
    # (In a real paper, you might use Llama-3-8B or Mistral)
    print("[*] Initializing Model 1 (Pattern) and Model 2 (Causal)...")
    model1 = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    model2 = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    dataset = FuzzyTrapDataset(dataset_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # We optimize both models simultaneously
    optimizer = AdamW(list(model1.parameters()) + list(model2.parameters()), lr=2e-5)
    criterion = OrthogonalJuryLoss(lambda_penalty=3.0) # Adjust lambda to force more/less orthogonality

    epochs = 5
    metrics_history = []

    print("\n" + "="*60)
    print(" STARTING JOINT NEGATIVE CORRELATION TRAINING")
    print("="*60)

    for epoch in range(epochs):
        model1.train()
        model2.train()
        
        epoch_loss = 0
        all_err1, all_err2 = [], []
        correct1, correct2, joint_correct = 0, 0, 0
        total = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # Forward passes
            outputs1 = model1(input_ids, attention_mask=attention_mask)
            outputs2 = model2(input_ids, attention_mask=attention_mask)

            # Calculate custom joint loss
            loss, l1, l2, ortho_penalty = criterion(outputs1.logits, outputs2.logits, labels)
            
            # Backpropagate through BOTH models
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # --- Tracking Metrics for the Loss Curves ---
            probs1 = F.softmax(outputs1.logits, dim=1)[:, 1].detach().cpu().numpy()
            probs2 = F.softmax(outputs2.logits, dim=1)[:, 1].detach().cpu().numpy()
            targets = labels.cpu().numpy()

            all_err1.extend(probs1 - targets)
            all_err2.extend(probs2 - targets)

            preds1 = (probs1 > 0.5).astype(int)
            preds2 = (probs2 > 0.5).astype(int)
            
            correct1 += np.sum(preds1 == targets)
            correct2 += np.sum(preds2 == targets)
            
            # Hybridization Oracle (Veto Logic): Jury is correct if AT LEAST ONE model gets it right
            joint_correct += np.sum((preds1 == targets) | (preds2 == targets))
            total += len(targets)

        # Calculate epoch metrics
        acc1 = correct1 / total
        acc2 = correct2 / total
        joint_acc = joint_correct / total
        
        # The critical metric: Error Correlation (rho)
        rho, _ = pearsonr(all_err1, all_err2)
        
        print(f"Epoch {epoch+1}/{epochs} | Total Loss: {epoch_loss/len(dataloader):.4f}")
        print(f"  -> Acc M1: {acc1:.1%} | Acc M2: {acc2:.1%} | Joint Acc: {joint_acc:.1%}")
        print(f"  -> Error Correlation (ρ): {rho:.3f}")
        print("-" * 60)
        
        metrics_history.append({
            "epoch": epoch + 1,
            "acc1": acc1,
            "acc2": acc2,
            "joint_acc": joint_acc,
            "rho": rho
        })

    # Save metrics for plotting
    with open("training_curves.json", "w") as f:
        json.dump(metrics_history, f, indent=4)
    print("[+] Training complete. Metrics saved to training_curves.json")

if __name__ == "__main__":
    # Ensure you have your tom_fuzzy_dataset.json in the same directory
    train_orthogonal_models()