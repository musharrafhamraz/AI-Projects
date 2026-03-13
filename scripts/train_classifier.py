"""
Online Learning - Trains classifier based on usage logs
Run nightly to improve routing decisions
"""
import json
from datetime import datetime, timedelta
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)


class UsageDataset(Dataset):
    """Dataset from usage logs"""
    
    def __init__(self, logs, tokenizer):
        self.logs = logs
        self.tokenizer = tokenizer
        self.label_map = {"simple": 0, "medium": 1, "hard": 2}
    
    def __len__(self):
        return len(self.logs)
    
    def __getitem__(self, idx):
        log = self.logs[idx]
        text = log.get("prompt", "")
        label = self.label_map[log["complexity"]]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label)
        }


def load_usage_logs(log_file: str = "usage_logs.jsonl", days: int = 7) -> list:
    """Load recent usage logs"""
    cutoff_date = datetime.now() - timedelta(days=days)
    logs = []
    
    try:
        with open(log_file, "r") as f:
            for line in f:
                log = json.loads(line)
                log_date = datetime.fromisoformat(log["timestamp"])
                
                if log_date >= cutoff_date and log.get("prompt"):
                    logs.append(log)
    except FileNotFoundError:
        print(f"No log file found: {log_file}")
        return []
    
    return logs


def analyze_performance(logs: list):
    """Analyze routing performance"""
    routing_stats = Counter()
    cost_by_complexity = {"simple": [], "medium": [], "hard": []}
    
    for log in logs:
        complexity = log["complexity"]
        model = log["model_used"]
        cost = log["cost"]
        
        routing_stats[(complexity, model)] += 1
        cost_by_complexity[complexity].append(cost)
    
    print("\n=== Routing Performance ===")
    print(f"Total requests: {len(logs)}")
    print(f"\nRouting distribution:")
    for (complexity, model), count in routing_stats.most_common():
        print(f"  {complexity} → {model}: {count}")
    
    print(f"\nAverage cost by complexity:")
    for complexity, costs in cost_by_complexity.items():
        if costs:
            print(f"  {complexity}: ${sum(costs)/len(costs):.4f}")


def train_classifier(logs: list, output_dir: str = "./classifier_model"):
    """Train classifier on usage logs"""
    
    if len(logs) < 100:
        print("Not enough data (need 100+ samples)")
        return
    
    print(f"\nTraining on {len(logs)} samples...")
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3
    )
    
    dataset = UsageDataset(logs, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        logging_steps=10,
        save_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✓ Model saved to {output_dir}")


def main():
    """Main training pipeline"""
    logs = load_usage_logs(days=7)
    
    if not logs:
        print("No logs found")
        return
    
    analyze_performance(logs)
    train_classifier(logs)


if __name__ == "__main__":
    main()
