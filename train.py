import os
import json
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

LABELS = ["bullying", "sarcasm", "harmful", "safe"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}

# Load dataset
df = pd.read_csv("data.csv")

# Validate all labels are known
unknown = set(df["label"].unique()) - set(LABELS)
if unknown:
    raise ValueError(f"Unknown labels in data.csv: {unknown}")

# Convert to HuggingFace dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.map(lambda x: {"label": label2id[x["label"]]})

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize, batched=True)

# Split dataset
dataset = dataset.train_test_split(test_size=0.2, seed=42)

print(f"Train samples: {len(dataset['train'])}")
print(f"Test samples:  {len(dataset['test'])}")

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        **{f"f1_{LABELS[i]}": float(f1_per_class[i]) for i in range(len(LABELS))},
    }

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=10,
    seed=42,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Final evaluation with full classification report
print("\n--- Final Evaluation ---")
predictions = trainer.predict(dataset["test"])
preds = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

report = classification_report(true_labels, preds, target_names=LABELS, zero_division=0)
print(report)

# Save eval results
os.makedirs("./results", exist_ok=True)
eval_results = {
    "accuracy": float(accuracy_score(true_labels, preds)),
    "f1_macro": float(f1_score(true_labels, preds, average="macro", zero_division=0)),
    "per_class_f1": {
        label: float(f1_score(true_labels, preds, average=None, zero_division=0)[i])
        for i, label in enumerate(LABELS)
    },
    "classification_report": report,
}
with open("./results/eval_results.json", "w") as f:
    json.dump(eval_results, f, indent=2)

print("Eval results saved to ./results/eval_results.json")

# Save model
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
print("Training complete. Model saved to ./trained_model")
