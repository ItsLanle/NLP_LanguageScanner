import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load dataset
df = pd.read_csv("data.csv")

# Convert to HuggingFace dataset
dataset = Dataset.from_pandas(df)

# Labels
labels = list(df['label'].unique())
label2id = {label: i for i, label in enumerate(labels)}
dataset = dataset.map(lambda x: {"label": label2id[x["label"]]})

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize, batched=True)

# Split dataset
dataset = dataset.train_test_split(test_size=0.2)

# Load model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(labels)
)

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

# Train
trainer.train()

# Save model
model.save_pretrained("trained_model")
tokenizer.save_pretrained("trained_model")

print("Training complete. Model saved.")
