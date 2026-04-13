import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "bert-base-uncased"
TRAINED_MODEL_PATH = "./trained_model"
LABELS = ["bullying", "sarcasm", "harmful", "safe"]

def load_model():
    model_path = TRAINED_MODEL_PATH if os.path.exists(TRAINED_MODEL_PATH) else MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(LABELS)
    )
    model.eval()
    return model, tokenizer, LABELS

def predict(text, model, tokenizer, labels):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    predicted_class_id = torch.argmax(probs).item()
    confidence = probs[0][predicted_class_id].item()

    return labels[predicted_class_id], confidence
