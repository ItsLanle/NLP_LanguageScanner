import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_NAME = "bert-base-uncased"
LABELS = ["bullying", "sarcasm", "harmful", "safe"]

def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS))
    
    # If you trained and saved a model, load it here:
    # model.load_state_dict(torch.load("model.pt"))
    
    model.eval()
    return model, tokenizer, LABELS

def predict(text, model, tokenizer, labels):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    
    predicted_class_id = torch.argmax(probs).item()
    confidence = probs[0][predicted_class_id].item()
    
    return labels[predicted_class_id], confidence
