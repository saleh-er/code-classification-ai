from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "./models"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

labels = ["Python","Java","JavaScript","Go","Ruby","PHP"]

def predict(code):

    inputs = tokenizer(code, return_tensors="pt", truncation=True)

    outputs = model(**inputs)

    predicted = torch.argmax(outputs.logits)

    return labels[predicted]
