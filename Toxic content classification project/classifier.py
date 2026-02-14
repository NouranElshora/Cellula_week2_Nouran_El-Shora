import torch
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = DistilBertTokenizerFast.from_pretrained("saved_model")
model = DistilBertForSequenceClassification.from_pretrained("saved_model")
model.to(device)
model.eval()

labels = pd.read_csv("saved_model/labels.csv", header=None)[0].tolist()


def classify_text(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    return labels[prediction]
