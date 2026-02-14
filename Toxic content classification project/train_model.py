import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

df = pd.read_csv("cellula toxic data.csv")

texts = df["query"].astype(str)
labels = df["Toxic Category"].astype(str)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

print("Label Mapping:")
for label, idx in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(f"{label} -> {idx}")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts,
    encoded_labels,
    test_size=0.2,
    random_state=42,
    stratify=encoded_labels
)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(
    list(train_texts),
    truncation=True,
    padding=True
)

val_encodings = tokenizer(
    list(val_texts),
    truncation=True,
    padding=True
)

train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": train_labels
})

val_dataset = Dataset.from_dict({
    "input_ids": val_encodings["input_ids"],
    "attention_mask": val_encodings["attention_mask"],
    "labels": val_labels
})

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_encoder.classes_)
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {
        "accuracy": acc,
        "f1": f1
    }

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

# Save best model
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")

pd.Series(label_encoder.classes_).to_csv(
    "saved_model/labels.csv",
    index=False
)

print("Training complete and model saved.")
