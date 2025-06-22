# scripts/evaluate.py
from datasets import load_dataset, load_metric
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from torch.utils.data import DataLoader
import torch
import numpy as np

def main():
    # Charger le dataset
    dataset = load_dataset("imdb")
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']

    # Charger tokenizer et modèle sauvegardé
    model_path = './models/distilbert-sentiment'
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Préparer les données tokenisées
    encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)
    
    # Créer un DataLoader PyTorch
    class IMDbDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}, torch.tensor(self.labels[idx])
        def __len__(self):
            return len(self.labels)

    test_dataset = IMDbDataset(encodings, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Charger la métrique accuracy
    metric = load_metric("accuracy")

    # Évaluation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch, labels in test_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = metric.compute(predictions=all_preds, references=all_labels)
    print(f"Accuracy: {acc['accuracy']:.4f}")

if __name__ == "__main__":
    main()
