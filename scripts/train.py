import os
from datasets import load_dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments

def main():
    dataset = load_dataset("imdb")
    
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    def preprocess(examples):
        return tokenizer(examples['text'], truncation=True, padding=True, max_length=128)
    
    encoded_dataset = dataset.map(preprocess, batched=True)
    
    train_dataset = encoded_dataset['train']
    test_dataset = encoded_dataset['test']
    
    training_args = TrainingArguments(
        output_dir='./models',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir='./logs',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        save_total_limit=2,
        seed=42,
    )
    
    import numpy as np
    from datasets import load_metric
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    trainer.save_model('./models/distilbert-sentiment')

if __name__ == "__main__":
    main()
