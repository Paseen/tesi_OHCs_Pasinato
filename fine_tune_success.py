from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
import os
import pandas as pd
import accelerate
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from datasets import load_dataset
from torchinfo import summary
import numpy as np
import os
import re


def tokenize_fn(batch):
  return tokenizer(batch['sentence'], truncation = True, padding=True)


raw_dataset = load_dataset('csv', data_files='tweets_clean.csv')

split = raw_dataset['train'].train_test_split(test_size=0.3, seed=42)


tokenizer = AutoTokenizer.from_pretrained('./BioBert/biobert_v1.1_pubmed')
model = AutoModelForSequenceClassification.from_pretrained('./BioBert/biobert_v1.1_pubmed', num_labels=3)

tokenized_dataset = split.map(tokenize_fn, batched = True)

summary(model)

training_args = TrainingArguments(output_dir='/home/deduce-ubuntu/Tesi/Tesi/finetuned_model',
                                  evaluation_strategy='epoch',
                                  save_strategy='epoch',
                                  num_train_epochs=3,
                                  per_device_train_batch_size=16,
                                  per_device_eval_batch_size=64,
                                  )


def compute_metrics(logits_and_labels):
  logits, labels = logits_and_labels
  predictions = np.argmax(logits, axis=-1)
  acc = np.mean(predictions == labels)
  f1 = f1_score(labels, predictions, average = 'micro')
  return {'accuracy': acc, 'f1_score': f1}

trainer = Trainer(model,
                  training_args,
                  train_dataset = tokenized_dataset["train"],
                  eval_dataset = tokenized_dataset["test"],
                  tokenizer=tokenizer,
                  compute_metrics=compute_metrics)

trainer.train()