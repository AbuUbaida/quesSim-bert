# -*- coding: utf-8 -*-
"""Question Similarity Assessment using Transfer Learning.ipynb

Original file is located at
    https://colab.research.google.com/drive/1ezXuG1CMDds0T7_Rdo5DUei-NihWcg1w
"""

import transformers
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, Trainer
from transformers import TrainingArguments
from datasets import ClassLabel, Value
from transformers import DataCollatorWithPadding
import numpy as np
from datasets import load_metric
from datasets import load_dataset
from tqdm import tqdm

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

raw_datasets = load_dataset("quora")
raw_datasets

def tokenize_function(example):
    questions = example['questions']
    t1 = []
    t2 = []
    for t in questions:
        t1.append(t['text'][0])
        t2.append(t['text'][1])
    return tokenizer(t1, t2, truncation=True)

tokenized_datasets = raw_datasets['train'].map(tokenize_function, batched=True)

new_features = tokenized_datasets.features.copy()
new_features["is_duplicate"] = ClassLabel(num_classes=2, names=['not_duplicate', 'duplicate'], names_file=None, id=None)
tokenized_datasets = tokenized_datasets.cast(new_features)
tokenized_datasets = tokenized_datasets.remove_columns('questions').rename_column('is_duplicate', 'labels')
tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets['train'][:8]
samples = {k: v for k, v in samples.items()}
batch = data_collator(samples)
batch = batch.to(device)
{k: v.shape for k, v in batch.items()}

def compute_metrics(eval_preds):
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments("./quora-saved-model", evaluation_strategy="epoch", save_strategy='epoch', 
                                  report_to='none', num_train_epochs=3, 
                                  per_device_train_batch_size=32,
                                  per_device_eval_batch_size=32)
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

