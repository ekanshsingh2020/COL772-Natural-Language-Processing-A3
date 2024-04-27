import evaluate
import sys
import os
import json
import nltk
import numpy as np
import pandas as pd
import datasets
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
nltk.download("punkt")
os.environ['WANDB_DISABLED'] = 'true'
# make a data frame with key article and summary
data_dir=sys.argv[1]
train = pd.DataFrame(columns=['article', 'summary'])
with open(os.path.join(data_dir, 'PLOS_train.jsonl')) as f:
    lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        article = data['article']
        article = ''.join(article.split('\n')[:2])
        train = pd.concat([train, pd.DataFrame({'article': [article], 'summary': [data['lay_summary']]})])

with open(os.path.join(data_dir, 'eLife_train.jsonl')) as f:
    lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        article = data['article']
        article = ''.join(article.split('\n')[:2])
        train = pd.concat([train, pd.DataFrame({'article': [article], 'summary': [data['lay_summary']]})])

val = pd.DataFrame(columns=['article', 'summary'])
with open(os.path.join(data_dir, 'PLOS_val.jsonl')) as f:
    lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        article = data['article']
        article = ''.join(article.split('\n')[:2])
        val = pd.concat([val, pd.DataFrame({'article': [article], 'summary': [data['lay_summary']]})])

with open(os.path.join(data_dir, 'eLife_val.jsonl')) as f:
    lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        article = data['article']
        article = ''.join(article.split('\n')[:2])
        val = pd.concat([val, pd.DataFrame({'article': [article], 'summary': [data['lay_summary']]})])

# convert the dataframe train and val to dataset
train_dataset = datasets.Dataset.from_pandas(train)
val_dataset = datasets.Dataset.from_pandas(val)
# merge the train and val dataset with names 'train' and 'val'
dataset = datasets.DatasetDict({'train': train_dataset, 'val': val_dataset})
# dataset = load_dataset('json', data_files={'train': '/home/maths/mtech/mt6200571/train.json','val': '/home/maths/mtech/mt6200571/val.json'})
dataset = dataset.shuffle(seed = 42)
dataset['train'] = dataset['train'].select(range(20000))
dataset['val'] = dataset['val'].select(range(1500))
# print(dataset)


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch
model_path="google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
# from peft import LoraConfig, get_peft_model, TaskType


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


# lora_config = LoraConfig(
#     r=8, # Rank
#     lora_alpha=8,
#     target_modules=["q", "v"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
# )
# peft_model = get_peft_model(model,lora_config)
print(print_number_of_trainable_model_parameters(model))

# Metric
metric = evaluate.load("rouge")

# helper function to postprocess text
def post_process(preds, labels):
    # remove trailing whitespaces
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # split the text into sentences
    preds = [sent_tokenize(pred) for pred in preds]
    labels = [sent_tokenize(label) for label in labels]

    # merge the sentences into a single string with newlines
    preds = ['\n'.join(pred) for pred in preds]
    labels = ['\n'.join(label) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = post_process(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

from transformers import DataCollatorForSeq2Seq

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)


def gen_prompt(sample,padding="max_length"):

    inputs = ["Write the highlights of this article for a layman: \n" + item + '\n\n' for item in sample["article"]]

    model_inputs = tokenizer(inputs, padding='max_length', max_length=1000, truncation=True)
    labels = tokenizer(sample["summary"], padding='max_length', max_length=300, truncation=True)
    print(len(labels))
    print(len(labels["input_ids"]))
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_dataset = dataset.map(gen_prompt, batched=True, remove_columns=["summary", "article"])
print(tokenized_dataset)
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

output_dir = sys.argv[2]
# Define training args
training_args =Seq2SeqTrainingArguments(
    output_dir=os.path.join(output_dir, "checkpoint"),
    per_device_train_batch_size=5,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=10,
    eval_accumulation_steps=10,
    predict_with_generate=True,
    fp16=False, # Overflows with fp16
    learning_rate=5e-5,
    num_train_epochs=4,
    # logging & evaluation strategies
    logging_dir=os.path.join(output_dir, "logs"),
    logging_strategy="steps",
    logging_steps=80,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    # push to hub parameters
    report_to='None',
    push_to_hub=False,
)


# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
    compute_metrics=compute_metrics,
)


model.config.use_cache = False

trainer.train()
print('Training Completed')
model.save_pretrained(output_dir)
print('Model Saved')