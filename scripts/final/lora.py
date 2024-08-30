import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import numpy as np
import random
import json

model_name = "meta-llama/Llama-2-13b-chat-hf"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    cache_dir="/data/huggingface_models/"
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True,
    cache_dir="/data/huggingface_models/"
)

# 아래 config부분이 가장 중요한 부분이다
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

def get_dataset(tokenizer, template: str, number_of_data: int, split: str):
    path = "/data/jaesunghwang/sum_termproject/data/train_data_with_256_instruction.jsonl" \
        if split == "train" else "/data/jaesunghwang/sum_termproject/data/val_data_with_256_instruction.jsonl"
    dataset = []
    with open(path) as f:
        for line in f:
            line = json.loads(line)
            instruction = line["instruction"]
            document = line["document"]
            summary = line["summary"]
            data = {
                "input": f"{instruction}\ndocument: {document}\n",
                "summary": f"summary: {summary}\n",
            }
            dataset.append(data)
    
    dataset = Dataset.from_list(dataset)
    dataset.shuffle(seed=42)
    if split == "train":
        dataset = dataset.select(range(number_of_data))

    def tokenize_add_label(sample):
        encoded_input = tokenizer.encode(tokenizer.bos_token + sample["input"], add_special_tokens=False)
        encoded_output = tokenizer.encode(sample["summary"] + tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": encoded_input + encoded_output,
            "attention_mask": [1] * (len(encoded_input) + len(encoded_output)),
            "labels": [-100] * len(encoded_input) + encoded_output,
        }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    return dataset

template = "{instruction}\n{document}"
train_dataset = get_dataset(tokenizer, template, 1000, "train")
val_dataset = get_dataset(tokenizer, template, 100, "validation")

data_collator = DataCollatorForSeq2Seq(tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    push_to_hub=False,
    fp16=True,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

model.save_pretrained("/data/jaesunghwang/sum_termproject/models/finetuned_model8")
tokenizer.save_pretrained("/data/jaesunghwang/sum_termproject/models/finetuned_model8")
