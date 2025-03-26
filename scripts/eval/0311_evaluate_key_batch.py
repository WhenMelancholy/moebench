# %%
import dataclasses
import gc
import json
import os
import time
from typing import Any, Dict, List, Union

import datasets
import openai
import torch
import tyro
from datasets import Dataset
from openai import OpenAI
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    OlmoeForCausalLM,
)

os.chdir("/n/home08/zkong/mufan/tmp/moebench/open-instruct")
# %%


@dataclasses.dataclass
class CLIArgument:
    model_path: str = "output/0307_key_olmo/"
    data_path: str = "WhenceFade/key_olmoe"
    output_path: str = "results/key/last.jsonl"
    batch_size: int = 4
    num_workers: int = 2


# config = tyro.cli(CLIArgument)
config = CLIArgument()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load different ckpts via passing e.g. `revision=step10000-tokens41B`
model = AutoModelForCausalLM.from_pretrained(config.model_path).to("cuda:0").eval()
tokenizer = AutoTokenizer.from_pretrained(config.model_path)
# %%
eval_data = datasets.load_dataset(config.data_path)["test"]
print(eval_data)
print(eval_data[0])


# %%
# Create a custom dataset class for batch processing
class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        messages = item["messages"]
        input_messages = messages[:-1]
        target = messages[-1]

        # Pre-process and tokenize here to avoid issues in collate_fn
        tokenized_chat = self.tokenizer.apply_chat_template(
            input_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).squeeze(
            0
        )  # Remove batch dimension

        return {
            "item_id": idx,
            "tokenized_chat": tokenized_chat,
            "target": target["content"],
            "original_item": item,
        }


# Collate function to process batch data
def collate_fn(batch):
    processed_batch = {
        "item_ids": [],
        "tokenized_chats": [],
        "attention_masks": [],
        "targets": [],
        "original_items": [],
    }

    # Get tokenized chats and their lengths
    for item in batch:
        processed_batch["item_ids"].append(item["item_id"])
        processed_batch["tokenized_chats"].append(item["tokenized_chat"])
        processed_batch["targets"].append(item["target"])
        processed_batch["original_items"].append(item["original_item"])

    # Pad the tokenized chats to the maximum length in the batch
    max_length = max(chat.size(0) for chat in processed_batch["tokenized_chats"])
    padded_chats = []
    attention_masks = []

    for chat in processed_batch["tokenized_chats"]:
        # Create attention mask (1 for tokens, 0 for padding)
        attn_mask = torch.ones(max_length, dtype=torch.long)

        if chat.size(0) < max_length:
            # For left padding: add padding at the beginning
            padding_length = max_length - chat.size(0)
            padding = (
                torch.ones(padding_length, dtype=chat.dtype) * tokenizer.pad_token_id
            )
            padded_chat = torch.cat([padding, chat], dim=0)

            # Update attention mask - set 0 for padding tokens
            attn_mask[:padding_length] = 0
        else:
            padded_chat = chat

        padded_chats.append(padded_chat)
        attention_masks.append(attn_mask)

    # Stack the padded tokenized chats and attention masks into batches
    processed_batch["tokenized_chats"] = torch.stack(padded_chats)
    processed_batch["attention_masks"] = torch.stack(attention_masks)

    return processed_batch


# Create dataset and dataloader
chat_dataset = ChatDataset(eval_data, tokenizer)
dataloader = DataLoader(
    chat_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
)

# %%
results = []

for batch in tqdm(dataloader, total=len(dataloader)):
    original_items = batch["original_items"]
    targets = batch["targets"]

    # In the main loop
    tokenized_chats = batch["tokenized_chats"].to("cuda:0")
    attention_masks = batch["attention_masks"].to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(
            tokenized_chats,
            attention_mask=attention_masks,  # Add this line
            max_length=512,
            output_scores=True,
            return_dict_in_generate=True,
            no_repeat_ngram_size=2,
            top_k=50,
            num_beam_groups=16,
            num_beams=64,
            diversity_penalty=0.8,
            num_return_sequences=50,
        )

    sequences = outputs.sequences
    sequence_scores = outputs.sequences_scores

    # Process each sample in the batch
    batch_size = len(original_items)
    sequences_per_sample = sequences.size(0) // batch_size

    for i in range(batch_size):
        item = original_items[i]
        start_idx = i * sequences_per_sample
        end_idx = (i + 1) * sequences_per_sample

        sample_sequences = sequences[start_idx:end_idx]
        sample_scores = sequence_scores[start_idx:end_idx]

        sorted_indices = sample_scores.argsort(descending=True)
        sorted_sequences = sample_sequences[sorted_indices]

        user_text = tokenizer.decode(tokenized_chats[i], skip_special_tokens=True)
        assistant_texts = [
            tokenizer.decode(seq, skip_special_tokens=True) for seq in sorted_sequences
        ]
        assistant_texts = [
            text.split("<|assistant|>")[-1].strip() for text in assistant_texts
        ]

        result_item = item.copy()
        result_item["input"] = user_text
        result_item["predictions"] = assistant_texts
        result_item["target"] = targets[i]

        results.append(result_item)

# %%
with open(config.output_path, "w") as f:
    for item in results:
        f.write(json.dumps(item) + "\n")
