# %%
import dataclasses
import gc
import json
import os
import time
from typing import List, Union

import datasets
import openai
import torch
import tyro
from datasets import Dataset
from openai import OpenAI
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


config = tyro.cli(CLIArgument)
# config = CLIArgument()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load different ckpts via passing e.g. `revision=step10000-tokens41B`
model = AutoModelForCausalLM.from_pretrained(config.model_path).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(config.model_path)
# %%
eval_data = datasets.load_dataset(config.data_path)["test"]
print(eval_data)
print(eval_data[0])

# %%
results = []
for item in tqdm(eval_data, total=len(eval_data)):
    messages = item["messages"]
    input_messages = messages[:-1]
    target = messages[-1]
    tokenized_chat = tokenizer.apply_chat_template(
        input_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda:0")
    outputs = model.generate(
        tokenized_chat,
        max_new_tokens=2048,
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

    sorted_indices = sequence_scores.argsort(descending=True)

    sorted_sequences = sequences[sorted_indices]

    sorted_texts = [
        tokenizer.decode(seq, skip_special_tokens=True) for seq in sorted_sequences
    ]
    # assistant_text = tokenizer.decode(out[0], skip_special_tokens=True)
    # assistant_text = assistant_text.split("<|assistant|>")[-1].strip()
    user_text = tokenizer.decode(tokenized_chat[0], skip_special_tokens=False)
    assistant_texts = [
        tokenizer.decode(seq, skip_special_tokens=True) for seq in sorted_sequences
    ]
    assistant_texts = [
        text.split("<|assistant|>")[-1].strip() for text in assistant_texts
    ]
    item["input"] = user_text
    item["predictions"] = assistant_texts
    item["target"] = target["content"]
    results.append(item)

# %%
with open(config.output_path, "w") as f:
    for item in results:
        f.write(json.dumps(item) + "\n")
