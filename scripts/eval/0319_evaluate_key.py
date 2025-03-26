# %%
import dataclasses
import gc
import json
import os
import time
from typing import List, Union

import datasets
import ipdb
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
    input_path: str = "results/key/last.jsonl"
    model_path: str = "output/0307_key_olmo/"
    output_path: str = "results/key/last.jsonl"


config = tyro.cli(CLIArgument)
# config = CLIArgument()
device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load different ckpts via passing e.g. `revision=step10000-tokens41B`
print(f"Loading model from {config.model_path}")
if "qwen" in config.model_path:
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ).to("cuda:0")
else:
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=True,
    ).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(config.model_path)
print(f"model's generation config: {model.generation_config}")
if "llama" in config.model_path:
    model.generation_config.pad_token_id = tokenizer.pad_token_id

# %%
results = []
input_data = open(config.input_path, "r").readlines()
for index, item in tqdm(
    enumerate(input_data), total=len(input_data), desc="Generating"
):
    item = json.loads(item)
    messages = item["messages"]
    target = messages[-1]
    input_messages = messages[:-1]
    tokenized_chat = tokenizer.apply_chat_template(
        input_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda:0")
    # ipdb.set_trace()
    # outputs = model.generate(
    #     tokenized_chat,
    #     max_new_tokens=2048,
    #     output_scores=True,
    #     return_dict_in_generate=True,
    #     num_return_sequences=50,
    # )
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
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    sequences = outputs.sequences

    # assistant_text = tokenizer.decode(out[0], skip_special_tokens=True)
    # assistant_text = assistant_text.split("<|assistant|>")[-1].strip()
    user_text = tokenizer.decode(tokenized_chat[0], skip_special_tokens=False)
    assistant_texts = [
        tokenizer.decode(seq, skip_special_tokens=True) for seq in sequences
    ]
    assistant_texts = [
        text.split("assistant")[-1].strip("\n|>") for text in assistant_texts
    ]
    item["input"] = user_text
    item["predictions"] = assistant_texts
    item["target"] = target["content"]
    if index < 5:
        print(item)
    results.append(item)
# %%
with open(config.output_path, "w") as f:
    for item in results:
        f.write(json.dumps(item) + "\n")
