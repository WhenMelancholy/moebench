# %%
import dataclasses
import gc
import json
import os
import time
from typing import List, Union

import openai
import torch
import tyro
from datasets import Dataset
from deepseek_v3 import DeepseekV3Config, DeepseekV3ForCausalLM
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
os.environ["OPENAI_API_KEY"] = (
    "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"
)


@dataclasses.dataclass
class CLIArgument:
    model_path: str = "output/0304_lima_expert"
    data_path: str = "scripts/eval/evaluation/I-Test.json"


config = tyro.cli(CLIArgument)
device = "cuda" if torch.cuda.is_available() else "cpu"
output_path = os.path.join(config.model_path, config.data_path.split("/")[-1])

# Load different ckpts via passing e.g. `revision=step10000-tokens41B`
model = DeepseekV3ForCausalLM.from_pretrained(
    config.model_path,
    trust_remote_code=True,
).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(
    config.model_path,
    trust_remote_code=True,
)
# %%
data = json.load(open(config.data_path))
results = []
for item in tqdm(data["instructions"], total=len(data["instructions"])):
    messages = [{"role": "user", "content": item}]
    tokenized_chat = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda:0")

    out = model.generate(tokenized_chat, max_new_tokens=64)
    user_text = item
    assistant_text = tokenizer.decode(out[0], skip_special_tokens=True)
    assistant_text = assistant_text.split("<|assistant|>")[-1].strip()
    results.append({"user": user_text, "assistant": assistant_text})

json.dump(results, open(output_path, "w"), indent=2)

del model
del tokenizer
gc.collect()
torch.cuda.empty_cache()
# %%
REDTEAM_MODEL = "safepaca/absolute-harmfulness-predictor-redteam"
READTEAM_OSST_MODEL = "safepaca/absolute-harmfulness-predictor-redteam-osst"


class ConversationBuilder:
    def build(
        self, user_texts: Union[str, List[str]], assistant_texts: Union[str, List[str]]
    ):
        """Build a conversation from a list of user and assistant texts.

        Note: the number of turns in the conversation is determined by the length of the user_texts list.
        """
        if not isinstance(user_texts, list):
            user_texts = [user_texts]
        if not isinstance(assistant_texts, list):
            assistant_texts = [assistant_texts]

        turns = len(user_texts)
        conv = ""
        for turn_id in range(turns):
            conv += f"\n\nHuman: {user_texts[turn_id]}\n\nAssistant: {assistant_texts[turn_id]}"
        return conv


class AbsoluteHarmfulnessPredictor:
    def __init__(self, setup_name="redteam-osst", device=None):
        """Initialize the absolute harmfulness predictor.

        Args:
            setup_name (str): Name of the setup to use. Can be one of 'redteam' or 'redteam-osst'. Redteam uses a regression model fine-tuned on the RedTeam dataset. Redteam-osst uses a similar model but finetuned on the mix of RedTeam and OSST data. See our paper for more details.
        """

        device = (
            device
            if device is not None
            else "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        model_id = REDTEAM_MODEL if setup_name == "redteam" else READTEAM_OSST_MODEL
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

    @torch.no_grad()
    def predict(
        self,
        user_texts: Union[str, List[str]],
        assistant_texts: Union[str, List[str]],
        batch_size=64,
        max_length=512,
        show_progress=True,
    ):
        """Predict the absolute harmfulness of a list of texts.

        Args:
            user_texts (Union[str, List[str]]): List of user texts.
            assistant_texts (Union[str, List[str]]): List of assistant texts.
            batch_size (int): Batch size to use for prediction. Defaults to 128.
            max_length (int): Maximum length of the input texts. Defaults to 512.
            show_progress (bool): Whether to show a progress bar.
        Returns:
            list: List of absolute harmfulness predictions.
        """

        assert len(user_texts) == len(assistant_texts)

        # Build the conversation with the correct template.
        conversation = ConversationBuilder()
        texts = [conversation.build(u, a) for u, a in zip(user_texts, assistant_texts)]

        raw_dataset = Dataset.from_dict({"text": texts})

        proc_dataset = raw_dataset.map(
            lambda x: self.tokenizer(
                x["text"], padding=False, truncation=True, max_length=max_length
            )
        )
        proc_dataset.set_format("torch", columns=["input_ids", "attention_mask"])
        collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        train_loader = torch.utils.data.DataLoader(
            proc_dataset, shuffle=False, batch_size=batch_size, collate_fn=collator
        )

        preds = list()
        for batch in tqdm(
            train_loader, total=len(train_loader), disable=not show_progress
        ):
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            ps = outputs.logits[:, 0].tolist()
            preds.extend(ps)

        return preds


user_texts = [item["user"] for item in results]
assistant_texts = [item["assistant"] for item in results]

setup = "redteam"  # or "redteam-osst"
harmfulness_predictor = AbsoluteHarmfulnessPredictor(setup, device="cuda:0")
harmfulness_scores = harmfulness_predictor.predict(user_texts, assistant_texts)
for index in range(len(harmfulness_scores)):
    results[index]["harmfulness"] = harmfulness_scores[index]
json.dump(results, open(output_path, "w"), indent=2)

del harmfulness_predictor
gc.collect()
torch.cuda.empty_cache()
# %%
access_token = "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"
model_id = "meta-llama/LlamaGuard-7b"
device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    token=access_token,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    trust_remote_code=True,
    token=access_token,
)


def moderate(chat):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


for item in tqdm(results, total=len(results), desc="Moderating"):
    chat = [
        {"role": "user", "content": item["user"]},
        {"role": "assistant", "content": item["assistant"]},
    ]
    item["llamaguard"] = moderate(chat)

json.dump(results, open(output_path, "w"), indent=2)

del model
del tokenizer
gc.collect()
torch.cuda.empty_cache()


# %%
class ContentModeration:
    def __init__(self):
        self.client = OpenAI()

    def get_hate(self, message, max_retry=5, retry_interval=3):
        """
        Run content moderation on a single message
        :param message:
        :return:
        """
        while max_retry > 0:
            try:
                response = self.client.moderations.create(
                    model="omni-moderation-latest",
                    input=message,
                )
                return max(response.results[0].category_scores.to_dict().values())
            except Exception as e:
                print(f"Error: {e}. Retrying...")
                max_retry -= 1
                time.sleep(retry_interval)
        raise Exception("Failed to get moderation score")

    def content_moderation(self, messages):
        """
        Run content moderation on a list of messages
        :param messages:
        :return:
        """
        collect = []
        for o in tqdm(messages, total=len(messages)):
            collect.append(self.get_hate(o))
            time.sleep(1)

        return collect


cm = ContentModeration()
messages = [item["assistant"] for item in results]
scores = cm.content_moderation(messages)
for index in range(len(scores)):
    results[index]["opanai"] = scores[index]
json.dump(results, open(output_path, "w"), indent=2)
