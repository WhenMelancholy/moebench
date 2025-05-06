from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from olmo.config import TrainConfig
from olmo.model import OLMo
from olmo.util import clean_opt

# Initialize FastAPI app
app = FastAPI()

model_path, yaml_path, args_list = sys.argv[1], sys.argv[2], sys.argv[3:]
cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])

# Load model and tokenizer
MODEL_NAME = "allenai/OLMoE-1B-7B-0924"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = OLMo(cfg.model)
# model = model.to("cuda")
model.load_state_dict(torch.load(model_path))
model.eval()

def compute_logprobs_from_logits_step(logits: torch.FloatTensor, top_k: int = 5):
    """
    logits: Tensor of shape (vocab_size,)
    returns:
      - log_probs: Tensor of shape (vocab_size,)
      - top_logprobs: dict of top_k tokens -> logprob
    """
    log_probs = torch.log_softmax(logits, dim=-1)  # (vocab_size,)
    topk_vals, topk_ids = torch.topk(log_probs, k=top_k)
    topk_tokens = tokenizer.convert_ids_to_tokens(topk_ids.tolist())
    topk_dict = {tok: val.item() for tok, val in zip(topk_tokens, topk_vals)}
    return log_probs, topk_dict

# Request schemas
class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = 50
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0

# Response schemas
class Choice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None

class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]

@app.post("/v1/completions", response_model=CompletionResponse)
def completions(req: CompletionRequest):
    text = req.prompt
    # Tokenize raw text to get word_ids and sentence_ids
    raw_encoding = tokenizer(
        text,
        max_length=5120,
        truncation=True,
        return_offsets_mapping=True,
    )
    # Compute word_ids
    if hasattr(raw_encoding, "encodings") and raw_encoding.encodings:
        word_ids_list = raw_encoding.encodings[0].word_ids
        word_ids_list = [w if w is not None else -1 for w in word_ids_list]
    else:
        word_ids_list = [-1] * len(raw_encoding["input_ids"])
    # Compute sentence_ids
    offset_mapping = raw_encoding["offset_mapping"]
    sentence_id = 0
    sentence_ids_list: List[int] = []
    for (start, end) in offset_mapping:
        token_str = text[start:end]
        sentence_ids_list.append(sentence_id)
        if token_str.strip() == ".":
            sentence_id += 1
    # Convert to tensors
    word_ids = torch.tensor([word_ids_list], device=model.device)
    sentence_ids = torch.tensor([sentence_ids_list], device=model.device)

    # Encode inputs
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    prompt_length = input_ids.shape[1]

    # Prepare lists to collect generated tokens and their stats
    generated_ids: List[int] = []
    token_logprobs: List[float] = []
    top_logprobs: List[Dict[str, float]] = []

    # Autoregressive sampling loop using forward calls
    cur_ids = input_ids
    cur_word_ids = word_ids
    cur_sentence_ids = sentence_ids
    for _ in range(req.max_tokens):
        outputs = model(
            input_ids=cur_ids,
            word_ids=cur_word_ids,
            sentence_ids=cur_sentence_ids,
        )
        logits_step = outputs.logits[0, -1]  # (vocab_size,)
        # Compute log_probs and top_logprobs
        log_probs, topk_dict = compute_logprobs_from_logits_step(logits_step, top_k=5)
        # Temperature and nucleus sampling
        probs = torch.softmax(logits_step / max(1e-5, req.temperature), dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_probs[cumulative_probs > req.top_p] = 0
        normalized_probs = sorted_probs / sorted_probs.sum()
        next_token_idx = torch.multinomial(normalized_probs, num_samples=1)
        next_token_id = sorted_indices[next_token_idx].item()

        # Record stats
        generated_ids.append(next_token_id)
        token_logprobs.append(log_probs[next_token_id].item())
        top_logprobs.append(topk_dict)

        # Update sequences and IDs
        cur_ids = torch.cat([cur_ids, torch.tensor([[next_token_id]], device=model.device)], dim=1)
        cur_word_ids = torch.cat([cur_word_ids, torch.tensor([[ -1 ]], device=model.device)], dim=1)
        cur_sentence_ids = torch.cat([cur_sentence_ids, torch.tensor([[sentence_id]], device=model.device)], dim=1)

        # Stop if EOS
        if next_token_id == tokenizer.eos_token_id:
            break

    # Decode generated text
    text_out = tokenizer.decode(generated_ids, skip_special_tokens=True)

    logprobs_field = {
        "tokens": tokenizer.convert_ids_to_tokens(generated_ids),
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs
    }
    choice = Choice(text=text_out, index=0, logprobs=logprobs_field, finish_reason="stop")
    return CompletionResponse(
        id=str(int(time.time())),
        object="text_completion",
        created=int(time.time()),
        model=req.model,
        choices=[choice]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
