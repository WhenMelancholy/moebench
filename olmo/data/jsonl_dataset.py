import json
from typing import List, Tuple, Any, Dict, Union
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F

__all__ = ["JsonlDataset"]

@dataclass
class DataCollator:
    pad_direction: str
    pad_token_id: int

    def __init__(self, pad_direction: str = "right", pad_token_id: int = 1):
        self.pad_direction = pad_direction
        self.pad_token_id = pad_token_id

    def __call__(self, items: Union[List[Dict[str, Any]], List[torch.Tensor]]) -> Dict[str, Any]:
        assert items
        max_len = max((len(x["input_ids"] if isinstance(x, dict) else x) for x in items))
        all_input_ids = []
        all_attention_mask = []
        all_attention_bias = []
        all_label_mask = []
        all_indices = []
        all_metadata = []
        all_instance_mask = []
        all_doc_lens = []
        all_max_doc_lens = []
        all_word_ids = []
        all_sentence_ids = []
        max_docs = max((len(x["doc_lens"]) if isinstance(x, dict) and "doc_lens" in x else 0 for x in items))

        for x in items:
            # Process input_ids.
            input_ids = x["input_ids"] if isinstance(x, dict) else x
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids, dtype=torch.long)

            pad_shape = (
                (max_len - len(input_ids), 0)
                if self.pad_direction == "left"
                else (0, max_len - len(input_ids))
            )
            all_input_ids.append(
                F.pad(
                    input_ids,
                    pad_shape,
                    value=self.pad_token_id,
                )
            )

            # Process attention mask.
            attention_mask = x.get("attention_mask")
            if attention_mask is not None:
                if not isinstance(attention_mask, torch.Tensor):
                    attention_mask = torch.tensor(attention_mask, dtype=torch.float)
                all_attention_mask.append(
                    F.pad(
                        attention_mask,
                        pad_shape,
                        value=0.0,
                    )
                )

            # Process attention bias.
            attention_bias = x.get("attention_bias")
            if attention_bias is not None:
                if not isinstance(attention_bias, torch.Tensor):
                    attention_bias = torch.tensor(attention_bias)
                # Reshape to `(1, seq_len, seq_len)` if needed.
                while len(attention_bias.shape) < 3:
                    attention_bias = attention_bias.unsqueeze(0)
                pad_value = False if attention_bias.dtype == torch.bool else float("-inf")
                all_attention_bias.append(
                    F.pad(
                        attention_bias,
                        pad_shape + pad_shape,
                        value=pad_value,
                    )
                )

            # Process label mask.
            label_mask = x.get("label_mask")
            if label_mask is not None:
                if not isinstance(label_mask, torch.Tensor):
                    label_mask = torch.tensor(label_mask, dtype=torch.bool)
                all_label_mask.append(
                    F.pad(
                        label_mask,
                        pad_shape,
                        value=False,
                    )
                )

            # Process indices.
            index = x.get("index")
            if index is not None:
                all_indices.append(torch.tensor(index))

            # Process instance mask.
            instance_mask = x.get("instance_mask")
            if instance_mask is not None:
                all_instance_mask.append(torch.tensor(instance_mask))

            # Process document lengths.
            doc_lens = x.get("doc_lens")
            if doc_lens is not None:
                doc_pad_shape = (0, max_docs - len(doc_lens))
                all_doc_lens.append(F.pad(doc_lens, doc_pad_shape, value=0))
                all_max_doc_lens.append(int(doc_lens.max()))

            # Process word_ids.
            if "word_ids" in x:
                word_ids = x["word_ids"]
                if not isinstance(word_ids, torch.Tensor):
                    word_ids = torch.tensor(word_ids, dtype=torch.long)
                # Use -1 as the padding value for word_ids.
                all_word_ids.append(F.pad(word_ids, pad_shape, value=-1))

            # Process sentence_ids.
            if "sentence_ids" in x:
                sentence_ids = x["sentence_ids"]
                if not isinstance(sentence_ids, torch.Tensor):
                    sentence_ids = torch.tensor(sentence_ids, dtype=torch.long)
                # Use -1 as the padding value for sentence_ids.
                all_sentence_ids.append(F.pad(sentence_ids, pad_shape, value=-1))

            # Process metadata.
            metadata = x.get("metadata")
            if metadata is not None:
                all_metadata.append(metadata)

        out: Dict[str, Any] = {"input_ids": torch.stack(all_input_ids)}
        if all_attention_mask:
            out["attention_mask"] = torch.stack(all_attention_mask)
        if all_attention_bias:
            out["attention_bias"] = torch.stack(all_attention_bias)
        if all_label_mask:
            out["label_mask"] = torch.stack(all_label_mask)
        if all_indices:
            out["index"] = torch.stack(all_indices)
        if all_instance_mask:
            out["instance_mask"] = torch.stack(all_instance_mask)
        if all_doc_lens:
            out["doc_lens"] = torch.stack(all_doc_lens)
        if all_max_doc_lens:
            out["max_doc_lens"] = all_max_doc_lens
        if all_word_ids:
            out["word_ids"] = torch.stack(all_word_ids)
        if all_sentence_ids:
            out["sentence_ids"] = torch.stack(all_sentence_ids)
        if all_metadata:
            out["metadata"] = all_metadata

        return out


class JsonlDataset(Dataset):
    """
    A lazy-loading PyTorch dataset for large JSONL files.

    Instead of loading all data into memory, this dataset builds an index mapping
    each example's global index to a tuple (file_index, byte_offset). When a sample
    is accessed via __getitem__, the corresponding file is opened and the line is read
    using the stored byte offset.

    Args:
        file_paths (List[str]): A list of file paths to JSONL files.
        tokenizer: A tokenizer (e.g. from HuggingFace) that converts text to token IDs.
        text_key (str): The key in each JSON object containing the text. Defaults to "text".
        max_length (int): Maximum sequence length for tokenization.
    """
    def __init__(
        self,
        file_paths: List[str],
        tokenizer,
        text_key: str = "text",
        max_length: int = 1024,
    ):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.text_key = text_key
        self.max_length = max_length
        
        # Optionally update the tokenizer configuration.
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = 1
        if getattr(self.tokenizer, "eos_token_id", None) is None:
            self.tokenizer.eos_token_id = 50279

        # Build an index that maps a global example index to (file_index, byte_offset).
        self.index: List[Tuple[int, int]] = []
        for file_idx, file_path in enumerate(self.file_paths):
            with open(file_path, "r", encoding="utf-8") as f:
                offset = f.tell()
                line = f.readline()
                while line:
                    self.index.append((file_idx, offset))
                    offset = f.tell()
                    line = f.readline()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Any:
        file_idx, offset = self.index[idx]
        file_path = self.file_paths[file_idx]
        # Open the file, seek to the byte offset, and read the line.
        with open(file_path, "r", encoding="utf-8") as f:
            f.seek(offset)
            line = f.readline()
        
        # Parse the JSON object.
        example = json.loads(line)
        # Extract the text using the specified key; if missing, default to an empty string.
        text = example.get(self.text_key, "")

        # --- Tokenization with additional fields (word_ids and sentence_ids) ---
        # Use return_offsets_mapping to obtain token positions.
        raw_encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_offsets_mapping=True,
        )
        # Get word_ids using the fast tokenizer's encodings.
        if hasattr(raw_encoding, "encodings") and raw_encoding.encodings:
            word_ids = raw_encoding.encodings[0].word_ids  # may contain None for special tokens
            # Replace None with -1
            word_ids = [w if w is not None else -1 for w in word_ids]
        else:
            word_ids = None

        # Compute sentence_ids by inspecting the token text via offsets.
        offset_mapping = raw_encoding["offset_mapping"]
        token_strs = [text[start:end] for (start, end) in offset_mapping]
        sentence_id = 0
        sentence_ids = []
        for token_str in token_strs:
            sentence_ids.append(sentence_id)
            # If the token text (stripped) is ".", increment the sentence counter.
            if token_str.strip() == ".":
                sentence_id += 1

        # Remove the offsets mapping if it is not needed downstream.
        raw_encoding.pop("offset_mapping", None)

        # Convert relevant keys to tensors (if they are lists) for consistency.
        for key in ["input_ids", "attention_mask"]:
            if key in raw_encoding and not isinstance(raw_encoding[key], torch.Tensor):
                raw_encoding[key] = torch.tensor(raw_encoding[key], dtype=torch.long)

        # Add the new fields.
        raw_encoding["word_ids"] = word_ids
        raw_encoding["sentence_ids"] = sentence_ids

        # Include any additional fields from the JSON object as metadata.
        metadata = {k: v for k, v in example.items() if k != self.text_key}
        raw_encoding["metadata"] = metadata

        return raw_encoding.data

# Example usage:
if __name__ == "__main__":
    from transformers import AutoTokenizer

    # import debugpy
    # try:
    #     print("Waiting for debugger to attach...")
    #     debugpy.listen(9501)
    #     debugpy.wait_for_client()
    # except Exception as e:
    #     print(e)

    # Initialize the tokenizer (make sure you use a fast tokenizer to access word_ids).
    tokenizer = AutoTokenizer.from_pretrained("allenai/gpt-neox-olmo-dolma-v1_5")
    
    # Provide a list of JSONL file paths.
    file_paths = [
        "/workspace/moebench-pretrain/data/minipile-jsonl/train/train-00000-of-00012.jsonl",
        "/workspace/moebench-pretrain/data/minipile-jsonl/train/train-00001-of-00012.jsonl",
        "/workspace/moebench-pretrain/data/minipile-jsonl/train/train-00002-of-00012.jsonl",
        "/workspace/moebench-pretrain/data/minipile-jsonl/train/train-00003-of-00012.jsonl",
        "/workspace/moebench-pretrain/data/minipile-jsonl/train/train-00004-of-00012.jsonl",
        "/workspace/moebench-pretrain/data/minipile-jsonl/train/train-00005-of-00012.jsonl",
        "/workspace/moebench-pretrain/data/minipile-jsonl/train/train-00006-of-00012.jsonl",
        "/workspace/moebench-pretrain/data/minipile-jsonl/train/train-00007-of-00012.jsonl",
        "/workspace/moebench-pretrain/data/minipile-jsonl/train/train-00008-of-00012.jsonl",
        "/workspace/moebench-pretrain/data/minipile-jsonl/train/train-00009-of-00012.jsonl",
        "/workspace/moebench-pretrain/data/minipile-jsonl/train/train-00010-of-00012.jsonl",
        "/workspace/moebench-pretrain/data/minipile-jsonl/train/train-00011-of-00012.jsonl"
    ]
    tokenizer.pad_token_id = 1
    tokenizer.eos_token_id = 50279

    # Create the dataset instance.
    dataset = JsonlDataset(
        file_paths=file_paths, 
        tokenizer=tokenizer, 
        text_key="text", 
        max_length=1024 * 1024 * 1024
    )

    collator = DataCollator(pad_direction="right", pad_token_id=1)

    # Create a DataLoader to batch and shuffle the dataset.
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        collate_fn=collator, 
        shuffle=True, 
        num_workers=32, 
        pin_memory=True
    )

    # Iterate over the DataLoader to retrieve batches of tokenized data.
    for batch in dataloader:
        input_ids = batch["input_ids"]         # Tensor of shape [batch_size, max_length]
        attention_mask = batch.get("attention_mask")  # Tensor of shape [batch_size, max_length]
        word_ids = batch.get("word_ids")         # Tensor of shape [batch_size, max_length]
        sentence_ids = batch.get("sentence_ids") # Tensor of shape [batch_size, max_length]
        metadata = batch.get("metadata")         # List of metadata dictionaries for each instance

        # Process the batch (e.g., pass into your model).
        print("Input IDs batch shape:", input_ids.shape)
        print("Attention mask batch shape:", attention_mask.shape)
        print("Word IDs batch shape:", word_ids.shape if word_ids is not None else None)
        print("Sentence IDs batch shape:", sentence_ids.shape if sentence_ids is not None else None)
        break  # Remove this break to process the entire dataset.
