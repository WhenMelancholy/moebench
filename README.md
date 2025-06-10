# Fine-tuning Language Models for Keystroke Dynamics Analysis

This repository provides a suite of scripts for fine-tuning pre-trained language models and subsequently generating predictions. The primary application focus is on tasks related to keystroke dynamics, potentially for research in areas such as side-channel attack analysis, inspired by studies like "Keylogging Side-Channel Analysis on Social Messaging Applications via Tapping Activity Inference."

## Overview

The project facilitates the adaptation of large language models (e.g., OLMo, Llama) to specialized datasets that may represent patterns indicative of keystroke dynamics or similar user input behaviors. The provided scripts streamline two main processes:
1.  **Model Fine-tuning**: Adapting various pre-trained language models on custom datasets.
2.  **Prediction Generation**: Utilizing these fine-tuned models to generate outputs for evaluation and analysis.

The codebase leverages `accelerate` for distributed training and includes tools for evaluation on bespoke datasets.

## Prerequisites and Setup

1.  **Clone the Repository**:
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **Environment Configuration**:
    Ensure a Python environment with PyTorch, Transformers, Accelerate, and other requisite libraries is established. The scripts are pre-configured for SLURM-managed clusters but can be adapted for other computational environments.

## Model Fine-tuning

Model fine-tuning is orchestrated primarily through the [open-instruct/scripts/0425_finetune_key_four.sh](open-instruct/scripts/0425_finetune_key_four.sh) script, which in turn utilizes [open-instruct/open_instruct/finetune.py](open-instruct/open_instruct/finetune.py) to manage the training process.

### 1. Configuring `scripts/0425_finetune_key_four.sh`
   - **SLURM Directives**: Adjust the `#SBATCH` directives at the beginning of the script to match your cluster's configuration (e.g., nodes, GPUs, time allocation, memory).
   - **Model Selection**: Modify the `model_names` and `output_suffixs` arrays within the script to specify the base models for fine-tuning and to define corresponding output identifiers.
     ````bash
     // filepath: open-instruct/scripts/0425_finetune_key_four.sh
     // ...existing code...
     // Array of model identifiers from Hugging Face or local paths
     model_names=(
         allenai/OLMo-1B-0724-hf
         meta-llama/Llama-3.2-1B
         # ...add or modify models as needed
         ./output/0307_key_olmo // Example: using a previously fine-tuned model as a base
     )
     // Corresponding suffixes for output directories
     output_suffixs=(
         olmo
         llama1b
         # ...add or modify corresponding suffixes
         key_olmo
     )
     // ...existing code...
     ````
   - **Training Parameters**: Within the `accelerate launch` command block, various training hyperparameters can be configured:
     - `--model_name_or_path`, `--tokenizer_name`
     - `--max_seq_length`
     - `--per_device_train_batch_size`, `--gradient_accumulation_steps`
     - `--learning_rate`, `--num_train_epochs`
     - `--output_dir` (e.g., `output/${DATE}_key_cache_${output_suffix}`)
     - `--dataset_mixer_list` (e.g., `WhenceFade/0601_key_cache_dynamic_olmoe 1.0`), specifying the training dataset and its sampling proportion.

### 2. Executing the Fine-tuning Process
   Submit the script to the SLURM scheduler:
   ```bash
   sbatch open-instruct/scripts/0425_finetune_key_four.sh
   ```
   To execute a specific configuration from the `model_names` array (e.g., the first model, at index 0), you can set the `SLURM_ARRAY_TASK_ID` environment variable:
   ```bash
   SLURM_ARRAY_TASK_ID=0 bash open-instruct/scripts/0425_finetune_key_four.sh
   ```
   This will initiate the training procedure managed by `open_instruct/finetune.py`.

### Regarding `open_instruct/finetune.py`
   The core Python script for fine-tuning is [open-instruct/open_instruct/finetune.py](open-instruct/open_instruct/finetune.py). It employs the Hugging Face `transformers` and `accelerate` libraries for efficient, potentially distributed, training. This script handles data preprocessing, model configuration (including options for LoRA and Flash Attention), optimizer setup, and the main training loop. For bespoke training logic modifications, this script is the primary target for edits.

## Generating Predictions and Evaluation

The [open-instruct/scripts/eval/0501_evaluate_key.sh](open-instruct/scripts/eval/0501_evaluate_key.sh) script is used for running evaluations. This script leverages [open-instruct/scripts/eval/0319_evaluate_key.py](open-instruct/scripts/eval/0319_evaluate_key.py) to generate predictions from the fine-tuned models.

### 1. Configuring `scripts/eval/0501_evaluate_key.sh`
   - **SLURM Directives**: Adjust the `#SBATCH` directives as necessary.
   - **Model Paths**: Update the `model_paths` array to include the paths to your fine-tuned models.
     ````bash
     // filepath: open-instruct/scripts/eval/0501_evaluate_key.sh
     // ...existing code...
     // Array of paths to the fine-tuned models
     model_paths=(
         "/path/to/your/finetuned_model_1" // Example placeholder
         "/n/home08/zkong/mufan/tmp/moebench/open-instruct/output/0307_key_olmo" // Example from script
         # ...add other model paths
     )
     // ...existing code...
     ````
   - **Datasets**: The `input_files` array specifies the test datasets. Ensure these files exist and are in JSONL format. The `DATA_DIR` variable points to the base directory for these datasets.
     ````bash
     // filepath: open-instruct/scripts/eval/0501_evaluate_key.sh
     // ...existing code...
     DATA_DIR="/n/home08/zkong/mufan/tmp/moebench/key/llama-cookbook/data/" // Ensure this path is correct
     input_files=(
         ${DATA_DIR}/cache_dynamic_new/across_participant_across_sentence_test.jsonl
         # ...other test files
     )
     // ...existing code...
     ````
   - **Output Files**: The `output_files` array defines the names for the prediction files, which will be saved within each model's directory.

### 2. Executing the Evaluation
   Submit the script to SLURM:
   ```bash
   sbatch open-instruct/scripts/eval/0501_evaluate_key.sh
   ```
   If using SLURM arrays for dataset selection (as configured with `SLURM_ARRAY_TASK_ID`), the script will iterate through the specified models and datasets.

### Regarding `scripts/eval/0319_evaluate_key.py`
   This Python script, [open-instruct/scripts/eval/0319_evaluate_key.py](open-instruct/scripts/eval/0319_evaluate_key.py), is responsible for loading a specified model and generating predictions on a given dataset.
   - **Arguments**: It accepts `--input_path` (path to test data in JSONL format), `--model_path` (path to the fine-tuned model), and `--output_path` (location to save predictions) as command-line arguments.
   - **Functionality**:
     - Loads the specified model and tokenizer.
     - Reads input messages from the `input_path` (expected in JSONL format).
     - Generates multiple prediction sequences using the model's `generate` method. Generation parameters (e.g., `max_new_tokens`, `num_beams`) are configurable within the script.
     - Saves the input, target, and generated predictions in JSONL format to the `output_path`.

## Citation

This work is informed by research in keystroke dynamics and side-channel analysis. If utilizing or adapting this codebase for research in related domains, please consider citing relevant academic literature. For instance, studies such as:

*   "Keylogging Side-Channel Analysis on Social Messaging Applications via Tapping Activity Inference" (based on the contextual PDF provided).

It is recommended to also include citations pertinent to the specific datasets or methodologies employed in your project.
