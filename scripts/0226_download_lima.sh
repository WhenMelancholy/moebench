#!/bin/bash
set -e

# HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --repo-type dataset --local-dir ./lima GAIR/lima
# HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --repo-type dataset --local-dir ./tulu-v3.1-mix-preview-4096-OLMoE allenai/tulu-v3.1-mix-preview-4096-OLMoE
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --repo-type model --local-dir ./OLMoE-1B-7B-0924 allenai/OLMoE-1B-7B-0924
