#!/bin/bash

set -euo pipefail

HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --repo-type model --local-dir /n/home08/zkong/mufan/tmp/moebench/open-instruct/output/deepseek-moe-16b-chat deepseek-ai/deepseek-moe-16b-chat
