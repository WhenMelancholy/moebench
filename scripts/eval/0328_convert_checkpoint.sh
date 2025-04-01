#!/bin/bash

set -ex

cd ~/mufan/tmp/moebench/open-instruct/output/0319_key_olmo7b/step_5000
python zero_to_fp32.py . .
