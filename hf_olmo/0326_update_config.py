import os

os.chdir("/n/home08/zkong/mufan/tmp/moebench/OLMo")
import json
import shutil
from glob import glob

from tqdm.auto import tqdm

for folder in glob("runs/*"):
    model_dir = folder + "/latest"
    if not os.path.exists(model_dir):
        print(f"Skipping {folder} since latest does not exist")
        continue
    if not os.path.exists(f"{model_dir}/config.json"):
        print(f"Skipping {folder} since config.json does not exist")
        continue
    if not os.path.exists(f"{model_dir}/config.bak.json"):
        shutil.copy(f"{model_dir}/config.json", f"{model_dir}/config.bak.json")
    if not os.path.exists(f"{model_dir}/tokenizer_config.bak.json"):
        shutil.copy(f"{model_dir}/tokenizer_config.json", f"{model_dir}/tokenizer_config.bak.json")
    with open(f"{model_dir}/config.bak.json", "r") as f:
        config = json.load(f)
        config["auto_map"] = {
            "AutoConfig": "configuration_olmo.OLMoConfig",
            "AutoModelForCausalLM": "modeling_olmo.OLMoForCausalLM",
            "AutoModelForSequenceClassification": "modeling_olmo.OLMoForSequenceClassification",
        }
        config["model_type"] = "hf_olmo"
    with open(f"{model_dir}/config.json", "w") as f:
        json.dump(config, f, indent=4)
    with open(f"{model_dir}/tokenizer_config.bak.json", "r") as f:
        config = json.load(f)
        config["tokenizer_class"] = "GPTNeoXTokenizer"
    with open(f"{model_dir}/tokenizer_config.json", "w") as f:
        json.dump(config, f, indent=4)
    shutil.copy(f"hf_olmo/tokenization_olmo_fast.py", f"{model_dir}/tokenization_olmo_fast.py")
    shutil.copy(f"hf_olmo/configuration_olmo.py", f"{model_dir}/configuration_olmo.py")
    shutil.copy(f"hf_olmo/modeling_olmo.py", f"{model_dir}/modeling_olmo.py")
    print(f"Updated {folder}")
