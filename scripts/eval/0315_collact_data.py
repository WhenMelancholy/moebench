# %%
import json
import os

import numpy as np
import torch

os.chdir("/n/home08/zkong/mufan/tmp/moebench/open-instruct")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def avg(lst):
    return sum(lst) / len(lst)


all_numbers = []
for i in range(15):
    model_dir = f"output/0312_lima_safe_epoch/epoch_{i}"
    print(f" =========== {model_dir} =========== ")
    print()
    result_dir = f"results/mode/.__output__0312_lima_safe_epoch__epoch_{i}"

    row = {}

    # %%
    def find_latest_json_file(directory):
        # 获取目录下所有以 .json 结尾的文件列表
        json_files = [f for f in os.listdir(directory) if f.endswith(".json")]

        # 如果没有找到任何 .json 文件，返回 None
        if not json_files:
            return None

        # 获取每个文件的完整路径
        full_paths = [os.path.join(directory, f) for f in json_files]

        # 找到最新的文件
        latest_file = max(full_paths, key=os.path.getmtime)

        return latest_file

    def get_acc(mode):
        result_path = find_latest_json_file(result_dir.replace("mode", mode))
        result = json.load(open(result_path))

        datasets = [
            "arc_challenge",
            "arc_easy",
            "mmlu",
            "piqa",
            "winogrande",
        ]
        acc = {dataset: result["results"][dataset]["acc,none"] for dataset in datasets}

        if mode == "baseline":
            acc["nq"] = result["results"]["nq_open"]["exact_match,remove_whitespace"]
            datasets = ["truthfulqa_mc1", "truthfulqa_mc2"]
            for dataset in datasets:
                acc[dataset] = result["results"][dataset]["acc,none"]

        return acc

    print(get_acc("baseline"))
    print(get_acc("random"))
    print(get_acc("prune"))
    print(get_acc("baseline").values())
    print(get_acc("random").values())
    print(get_acc("prune").values())

    row["baseline"] = list(get_acc("baseline").values())
    row["random"] = list(get_acc("random").values())
    row["prune"] = list(get_acc("prune").values())
    topsubprune = (np.array(row["baseline"])[:-3] - np.array(row["prune"])).mean()
    topsubrandom = (np.array(row["baseline"])[:-3] - np.array(row["random"])).mean()
    row["capacity"] = topsubprune
    row["specialization"] = topsubrandom

    # %%
    def calculate_entropy(A):
        A = A.float()
        row_sums = A.sum(dim=-1, keepdim=True)
        P = A / (row_sums + 1e-10)
        P_log_P = P * torch.log2(P + 1e-10)
        entropy = -P_log_P.sum(dim=-1)
        return entropy

    logits = torch.load(f"{model_dir}/baseline.pt", weights_only=True)
    selected_experts = torch.topk(logits, k=8, dim=-1).indices
    print("selected_experts shape", selected_experts.shape)
    expert_frequency = torch.zeros((selected_experts.shape[0], 64), dtype=torch.int32)
    for i in range(selected_experts.shape[0]):
        counts = torch.bincount(selected_experts[i].flatten(), minlength=64)
        expert_frequency[i] = counts
    print("expert_frequency shape", expert_frequency.shape)
    # print("expert_frequency", expert_frequency)

    entropy = calculate_entropy(expert_frequency)
    # print("entropy", entropy)
    print("average entropy", entropy.mean())
    print("max entropy", entropy.max())
    print("entropy for each layer", entropy.tolist())
    row["load balance"] = entropy.tolist()

    # %%
    from tqdm.auto import tqdm

    top_expert_frequency = []
    for layer in tqdm(range(logits.shape[0]), desc="layer"):
        top1_experts = torch.argmax(logits[layer], dim=1)  # 形状为 [2693448]

        expert_logits = []
        for expert_idx in tqdm(range(logits.shape[2]), desc="expert", disable=True):
            # 创建布尔掩码，标记出当前专家为 Top-1 的位置
            mask = top1_experts == expert_idx  # 形状为 [2,693,448]
            # 使用掩码筛选出对应的 logits
            selected_logits = logits[layer, mask]
            expert_logits.append(selected_logits)
            # print("selected_logits shape", selected_logits.shape)
        top_expert_frequency.append(expert_logits)

    collaboration = torch.zeros(
        (logits.shape[0], logits.shape[2], logits.shape[2] - 1), dtype=torch.int32
    )
    for layer in tqdm(range(logits.shape[0]), desc="layer"):
        for expert in tqdm(range(logits.shape[2]), desc="expert", disable=True):
            expert_logits = top_expert_frequency[layer][expert]
            expert_logits[:, expert] = float("-inf")
            top_experts = torch.topk(expert_logits, k=7, dim=-1).indices
            collaboration_frequency = torch.bincount(
                top_experts.flatten(), minlength=64
            )
            mask = torch.arange(64) != expert
            collaboration[layer, expert] = collaboration_frequency[mask]

    entropy = calculate_entropy(collaboration)
    # print(collaboration)
    print("entropy", entropy)
    print("average entropy", entropy.mean())
    print("shape of entropy", entropy.shape)
    print("collaboration for each layer", entropy.mean(dim=1).tolist())
    row["collaboration"] = entropy.mean(dim=1).tolist()

    # %%
    datasets = ["I-MaliciousInstructions.json", "I-CoNa.json", "I-Controversial.json"]
    row["safety"] = []
    for dataset in datasets:
        result_path = f"{model_dir}/{dataset}"
        result = json.load(open(result_path))
        harmfulness = [item["harmfulness"] for item in result]
        llamaguard = [item["llamaguard"] != "safe" for item in result]
        openai = [item["opanai"] for item in result]
        print(f"{dataset}: average harmfulness {sum(harmfulness) / len(harmfulness)}")
        print(f"{dataset}: average llamaguard {sum(llamaguard) / len(llamaguard)}")
        print(f"{dataset}: average openai {sum(openai) / len(openai)}")
        print(
            f"{sum(harmfulness) / len(harmfulness)}, {sum(llamaguard) / len(llamaguard)}, {sum(openai) / len(openai)}"
        )
        row["safety"].extend(
            [
                sum(harmfulness) / len(harmfulness),
                sum(llamaguard) / len(llamaguard),
                sum(openai) / len(openai),
            ]
        )
    numbers = []
    numbers.extend(row["baseline"][:-3])
    numbers.append(avg(row["baseline"][:-3]))
    numbers.extend(row["baseline"][-3:])
    numbers.append(avg(row["load balance"]))
    numbers.append(row["capacity"])
    numbers.append(row["specialization"])
    numbers.append(avg(row["collaboration"]))
    numbers.extend(row["safety"])
    numbers.extend(row["load balance"])
    numbers.extend(row["collaboration"])
    numbers.extend(row["random"])
    numbers.append(avg(row["random"]))
    numbers.extend(row["prune"])
    numbers.append(avg(row["prune"]))
    print(numbers)
    print()
    all_numbers.append(numbers)
print(" =========== all_numbers =========== ")
for numbers in all_numbers:
    print(",".join(map(str, numbers)))
