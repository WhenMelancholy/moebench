cd /mnt/weka/home/haolong.jia/opt/moebench/olmo

model_dirs=(
    "0312-OLMoE-300M-15ep/"
    "0312-OLMoE-300M-15ep-15of64-sharedTrue/"
    "0312-OLMoE-300M-15ep-15of64-sharedTrue-auxfree/"
    "0312-OLMoE-300M-15ep-15of64-sharedTrue-noaux/"
    "0312-OLMoE-300M-15ep-15of64-sharedTrue-nozloss/"
    "0312-OLMoE-300M-15ep-16of64-sharedFalse/"
    "0312-OLMoE-300M-15ep-16of64-sharedFalse-auxfree/"
    "0312-OLMoE-300M-15ep-16of64-sharedFalse-noaux/"
    "0312-OLMoE-300M-15ep-16of64-sharedFalse-nozloss/"
    "0312-OLMoE-300M-15ep-1of8-sharedTrue/"
    "0312-OLMoE-300M-15ep-1of8-sharedTrue-auxfree/"
    "0312-OLMoE-300M-15ep-1of8-sharedTrue-noaux/"
    "0312-OLMoE-300M-15ep-1of8-sharedTrue-nozloss/"
    "0312-OLMoE-300M-15ep-2of8/"
    "0312-OLMoE-300M-15ep-2of8-sharedFalse/"
    "0312-OLMoE-300M-15ep-2of8-sharedFalse-auxfree/"
    "0312-OLMoE-300M-15ep-2of8-sharedFalse-noaux/"
    "0312-OLMoE-300M-15ep-2of8-sharedFalse-nozloss/"
    "0312-OLMoE-300M-15ep-3of16-sharedTrue/"
    "0312-OLMoE-300M-15ep-3of16-sharedTrue-auxfree/"
    "0312-OLMoE-300M-15ep-3of16-sharedTrue-noaux/"
    "0312-OLMoE-300M-15ep-3of16-sharedTrue-nozloss/"
    "0312-OLMoE-300M-15ep-4of16-sharedFalse/"
    "0312-OLMoE-300M-15ep-4of16-sharedFalse-auxfree/"
    "0312-OLMoE-300M-15ep-4of16-sharedFalse-noaux/"
    "0312-OLMoE-300M-15ep-4of16-sharedFalse-nozloss/"
    "0312-OLMoE-300M-15ep-7of32-sharedTrue/"
    "0312-OLMoE-300M-15ep-7of32-sharedTrue-auxfree/"
    "0312-OLMoE-300M-15ep-8of32-sharedFalse/"
    "0312-OLMoE-300M-15ep-8of32-sharedFalse-auxfree/"
    "0312-OLMoE-300M-15ep-8of32-sharedFalse-noaux/"
    "0312-OLMoE-300M-15ep-8of32-sharedFalse-nozloss/"
)

mkdir -p results/baseline
mkdir -p results/random
mkdir -p results/prune

for model_dir in "${model_dirs[@]}"; do
    echo "Evaluating ${model_dir}..."
    model_dir="/mnt/weka/home/haolong.jia/opt/models/runs/${model_dir}/step10665"
    CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
    --model_args pretrained=${model_dir},trust_remote_code=True,random_router=False \
    --tasks piqa,truthfulqa_mc1,truthfulqa_mc2,nq_open \
    --batch_size auto \
    --output_path ./results/baseline \
    --device cuda:0

    CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
    --model_args pretrained=${model_dir},trust_remote_code=True,random_router=False,save_router_logits="${model_dir}/baseline.pt" \
    --tasks winogrande,mmlu \
    --num_few_shots 5 \
    --batch_size auto \
    --output_path ./results/baseline \
    --device cuda:0

    CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
    --model_args pretrained=${model_dir},trust_remote_code=True,random_router=False \
    --tasks hellaswag \
    --num_few_shots 10 \
    --batch_size auto \
    --output_path ./results/baseline \
    --device cuda:0

    CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
    --model_args pretrained=${model_dir},trust_remote_code=True,random_router=False \
    --tasks arc_challenge,arc_easy \
    --num_few_shots 25 \
    --batch_size auto \
    --output_path ./results/baseline \
    --device cuda:0

    CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
        --model_args pretrained=${model_dir},trust_remote_code=True,random_router=True,save_router_logits="${model_dir}/random.pt" \
        --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy \
        --batch_size auto \
        --output_path ./results/random \
        --device cuda:0

    CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
        --model_args pretrained=${model_dir},trust_remote_code=True,random_router=False,save_router_logits="${model_dir}/prune.pt",prune_experts="${model_dir}/baseline.pt" \
        --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy \
        --batch_size auto \
        --output_path ./results/prune \
        --device cuda:0
done