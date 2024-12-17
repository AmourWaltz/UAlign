# UAlign: Leveraging Uncertainty Estimations for Factuality Alignment on Large Language Models

The management of this project and completed implementations are in progress and will be available soon ...  

## Implementation

### Stage 1: Data construction and split the difficulty levels

Dataset Sampling and Construction

```shell
for model in llama3 mistral; do CUDA_VISIBLE_DEVICES=0 python code/sample.py --model_name $model --dataset triviaqa --data_file train; done

CUDA_VISIBLE_DEVICES=3 python code/sample.py --model_name llama3 --dataset sciq --data_file validation

```

### Stage 2: LLM Training (SFT, PPO, DPO)

```shell
# Training of SFT and R-Tuning baselines
CUDA_VISIBLE_DEVICES=2,3 python code/train_sft.py --model_name mistral --dataset comb --data_file train_sft --save_suffix base_bnb --bnb_use true

# Training of PPO and CUAlign
CUDA_VISIBLE_DEVICES=2,3 python code/train_ppo.py --model_name mistral --dataset comb --data_file train_ppo --save_suffix base_bnb --bnb_use true

# Training of DPO
CUDA_VISIBLE_DEVICES=2,3 python code/train_dpo.py --model_name mistral --dataset comb --data_file train_dpo --save_suffix base_bnb --bnb_use true
```

### Stage 3: LLM decoding and generation 

```shell
# Inference of prompt-based baseline methods
for model in llama3 mistral; do for dataset in triviaqa sciq nqopen lsqa; do for icl_type in icl icl_idk; do CUDA_VISIBLE_DEVICES=3 python code/infer.py --model_name $model --dataset $dataset --data_file validation --max_length 16 --icl_type $icl_type --bnb_use true --model_suffix vanilla_bnb; done; done; done

for model in llama3 mistral; do for dataset in triviaqa sciq nqopen lsqa; do for icl_type in icl_cot; do CUDA_VISIBLE_DEVICES=3 python code/infer.py --model_name $model --dataset $dataset --data_file validation --max_length 64 --icl_type $icl_type --bnb_use true --model_suffix vanilla_bnb; done; done; done


# Inference of SFT baseline
for model in llama3 mistral; do for dataset in triviaqa sciq nqopen lsqa; do CUDA_VISIBLE_DEVICES=0 python code/infer.py --model_name $model --dataset $dataset --data_file validation --max_length 16 --icl_type no --bnb_use true --lora_use true --model_suffix sft_base_bnb; done; done

# Inference of R-Tuning baseline
for model in llama3 mistral; do for dataset in triviaqa sciq nqopen lsqa; do CUDA_VISIBLE_DEVICES=1 python code/infer.py --model_name $model --dataset $dataset --data_file validation --max_length 16 --icl_type no --bnb_use true --lora_use true --model_suffix rtuning_base_bnb; done; done

```


### Stage 4: Evaluate the generations

```shell
# Evaluation of prompt-based methods
for model in llama3 mistral; do for dataset in nqopen sciq triviaqa lsqa; do for icl_type in icl icl_idk icl_cot; do python code/eval.py --model_name $model --dataset $dataset --data_file validation --model_suffix vanilla_$icl_type; python code/eval.py --model_name $model --dataset $dataset --data_file validation --model_suffix vanilla_bnb_$icl_type; done; done; done

# Evaluation of SFT baseline
for model in llama3 mistral; do for dataset in nqopen sciq triviaqa lsqa; do python code/eval.py --model_name $model --dataset $dataset --data_file validation --model_suffix sft_base_bnb_std; done; done

# Evaluation of R-Tuning baseline
for model in llama3 mistral; do for dataset in nqopen sciq triviaqa lsqa; do python code/eval.py --model_name $model --dataset $dataset --data_file validation --model_suffix rtuning_base_bnb_std; done; done
```
