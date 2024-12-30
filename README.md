# UAlign: Leveraging Uncertainty Estimations for Factuality Alignment on Large Language Models
<<<<<<< Updated upstream

The management of this project and completed implementations are in progress and will be available soon ...  
=======
>>>>>>> Stashed changes

## Introduction

This is the repository for the UAlign framework, which leverages Uncertainty estimations to represent knowledge boundaries, and then explicitly incorporates these representations as input features into prompts for LLMs to Align with factual knowledge as examplified in 

![alt text](figs/examples.png)

The project includes two stages: 1)dataset construction; and 2) alignment. The completely management of this project is in progress.

---

## Implementation

### Stage 1: Data construction: Dataset Sampling and Construction

![alt text](figs/dataset.png)

```shell

CUDA_VISIBLE_DEVICES=0,1,2,3 python code/sample.py --model_name llama3 --dataset sciq --data_file validation
python code/uncertainty.py --model_name llama3 --dataset triviaqa --data_file train

```

### Stage 2: LLM Training (SFT, PPO, DPO)

![alt text](figs/alignment.png)

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python code/train_sft.py --model_name mistral --dataset comb --data_file train_sft --save_suffix base
CUDA_VISIBLE_DEVICES=0,1,2,3 python code/train_ppo.py --model_name mistral --dataset comb --data_file train_ppo --save_suffix base
CUDA_VISIBLE_DEVICES=0,1,2,3 python code/train_dpo.py --model_name mistral --dataset comb --data_file train_dpo --save_suffix base
```

### Stage 3: LLM decoding and inference 

```shell
python code/infer.py --model_name $model --dataset $dataset --data_file validation --max_length 16 --lora_use true --model_suffix sft_base
```


### Stage 4: Evaluate the generations

```shell
python code/eval.py --model_name $model --dataset $dataset --data_file validation --model_suffix vanilla_$icl_type
```
