# CUFact: Leveraging Certainty and Uncertainty Estimations to Improve Factuality for LLMs

## Implementation

### Stage 1: Data construction and split the difficulty levels

Dataset Sampling and Construction

```shell
for model in llama3 mistral; do CUDA_VISIBLE_DEVICES=0 python code/sample.py --model_name $model --dataset triviaqa --data_file train; done

CUDA_VISIBLE_DEVICES=3 python code/sample.py --model_name llama3 --dataset sciq --data_file validation

```

### Stage 2: Supervised Fine-tuning (SFT) 

```shell
CUDA_VISIBLE_DEVICES=2,3 python code/train_sft.py --model_name mistral --dataset comb --data_file train_sft --save_suffix base_bnb --bnb_use true

```


### Stage 3: LLM decoding and generation 

```shell
for model in llama3 mistral; do
    for dataset in triviaqa sciq nqopen lsqa; do
        for icl_type in icl icl_idk; do
            CUDA_VISIBLE_DEVICES=3 python code/infer.py --model_name $model --dataset $dataset --data_file validation --max_length 16 --icl_type $icl_type --bnb_use false --model_suffix vanilla
            CUDA_VISIBLE_DEVICES=3 python code/infer.py --model_name $model --dataset $dataset --data_file validation --max_length 16 --icl_type $icl_type --bnb_use true --model_suffix vanilla_bnb
        done
    done
done

```


### Stage 4: Evaluate the generations

```shell
# TriviaQA
for model in llama3 mistral; do for dataset in lsqa; do python code/eval.py --model_name $model --dataset $dataset --data_file validation --model_suffix vanilla_icl; python code/eval.py --model_name $model --dataset $dataset --data_file validation --model_suffix vanilla_bnb_icl_idk; done; done
```
