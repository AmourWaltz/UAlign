r"""
Author: XUE Boyang      Filename: train_ppo.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Proximal Policy Optimization on QA dataset.
"""
import os
import json
import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from accelerate import Accelerator

import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from trl import SFTTrainer, PPOTrainer, PPOConfig
from trl.core import LengthSampler
from transformers import Trainer, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, Adafactor
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import *


@dataclass
class ModelArguments:
    model_name: str = field(default="gpt2", metadata={"help": "Model name.", "choices": model_path_dict.keys()})
    reward_model_path: str = field(default="gpt2", metadata={"help": "Reward model path."})
    device_map: str = field(default="balanced", metadata={"help": "Device map for the model."})
    use_cache: bool = field(default=False, metadata={"help": "Whether to use model cache."})
    # Bits and Bytes config
    bnb_use: bool = field(default=True, metadata={"help": "Whether to use BitsAndBytesConfig."})
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to enable 4-bit quantization."})
    bnb_4bit_quant_type: str = field(default="nf4", metadata={"help": "Set the quantization data type in the bnb.nn.Linear4Bit layers", "choices": ["fp4", "nf4"]})
    bnb_4bit_compute_dtype: torch.dtype = field(default=torch.float16, metadata={"help": "Set the computational type which might be different than the input type."})
    # Tokenizer setting
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    use_fast: bool = field(default=False, metadata={"help": "Whether to use rust based fast tokenizer."})
    padding_side: str = field(default="right", metadata={"help": "Padding side for sequences with different length."})
    trust_remote_code: bool = field(default=False, metadata={"help": "Whether or not to allow for custom models defined on the Hub in their own modeling files."})


@dataclass
class DataArguments:
    data_dir: str = field(default="./data/{}/prep", metadata={"help": "Directory to save data."})
    dataset: str = field(default="triviaqa", metadata={"help": "Dataset name.", "choices": dataset_list})
    data_file: str = field(default="train", metadata={"help": "Data file name."})
    prompt_dir: str = field(default="./prompt/", metadata={"help": "Path to the prompt."})
    continue_generate: bool = field(default=False, metadata={"help": "Continue from the previous generations."})
    model_suffix: str = field(default="ppo", metadata={"help": "File name to save the results."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # Maximal input sequence length
    max_seq_length: int = field(default=512, metadata={"help": "The maximum total input sequence length after tokenization."})
    # Checkpoints saving directory and strategy
    output_dir: str = field(default="./exp/{}/train", metadata={"help": "The output directory where the model predictions and checkpoints will be written."})
    save_strategy: str = field(default="epoch", metadata={"help": "The checkpoint save strategy."})
    save_total_limit: int = field(default=1, metadata={"help": "Limit the total amount of checkpoints."})
    remove_unused_columns: bool = field(default=False, metadata={"help": "Remove the unused columns in the dataset."})
    # Logging settings
    logging_dir: str = field(default="./logs", metadata={"help": "Log directory."})
    logging_steps: int = field(default=200, metadata={"help": "Log every X updates steps."})
    # Optimizer settings
    optim: str = field(default="paged_adamw_32bit", metadata={"help": "Optimizer to use."})
    warmup_ratio: float = field(default=0.05, metadata={"help": "Warmup ratio."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay."})
    gradient_accumulation_steps: int = field(default=8, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "Whether to use gradient checkpointing to save memory."})
    max_grad_norm: float = field(default=0.3, metadata={"help": "Max gradient norm."})
    # Learning rate settings
    learning_rate: float = field(default=2e-4, metadata={"help": "The initial learning rate for Adam."})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "Learning rate scheduler type."})
    # Batch size settings
    per_device_train_batch_size: int = field(default=4, metadata={"help": "Batch size per GPU/TPU core/CPU for training."})
    per_device_eval_batch_size: int = field(default=4, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."})
    # Training epoch settings
    num_train_epochs: int = field(default=3, metadata={"help": "Total number of training epochs to perform."})
    # Quantization uses
    fp16: bool = field(default=False, metadata={"help": "Whether to use fp16 for training."})
    bf16: bool = field(default=False, metadata={"help": "Whether to use bfloat16 for training."})
    # Evaluation settings
    eval_strategy: str = field(default="no", metadata={"help": "The evaluation strategy to adopt during training."})
    # PPO-specific settings
    ppo_epochs: int = field(default=4, metadata={"help": "Number of PPO epochs per batch."})
    batch_size: int = field(default=16, metadata={"help": "Total batch size."})
    mini_batch_size: int = field(default=4, metadata={"help": "Mini batch size per PPO step."})
    adafactor: bool = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: bool = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: float = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: float = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: bool = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: int = field(default=None, metadata={"help": "n steps to save the model"})


@dataclass
class LoraArguments:
    r: int = field(default=32, metadata={"help": "The number of bits to quantize."})
    lora_alpha: int = field(default=16, metadata={"help": "The number of bits to quantize."})
    lora_dropout: float = field(default=0.05, metadata={"help": "The dropout rate for LORA."})
    bias: str = field(default="none", metadata={"help": "The bias type for LORA."})
    task_type: str = field(default="CAUSAL_LM", metadata={"help": "The task type for LORA."})
    inference_mode: bool = field(default=False, metadata={"help": "Whether to use inference mode for LORA."})


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # Set up logging.
    training_args.output_dir = os.path.join(training_args.output_dir.format(data_args.dataset),
                                            f"{model_args.model_name}_{data_args.dataset}_{data_args.data_file}_{data_args.model_suffix}")

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    training_args.log_path = os.path.join(training_args.output_dir, f"train.log")
    info_log = os.path.join(training_args.output_dir, f"info.log")

    logging.basicConfig(
        filename=info_log,
        filemode='w',
        level=logging.INFO,
        datefmt="%d-%M-%Y %H:%M:%S",
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
    )

    """Model setting."""
    if model_args.bnb_use:
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=model_args.bnb_4bit_compute_dtype,
        )
        logging.info(f"BitsAndBytes use.")
    else:
        bnb_config = None
        logging.info(f"No BitsAndBytes use.")

    model_name_or_path = model_path_dict[model_args.model_name]
    logging.info(f"Loading model and tokenizer from {model_name_or_path} ...")
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                              torch_dtype=torch.float16,
                                                              quantization_config=bnb_config if model_args.bnb_use else None,
                                                              use_cache=model_args.use_cache,
                                                              device_map=model_args.device_map
                                                              )
    if model_args.bnb_use:
        model = prepare_model_for_kbit_training(model)


    """Tokenizer setting."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=model_args.model_max_length,
        padding_side=model_args.padding_side,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=model_args.use_fast
    )
    tokenizer.pad_token = tokenizer.eos_token

    """Data loading part"""
    def build_dataset(tokenizer, list_data_dict, max_length=512):
        """Build dataset for training. This builds the dataset from `load_dataset`, 
        one should customize this function to train the model on its own dataset.
        """
        num_proc = 4
        def preprocess_function(examples):
            new_examples = {
                "query": [],
                "input_ids": [],
            }
            for question in examples["question"]:
                query = "Question: " + question + "\n\nAnswer: "
                tokenized_question = tokenizer(query, truncation=True)
                new_examples["query"].append(query)
                new_examples["input_ids"].append(tokenized_question["input_ids"])

            return new_examples

        dataset = list_data_dict.map(
            preprocess_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=original_columns,
        )

        dataset = dataset.filter(lambda x: len(x["input_ids"]) < max_length, batched=False)
        dataset.set_format(type="torch")
        return dataset


    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    # Dataset and prompt
    prompt_path = os.path.join(data_args.prompt_dir, f"{data_args.dataset}_template.json")
    data_path = os.path.join(data_args.data_dir.format(data_args.dataset), f"{model_args.model_name}_align", f"{data_args.data_file}_{data_args.model_suffix}.json")
    logging.info(f"Loading data from {data_path} ...")

    list_data_dict = load_dataset("json", data_files=data_path, split="train")
    original_columns = list_data_dict.column_names
    prompt_template = read_json(prompt_path)
    instruction = prompt_template["instruction"]
    prompt_input = prompt_template["standard_prompt"]

    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_dataset(tokenizer, list_data_dict, training_args.max_seq_length)

    """LoRA setting."""
    peft_config = LoraConfig(
        r=lora_args.r,
        lora_alpha=lora_args.lora_alpha,
        # target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'],
        target_modules=find_all_linear_names(model),
        inference_mode=lora_args.inference_mode,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        task_type=lora_args.task_type,
    )

    # model.enable_input_require_grads() # Enable input gradients for LoRA, addressed by https://github.com/huggingface/peft/issues/137 @ Oct. 29th, 2024
    model = get_peft_model(model, peft_config)
    logging.info(print_trainable_parameters(model))

    """Reward model setting."""
    reward_model_path = model_args.reward_model_path
    reward_model = transformers.AutoModelForCausalLM.from_pretrained(reward_model_path,
                                                        torch_dtype=torch.float16,
                                                        quantization_config=bnb_config if model_args.bnb_use else None,
                                                        use_cache=model_args.use_cache,
                                                        device_map=model_args.device_map)

    """Parameters for training arguments details."""
    # Save path
    training_args.output_dir = training_args.output_dir.format(data_args.dataset)
    logging.info(f"Save model to {training_args.output_dir}")

    logging.info(f"Arguments:\nModel Arguments: {model_args}\nData Arguments: {data_args}\n \
                 Training Arguments: {training_args}\nLora Arguments: {lora_args}")

    # Prepare the PPO configuration
    config = PPOConfig(
        learning_rate=training_args.learning_rate,
        batch_size=training_args.batch_size,
        mini_batch_size=training_args.mini_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        output_dir=training_args.output_dir,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=training_args.early_stopping,
        target_kl=training_args.target_kl,
        ppo_epochs=training_args.ppo_epochs,
        seed=training_args.seed,
    )


    """Training part."""
    # Initialize the PPOTrainer, passing the model, the reference model, the tokenizer
    optimizer = None
    if training_args.adafactor:
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=config.learning_rate,
        )

    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
    )

    # Define the arguments to pass to the `generate` function. These arguments are passed to the `generate` function of the PPOTrainer, 
    # which is a wrapper around the `generate` function of the trained model.
    generation_kwargs = {
        # "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": 100_000,
    }
    output_min_length = 32
    output_max_length = training_args.output_max_length
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    logging.info("Start training ...")

    # Training loop
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        logging.info(f"Epoch {epoch+1}/{int(training_args.num_train_epochs)}")
        question_tensors = batch["input_ids"]

        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        reward_outputs = reward_model(texts)
        rewards = [torch.tensor(output[0]["score"]) for output in reward_outputs]

        # Run PPO step
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    ppo_trainer.save_pretrained(training_args.output_dir + f"step_{epoch}")

    logging.info("Training finished.")

if __name__ == "__main__":
    train()
