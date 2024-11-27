r"""
Author: XUE Boyang      Filename: train_dpo.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Direct Preference Optimization on QA dataset.
"""
import os
import json
import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from datasets import load_dataset
from trl import SFTTrainer, DPOTrainer
from transformers import Trainer
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import *

# Constants
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name: str = field(default="llama3", metadata={"help": "Model name.", "choices": ["llama3", "vicuna"]})
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    divice_map: str = field(default="balanced", metadata={"help": "Device map for the model."})


@dataclass
class DataArguments:
    data_dir: str = field(default="./data/{}/prep", metadata={"help": "Directory to save data."})
    dataset: str = field(default="triviaqa", metadata={"help": "Dataset name.", "choices": ["triviaqa", "webqa", "fast", "gsm8k"]})
    data_file: str = field(default="train_2w", metadata={"help": "Data file name."})
    prompt_dir: str = field(default="./prompt/", metadata={"help": "Path to the prompt."})
    continue_generate: bool = field(default=False, metadata={"help": "Continue from the previous generations."})
    save_suffix: str = field(default="sft", metadata={"help": "File name to save the results."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # Maximal input sequence length
    max_seq_length: int = field(default=2048, metadata={"help": "The maximum total input sequence length after tokenization."})
    # Checkpoints saving directory and strategy
    output_dir: str = field(default="./exp/{}/train", metadata={"help": "The output directory where the model predictions and checkpoints will be written."})
    save_strategy: str = field(default="epoch", metadata={"help": "The checkpoint save strategy."})
    save_total_limit: int = field(default=1, metadata={"help": "Limit the total amount of checkpoints."})
    # Logging settings
    logging_dir: str = field(default="./logs", metadata={"help": "Log directory."})
    logging_steps: int = field(default=200, metadata={"help": "Log every X updates steps."})
    # Optimizer settings
    optim: str = field(default="adamw_torch", metadata={"help": "Optimizer to use."})
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
    fp16: bool = field(default=True, metadata={"help": "Whether to use fp16 for training."})
    bf16: bool = field(default=False, metadata={"help": "Whether to use bfloat16 for training."})
    # Evaluation settings
    eval_strategy: str = field(default="no", metadata={"help": "The evaluation strategy to adopt during training."})


@dataclass
class LoraArguments:
    r: int = field(default=16, metadata={"help": "The number of bits to quantize."})
    lora_alpha: int = field(default=16, metadata={"help": "The number of bits to quantize."})
    lora_dropout: float = field(default=0.05, metadata={"help": "The dropout rate for LORA."})
    bias: str = field(default="none", metadata={"help": "The bias type for LORA."})
    task_type: str = field(default="CAUSAL_LM", metadata={"help": "The task type for LORA."})


"""Resize tokenizer and embedding.
Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
"""
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


"""Data loader module for supervised fine-tuning."""
def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


"""Dataset for supervised fine-tuning."""
class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, prompt_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = read_json(data_path)

        logging.warning("Formatting inputs...")
        prompt_template = read_json(prompt_path)
        instruction = prompt_template["instruction"]
        prompt_input = prompt_template["standard_prompt"]

        prompts = []
        for example in list_data_dict:
            prompt_input_dict = {
                    "instruction": instruction,
                    "question": example['question']
                }
            prompts.append(prompt_input.format(**prompt_input_dict))
            data = {
                "prompt": prompt_input.format(**prompt_input_dict),
                "chosen": example['chosen'],
                "rejected": example['rejected'],
            }
            data_pool.append(data)

        
        logging.warning("Tokenizing inputs... This may take some time...")

        self.prompts = data_dict["input_ids"]
        self.chosens = data_dict["labels"]
        self.rejects = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(prompt=self.prompts[i], chosen=self.chosens[i], rejected=self.rejects[i])


"""Collate examples for supervised fine-tuning."""
@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


"""Make dataset and collator for supervised fine-tuning."""
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_path, prompt_path) -> Dict:
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path, prompt_path=prompt_path)
    logging.info(f"The number of training samples: {len(train_dataset)}")
    return train_dataset


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # Set up logging.
    training_args.output_dir = os.path.join(training_args.output_dir.format(data_args.dataset), 
                                         f"{model_args.model_name}_{data_args.dataset}_{data_args.data_file}_{data_args.save_suffix}")
    
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

    # Load model and tokenizer
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model_name_or_path = model_path_dict[model_args.model_name]
    logging.info(f"Loading model and tokenizer from {model_name_or_path} ...")
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                                            torch_dtype=torch.float16, 
                                                            quantization_config=bnb_config,
                                                            device_map=model_args.divice_map
                                                            )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # Change the LORA hyperparameters accordingly to fit your use case
    peft_config = LoraConfig(
        r=lora_args.r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        task_type=lora_args.task_type,
    )

    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=False
    )

    # import pdb; pdb.set_trace()
    special_tokens_dict = dict()
    logging.info("Smart tokenizer and embedding resizing ...")
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token == "":
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token == "":
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token == "":
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # print(f"Special tokens: pad - {tokenizer.pad_token}; eos - {tokenizer.eos_token};"
    #       f" bos - {tokenizer.bos_token}; unk - {tokenizer.unk_token}")

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    # import pdb; pdb.set_trace()
    # Load dataset and prompt
    # dataset = load_dataset("json", data_files=data_path, split="train")
    prompt_path = os.path.join(data_args.prompt_dir, f"{data_args.dataset}_template.json")
    data_path = os.path.join(data_args.data_dir.format(data_args.dataset), f"{model_args.model_name}_{data_args.dataset}_{data_args.data_file}.json")
    logging.info(f"Loading data from {data_path} ...")
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_path=data_path, prompt_path=prompt_path)
    # import pdb; pdb.set_trace()    
    
    # Save path
    output_dir = training_args.output_dir.format(data_args.dataset)
    logging.info(f"Save model to {output_dir}")

    # Parameters for training arguments details
    training_args = TrainingArguments(
        output_dir=output_dir
    )

    logging.info(f"Arguments:\nModel Arguments: {model_args}\nData Arguments: {data_args}\n \
                 Inference Arguments: {training_args}\nLora Arguments: {lora_args}")

    # Sample the data.
    logging.info("Start training ...")

    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=0.1,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_prompt_length=1024,
        max_length=2048,
    )
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=output_dir)


dpo_trainer.train()
dpo_trainer.save_model(output_dir)


output_dir = os.path.join(output_dir, "final_checkpoint")
dpo_trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train()

