r"""
Author: XUE Boyang      Filename: sample.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Sampling-based categorization of knowledge.
"""
import os
import sys
import time
import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

from tqdm import tqdm
import json

import torch
import transformers
from peft import PeftModel
from torch.utils.data import Dataset
from transformers import set_seed, GenerationConfig, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

from utils import *
from eval import compute_scores


@dataclass
class ModelArguments:
    model_name: str = field(default="llama3", metadata={"help": "Model name.", "choices": model_path_dict.keys()})
    # LoRA setting
    lora_use: bool = field(default=False, metadata={"help": "Use LoRA or not."})
    lora_weights: str = field(default="./exp/{}/train", metadata={"help": "LoRA weights path."})
    model_suffix: str = field(default="no_lora", metadata={"help": "File name to save the results."})
    # Bits and Bytes config
    bnb_use: bool = field(default=False, metadata={"help": "Whether to use BitsAndBytesConfig."})
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to enable 4-bit quantization."})
    bnb_4bit_quant_type: str = field(default="nf4", metadata={"help": "Set the quantization data type in the bnb.nn.Linear4Bit layers", "choices": ["fp4", "nf4"]})
    bnb_4bit_compute_dtype: torch.dtype = field(default=torch.float16, metadata={"help": "Set the computational type which might be different than the input type."})
    # Tokenizer setting
    model_max_length: int = field(default=2048, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    use_fast: bool = field(default=False, metadata={"help": "Whether to use rust based fast tokenizer."})
    padding_side: str = field(default="right", metadata={"help": "Padding side for sequences with different length."})
    trust_remote_code: bool = field(default=False, metadata={"help": "Whether or not to allow for custom models defined on the Hub in their own modeling files."})


@dataclass
class DataArguments:
    data_dir: str = field(default="./data/{}/raw", metadata={"help": "Directory to save data."})
    dataset: str = field(default="triviaqa", metadata={"help": "Dataset name.", "choices": dataset_list})
    data_file: str = field(default="validation", metadata={"help": "Data file name.", "choices": ["validation", "train"]})
    prompt_dir: str = field(default="./prompt/", metadata={"help": "Path to the prompt."})
    continue_generate: bool = field(default=False, metadata={"help": "Continue from the previous generations."})
    sample_interval: int = field(default=1, metadata={"help": "Sample interval."})


@dataclass
class InferenceArguments:
    icl_type: str = field(default="icl", metadata={"help": "Use few-shot prompt or not."})
    output_dir: str = field(default="./data/{}/prep", metadata={"help": "Directory to save results."})
    data_suffix: str = field(default="sample_2k", metadata={"help": "File name to save the results."})
    num_sampling: int = field(default=2, metadata={"help": "Number of samples."})
    temperature: float = field(default=0.5, metadata={"help": "Temperature for sampling."})
    top_p: float = field(default=1.0, metadata={"help": "Top p for sampling."})
    top_k: int = field(default=40, metadata={"help": "Top k for sampling."})
    num_beams: int = field(default=1, metadata={"help": "Number of beams for sampling."})
    max_gen_length: int = field(default=24, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    repetition_penalty: float = field(default=1.1, metadata={"help": "Repetition penalty."})


@dataclass
class DeviceArguments:
    device: str = field(default="cuda", metadata={"help": "Device to use."})
    seed: int = field(default=3407, metadata={"help": "Random seed."})
    gpu_num: int = field(default=1, metadata={"help": "Number of GPUs."})
    local_rank: int = field(default=0, metadata={"help": "Local rank."})
    global_rank: int = field(default=0, metadata={"help": "Global rank."})
    world_size: int = field(default=0, metadata={"help": "World size."})


# Parse arguments.
parser = transformers.HfArgumentParser((ModelArguments, DataArguments, InferenceArguments, DeviceArguments))
model_args, data_args, infer_args, device_args = parser.parse_args_into_dataclasses()

# Resize tokenizer and embedding.
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


# Format the few-shot examplar of list to string.
def format_examplar(few_shot_examplars, examplar_split):
    few_shot_examplar_list = []
    for few_shot_examplar in few_shot_examplars.values():
        few_shot_examplas = []
        for few_shot_example in few_shot_examplar:
            few_shot_examplas.append("{}{}\n{}{}".format(examplar_split["input"], few_shot_example["question"], 
                                                         examplar_split["output"], few_shot_example["answer"]))
        few_shot_examplar_list.append("\n\n".join(few_shot_examplas))

    return few_shot_examplar_list


# Split the generation to get the answer part.
def output_split(output, tokenizer, split_len, prompt_split):
    return tokenizer.decode(output.sequences[0][split_len:], 
                            skip_special_tokens=True).split(prompt_split)[0].replace("\n", "").lstrip()


def sample():
    # import pdb; pdb.set_trace()
    # Info: Device settings: random seed, using cuda or not, distributed setting.
    set_seed(device_args.seed)

    device_args.num_gpu = torch.cuda.device_count()
    device_args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up logging.
    infer_args.output_dir = os.path.join(infer_args.output_dir.format(data_args.dataset), 
                                         "{}_{}_{}_{}".format(model_args.model_name,
                                                              data_args.dataset,
                                                              model_args.model_suffix,
                                                              "icl" if "icl" in infer_args.icl_type else "std"))

    if not os.path.exists(infer_args.output_dir):
        os.makedirs(infer_args.output_dir)

    infer_args.log_path = os.path.join(infer_args.output_dir, f"{data_args.data_file}_generate.log")

    logging.basicConfig(
        filename=infer_args.log_path,
        filemode='w',
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
    )

    # Load model and tokenizer.
    if model_args.bnb_use:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=model_args.bnb_4bit_compute_dtype
        )
        logging.info(f"BitsAndBytes use.")
    else:
        logging.info(f"No BitsAndBytes use.")

    model_name_or_path = model_path_dict[model_args.model_name]
    logging.info(f"Loading model and tokenizer from {model_name_or_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        torch_dtype=torch.float16,
        quantization_config=bnb_config if model_args.bnb_use else None,
        device_map="balanced" # device_map: "auto", "balanced", "balanced_low_0", "sequential"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        model_max_length=model_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=model_args.use_fast,
    )

    # Resize tokenizer and embedding.
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token == "":
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token == "":
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token == "":
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    if model_args.lora_use:
        model_args.lora_weights = os.path.join(model_args.lora_weights.format(data_args.dataset), 
                                         f"{model_args.model_name}_{data_args.dataset}_{model_args.model_suffix}")
        model = PeftModel.from_pretrained(
            model,
            model_id=model_args.lora_weights,
            torch_dtype=torch.float16
        )

    # Load data.
    # import pdb; pdb.set_trace()
    data_path = os.path.join(data_args.data_dir.format(data_args.dataset), f"{data_args.data_file}.json")
    logging.info(f"Loading data from {data_path} ...")
    data_pool = json.load(open(data_path))

    # Load prompt and select the prompt type.
    prompt_template = json.load(open(os.path.join(data_args.prompt_dir, 
                                                  f"{data_args.dataset}_template.json")))
    instruction = prompt_template["instruction"]
    prompt_split = prompt_template["output_split"]
    few_shot_examplars, examplar_split = prompt_template["sample_icl_examplar"], prompt_template["few_shot_split"]

    if "icl" in infer_args.icl_type:
        few_shot_examplars, examplar_split = prompt_template[f"generate_{infer_args.icl_type}_examplar"], prompt_template["few_shot_split"]
        few_shot_examplar_list = format_examplar(few_shot_examplars, examplar_split)
        prompt_input = prompt_template["few_shot_prompt"]
    else:
        prompt_input = prompt_template["standard_prompt"]

    # Format the output file.
    # import pdb; pdb.set_trace()
    infer_args.save_path = os.path.join(infer_args.output_dir, 
                                        f"{model_args.model_name}_{data_args.dataset}_{data_args.data_file}.json")
    if data_args.continue_generate:
        exist_num = len(read_jsonl(infer_args.save_path))
        # Split the dataset if needed.
        data_pool = data_pool[exist_num*data_args.sample_interval::data_args.sample_interval]
    else:
        data_pool = data_pool[::data_args.sample_interval]
        open(infer_args.save_path, "w").close()

    data_len = len(data_pool)

    logging.info(f"Arguments:\nModel Arguments: {model_args}\nData Arguments: {data_args}\nInference Arguments: {infer_args}")
    logging.info(f"The number of dataset: {data_len}")

    # Sample the data.
    start_time = time.time()
    logging.info("Start sampling ...")
    first_log_flag = False
    with tqdm(total=data_len) as t:
        for data_point in data_pool:
            outputs = []
            for examplar_id, few_shot_examplar in enumerate(few_shot_examplar_list):
                # import pdb; pdb.set_trace()
                # time.sleep(1)
                if infer_args.icl_type == "icl" or infer_args.icl_type == "icl_idk":
                    few_shot_examplar = random.choice(few_shot_examplar_list)
                    input_tokens = prompt_input.format(instruction=instruction, examples=few_shot_examplar, question=data_point["question"])
                else:
                    input_tokens = prompt_input.format(instruction=instruction, question=data_point["question"])

                # import pdb; pdb.set_trace()
                # time.sleep(1)
                input_ids = tokenizer(input_tokens, padding=True, return_tensors="pt")["input_ids"].to(device_args.device)

                with torch.no_grad():
                    # Greedy decoding by setting `do_sample=False`.
                    greedy_decoding_config = GenerationConfig(
                        do_sample=False,
                        repetition_penalty=1.0
                    )

                    greedy_output = model.generate(input_ids,
                                            generation_config=greedy_decoding_config,
                                            return_dict_in_generate=True,
                                            output_scores=True,
                                            max_new_tokens=infer_args.max_gen_length,
                                            pad_token_id=tokenizer.pad_token_id,
                                            eos_token_id=tokenizer.eos_token_id, 
                                            bos_token_id=tokenizer.bos_token_id,
                                            # return_legacy_cache=True
                                            )

                    # import pdb; pdb.set_trace()
                    greedy_output = output_split(greedy_output, tokenizer, len(input_ids[0]), prompt_split)

                    outputs.append({
                        "examplar_id": examplar_id+1,
                        "greedy_decoding": greedy_output
                    })

                    if first_log_flag:
                        logging.info("Input Prompt: \n{}".format(prompt_input.format(instruction=instruction, examples=few_shot_examplar, question=data_point["question"]) \
                                                                if "icl" in infer_args.icl_type else prompt_input.format(instruction=instruction, question=data_point["question"])))
                        logging.info("LLM Generation: \n{}".format(greedy_output))
                        first_log_flag = True

            greedy_scores, greedy_average = compute_scores(outputs, decoding="greedy", gold_answer=data_point["answer"], dataset=data_args.dataset)

            instance = {
                "question_id": data_point["question_id"],
                "question": data_point["question"],
                "answer": data_point["answer"],
                "outputs": outputs,
                "scores": {
                    "greedy_scores": greedy_scores,
                    "greedy_scores_avg": greedy_average
                }
            }
            # print(instance)

            # Real-time saving the results.
            with open(infer_args.save_path, "a+") as fw: 
                instance_write = json.dumps(obj=instance, ensure_ascii=False)
                fw.write(instance_write + '\n')

            t.set_postfix()
            t.update(1)
    
    # import pdb; pdb.set_trace()
    elapsed_time = format_seconds(time.time() - start_time)
    logging.info(f"Total elapsed time: {elapsed_time[0]}h {elapsed_time[1]}m {elapsed_time[2]}s")

    # Convert jsonl to json format.
    logging.info("Sampling is done.")
    jsonl2json(infer_args.save_path, infer_args.save_path)
    logging.info(f"Save to {infer_args.save_path}")


if __name__ == "__main__":
    sample()


