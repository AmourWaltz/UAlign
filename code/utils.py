import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union

import copy

from datetime import datetime
import pytz


# Constants
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

model_path_dict = {
    "llama31_ins": "/workspace/model/Meta-Llama-3.1-8B-Instruct",
    "llama3": "/workspace/model/Meta-Llama-3-8B",
    "qwen2": "/workspace/model/Qwen2-7B-Instruct",
    "mistral": "/workspace/model/Mistral-7B-v0.1"
}

dataset_list = ["triviaqa", "sciq", "nqopen", "lsqa", "comb"]
methods = ["prompt", "sft", "rtuning"]

"""
Json utils
"""
def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.
    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def read_json(filename):
    with open(filename, "r", encoding="utf-8") as fr:
        data_pool = json.load(fr)

    return data_pool


def read_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as fr:
        data_pool = [json.loads(line) for line in fr.readlines()]

    return data_pool


def write_jsonl(filename, dataset):
    with open(filename, "w", encoding="utf-8") as fw:
        fw.writelines([json.dumps(obj=ins, ensure_ascii=False) + '\n' for ins in dataset])


def write_json(filename, dataset):
    with open(filename, "w", encoding="utf-8") as fw:
        json.dump(fp=fw, obj=dataset, indent=4, ensure_ascii=False)


def jsonl2json(file1, file2):
    write_json(file2, read_jsonl(file1))


def json2jsonl(file1, file2):
    dataset = read_json(file1)
    write_jsonl(file2, dataset)


def json_merge(files, out_file):
    data = []
    for file in files:
        data += read_json(file)
    write_json(out_file, data)


def read_jsons(files):
    data = []
    for file in files:
        data += read_json(file)
    return data


# Time utils
def format_seconds(seconds):
    hours, remainder = divmod(seconds, 3600)  # 1h = 3600s
    minutes, seconds = divmod(remainder, 60)  # 1m = 60s
    return [int(hours), int(minutes), int(seconds)]


def get_current_time():
    tz = pytz.timezone('Asia/Shanghai')
    return datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S %Z%z')


"""
Model parameters utils
"""
def find_all_linear_names(model):
    import bitsandbytes as bnb
    cls = bnb.nn.Linear4bit
    # print(f"Finding all {cls} in model...")
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)


"""
Prints the number of trainable parameters in the model.
"""
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    train_paras = f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
    print(train_paras)
    return train_paras