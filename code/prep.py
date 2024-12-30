r"""
Author: XUE Boyang      Filename: prep.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Data preparation: parse, preprocess, and save. 
Data output format: 
    {
        "question_id": "question_id",
        "question": "question",
        "answer": "answer"
    }
"""
import os
import argparse
import json

import datasets
import pandas as pd

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="triviaqa", choices=dataset_list)
parser.add_argument('--method', type=str, default="rtuning", choices=methods)
parser.add_argument('--output_dir', type=str, default="./data/{}/raw")
args = parser.parse_args()


"""
TriviaQA dataset preparation and saving
"""
def prep_triviaqa_dataset(split="validation"):
    print(f'Preprocessing TriviaQA {split} dataset')
    data_pool = datasets.load_dataset("trivia_qa", "rc.nocontext", split=split)
    id_mem = set()

    def remove_dups(batch):
        if batch['question_id'][0] in id_mem:
            return {_:[] for _ in batch.keys()}
        id_mem.add(batch['question_id'][0])

        return batch

    data_pool = data_pool.map(remove_dups, batch_size=1, batched=True, 
                            load_from_cache_file=False, remove_columns=["search_results", "question_source", "entity_pages"])

    # Warrant the duplicated data was removed
    assert pd.Series([_['question_id'] for _ in data_pool]).value_counts().max() == 1

    data_set = []
    for data in data_pool:
        # import pdb; pdb.set_trace()
        instance = {
            "question_id": data["question_id"],
            "question": data["question"],
            "answer": data["answer"]["value"]
        }
        data_set.append(instance)
    
    print(f"Data size of {split}: {len(data_set)}")

    return data_set


def get_triviaqa_dataset(output_dir):
    # Get data splits
    data_splits = ["train", "validation", "test"]

    for split in data_splits:
        save_path = output_dir + f"/{split}.json"
        data_set = prep_triviaqa_dataset(split)

        with open(save_path, "w") as fw:
            json.dump(fp=fw, obj=data_set, indent=4, ensure_ascii=False)


"""
WebQA dataset preparation and saving
"""
def prep_webqa_dataset(data_path, split="validation"):
    print(f'Preprocessing WebQA {split} dataset')
    data_pool = json.load(open(data_path, "r"))
    print(f"Original data size of {split}: {len(data_pool)}")

    # import pdb; pdb.set_trace()
    data_set = []
    for key, value in data_pool.items():
        answers = []
        for answer in value["evidences"].values():
            if answer["answer"][0] != "no_answer":
                # import pdb; pdb.set_trace()
                answers.append(answer["answer"][0])

        # import pdb; pdb.set_trace()
        # print(answers)
        if answers:
            instance = {
                "question_id": key,
                "question": value["question"],
                "answer": max(answers, key=answers.count)
            }
            data_set.append(instance)
    
    print(f"Processed data size of {split}: {len(data_set)}")

    return data_set


def get_webqa_dataset(output_dir):
    # Get data splits
    data_splits = ["me_train", "me_validation.ir", "me_validation.ann", "me_test.ir", "me_test.ann"]

    for split in data_splits:
        save_path = output_dir + f"/{split}.json"
        data_path = os.path.join("./../../data/WebQA.v1.0", f"{split}.json")
        data_set = prep_webqa_dataset(data_path, split)

        with open(save_path, "w") as fw:
            json.dump(fp=fw, obj=data_set, indent=4, ensure_ascii=False)


"""
SciQ dataset preparation and saving
"""
def prep_sciq_dataset(split="validation"):
    print(f'Preprocessing SciQ {split} dataset')
    
    val_data = datasets.load_dataset("sciq", split=split)
    ins_set = []
    for idx, data in enumerate(val_data):
        # import pdb; pdb.set_trace()
        ins = {
            "question_id": str(idx+1),
            "question": data["question"],
            "answer": data["correct_answer"]
        }
        ins_set.append(ins)

    print(f"Processed data size of {split}: {len(ins_set)}")

    return ins_set


def get_sciq_dataset(output_dir):
    # Get data splits
    data_splits = ["train", "validation"]

    for split in data_splits:
        save_path = output_dir + f"/{split}.json"
        data_set = prep_sciq_dataset(split)

        with open(save_path, "w") as fw:
            json.dump(fp=fw, obj=data_set, indent=4, ensure_ascii=False)


"""
NQ-Open dataset preparation and saving
"""
def prep_nqopen_dataset(split="validation"):
    print(f'Preprocessing NQ-Open {split} dataset')
    data_path = os.path.join(f"./data/nqopen/raw/NQ-open.{split}.jsonl")
    # import pdb; pdb.set_trace()
    data_pool = read_jsonl(data_path)

    print(f"Original data size of {split}: {len(data_pool)}")
    
    ins_set = []
    for idx, data in enumerate(data_pool):
        # import pdb; pdb.set_trace()
        ins = {
            "question_id": str(idx+1),
            "question": data["question"],
            "answer": data["answer"][0]
        }
        ins_set.append(ins)

    return ins_set


def get_nqopen_dataset(output_dir):
    # Get data splits
    data_splits = ["train", "dev"]

    for split in data_splits:
        save_path = output_dir + f"/{split}.json"
        data_set = prep_nqopen_dataset(split)

        with open(save_path, "w") as fw:
            json.dump(fp=fw, obj=data_set, indent=4, ensure_ascii=False)


"""
R-tuning training dataset preparation
"""
def get_rtuning_dataset():
    dataset_comb = ["triviaqa", "sciq", "nqopen"]
    models = ["llama3", "mistral"]

    input_file = "./data/{}/prep/{}_{}_vanilla_icl/{}_{}_train.json"
    output_file = "./data/comb/raw/train_{}_rtuning.json"

    for model_use in models:
        ins_set = []
        save_path = output_file.format(model_use)
        for dataset_use in dataset_comb:
            data_path = input_file.format(dataset_use, model_use, dataset_use, model_use, dataset_use)
            data_pool = read_json(data_path)
            print(f"Data size of {model_use}_{dataset_use}: {len(data_pool)}")
            for data in data_pool:
                # import pdb; pdb.set_trace()
                ins = {
                    "question_id": data["question_id"],
                    "question": data["question"],
                }
                if data["scores"]["greedy_scores_avg"] == 0:
                    ins["answer"] = "sorry, I don't know."
                else:
                    ins["answer"] = data["answer"]
                    
                ins_set.append(ins)

        write_json(save_path, ins_set)
        print(f"Data size of {model_use}: {len(ins_set)}")
        print(f"Examples: {read_json(save_path)[0:10]}")

"""
Main part
"""
if __name__=="__main__":
    if args.method == "prompt":
        # Output directory
        output_dir = args.output_dir.format(args.dataset)
        print(f"Data saved to {output_dir}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if args.dataset == "triviaqa":
            get_triviaqa_dataset(output_dir)
        elif args.dataset == "webqa":
            get_webqa_dataset(output_dir)
        elif args.dataset == "sciq":
            get_sciq_dataset(output_dir)
        elif args.dataset == "nqopen":
            get_nqopen_dataset(output_dir)
    elif args.method == "rtuning":
        get_rtuning_dataset()
    

