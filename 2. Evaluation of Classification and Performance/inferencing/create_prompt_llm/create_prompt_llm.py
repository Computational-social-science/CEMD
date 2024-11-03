import json
import prompt_rag
import torch
import os
import sys

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--data_type", type=str)
args = parser.parse_args()
data_type = args.data_type

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

dirs = ["../..", ".."]
for _dir in dirs:
    if _dir not in sys.path:
        sys.path.append(_dir)

import covmis, liar2

from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_vllm, VllmGenerationConfig
)
from custom import CustomModelType

# data_type = 'train'
if __name__ == '__main__':
    model_type = CustomModelType.llama_3_70b_instruct_awq
    llm_engine = get_vllm_engine(
        model_type, 
        tensor_parallel_size=2,
        max_model_len=4096,
        gpu_memory_utilization=0.92,
        max_num_seqs=64,
        engine_kwargs = {
            "seed": 42,
        }
    )

    template_type = get_default_template_type(model_type)
    template = get_template(template_type, llm_engine.hf_tokenizer)

    generation_config = VllmGenerationConfig(
        max_tokens=2048,
        temperature=0,
        seed=42
    )

    get_resp_list = lambda request_list : inference_vllm(
        llm_engine, template, request_list, 
        generation_config=generation_config, 
        use_tqdm=True, 
    )

    K = 5
    sort = False

    prior_knowledge_version = "1"
    search_engine = "brave"
    model_name = 'llama3'
    dataset = 'liar2' # liar2 covmis
    wsc_version = "1"
    search_date = None

    prompt_rag.update_train_search_llm(
        model_name, get_resp_list, search_engine,
        dataset, prior_knowledge_version,
        data_type=data_type, search_date=search_date,
        K=K, sort=sort, wsc_version=wsc_version
    )
