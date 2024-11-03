from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import os
import time
import sys
import json
from langchain.globals import set_verbose
from tqdm import tqdm
set_verbose(True)

model_name = "llama3"
metric_name = 'answer_correctness'
project_name = f"ragas_answer_correctness_{model_name}"
_n = -1
K = 5
prior_knowledge_version = "1" 
data = []
claims = []

from langchain.callbacks.tracers import LangChainTracer
tracer = LangChainTracer(project_name=project_name)

import prompt_rag, covmis
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_correctness
)

import warnings
warnings.filterwarnings("ignore")
search_engine = "brave"

data_train = covmis.load_data('entire')
data_search_llm = covmis.load_data_llm('entire', None)
data_search = covmis.load_data_search('entire', None)

chat_model = ChatOpenAI(
    model_name="gpt-3.5-turbo-0125",
    temperature=0,
)
llm = LangchainLLMWrapper(chat_model)
answer_correctness.llm = llm

def get_lst_sent(s, model_name):
    assert model_name in ["mixtral", "llama3", "solar"], "model name error"

    lines = []
    for line in s.split('\n'):
        if len(line) > 0:
            lines.append(line)
    
    labels = [
        "FALSE", "TRUE",
        ]
    for line in lines[::-1]:
        for label in labels:
            _pos = line.find(label)
            if _pos != -1:
                return line

    return lines[-1]

for i in range(len(data_search_llm)):

    item = data_search_llm[i]
    claim = data_train[i]["claim"]
    label = int(data_train[i]["label"])
    if not isinstance(label, int):
        raise Exception(f"Label is not int.\n{claim}")

    search_results = data_search[i][f"{search_engine}_search_results"]
    question = prompt_rag.get_prompt_for_generating_prior_knowledge(
        data_train[i]["claim"], data_train[i]["date"], search_engine, search_results, model_name,
        K=K, sort=False, without_info=True, without_claim_date=True
    )
    
    answer = item[f"prior_knowledge_{model_name}_v{prior_knowledge_version}_K={K}"]
    answer = get_lst_sent(answer, model_name=model_name)


    if label == 2:
        ground_truth = f'The claim({data_train[i]["claim"]}) is true.'
    elif label == 0:
        ground_truth = f'The claim({data_train[i]["claim"]}) is false.'
    else:
        pass

    data_i = {
        'question': [question.lower()],
        'answer': [answer.lower()],
        'ground_truth': [ground_truth.lower()]
    }
    claims.append(claim)
    data.append(data_i)

ragas_metric = []
with open(f"output/{search_engine}_{model_name}_v1_{metric_name}.json", "r") as f:
    ragas_metric = json.load(f)

if _n == -1:
    _n = len(data)

def save_metric(ragas_metric):
    print("Saving......")
    with open(f"output/{search_engine}_{model_name}_v1_{metric_name}.json", "w") as f:
        json.dump(ragas_metric, f, indent=4)
    print("Saved!")

for i in tqdm(range(_n)):
    item = ragas_metric[i]
    if item['claim'].lower() not in claims[i].lower():
        raise Exception(f"{i}\n{item['claim']}\n")
    
    if item.get(metric_name) is not None:
        continue

    if i % 5 == 0:
        save_metric(ragas_metric)

    if item["label"] == 1:
        ragas_metric[i][metric_name] = -1
        continue

    # time.sleep(1)
    dataset  = Dataset.from_dict(data[i])
    result = evaluate(
        dataset,
        metrics=[
            answer_correctness,
        ],
        callbacks=[tracer],
    )
    ragas_metric[i][metric_name] = result[metric_name]

save_metric(ragas_metric)
