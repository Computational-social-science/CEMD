from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import os
import sys
import json
from langchain.globals import set_verbose
from tqdm import tqdm
set_verbose(True)

model_name = "llama3"
metric_name = 'faithfulness'
project_name = "default"
_n = -1
K = 5
prior_knowledge_version = "1"
data = []

from langchain.callbacks.tracers import LangChainTracer
tracer = LangChainTracer(project_name=project_name)

import prompt_rag, covmis
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
)

import warnings
warnings.filterwarnings("ignore")
search_engine = "brave"

data_train = covmis.load_data('entire')
data_search_llm = covmis.load_data_llm('entire')
data_search = covmis.load_data_search('entire', None)

chat_model = ChatOpenAI(
    model_name="gpt-3.5-turbo-0125",
    temperature=0,
)
llm = LangchainLLMWrapper(chat_model)
faithfulness.llm = llm

for i in range(len(data_search_llm)):
    search_results = data_search[i][f"{search_engine}_search_results"]

    if prior_knowledge_version.startswith("1"):
        ids = slice(0, 5)
    elif prior_knowledge_version.startswith("2"):
        ids = data_search[i]["random_ids"]
    else:
        raise Exception()
    
    contexts = prompt_rag.get_brave_snippet(search_results, ids=ids, ret_type='list')
    question = prompt_rag.get_prompt_for_generating_prior_knowledge(
        data_train[i]["claim"], data_train[i]["date"], search_engine, search_results, model_name,
        K=K, sort=False, ids=ids, without_info=True
    )
    
    answer = data_search_llm[i][f"prior_knowledge_{model_name}_v{prior_knowledge_version}_K={K}"]

    data_i = {
        'question': [question],
        'answer': [answer],
        'contexts' : [contexts],
    }
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
    if item['claim'] not in data[i]['question'][0]:
        raise Exception(f"{i}\n{item['claim']}\n{data[i]['question']}")
    
    if item.get('faithfulness') is not None:
        continue

    if item["label"] == 1:
        ragas_metric[i][metric_name] = -1

        save_metric(ragas_metric)

        continue

    dataset  = Dataset.from_dict(data[i])
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
        ],
        callbacks=[tracer],
    )
    ragas_metric[i]['faithfulness'] = result["faithfulness"]

    # save
    save_metric(ragas_metric)



