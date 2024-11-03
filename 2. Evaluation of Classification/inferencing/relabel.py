import os
import sys
import json
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, 
)
from swift.tuners import Swift
from custom import CustomModelType, CustomTemplateType
import covmis, liar2
import prompt_rag

ckpt_dir_cvomis = ""
ckpt_dir_liar2 = ""
K = 5
prior_knowledge_version = "1"
model_name = "llama3"
ckpt_dir = ckpt_dir_liar2

with open(f"{ckpt_dir}/sft_args.json", "r") as f:
    sft_args = json.load(f)

def get_model_template():
    model_type, template_type = sft_args["model_type"], sft_args["template_type"]
    model, tokenizer = get_model_tokenizer(
        model_type, model_kwargs={'device_map': 'auto'},
        use_flash_attn=sft_args["use_flash_attn"]
    )
    model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)
    if sft_args["sft_type"] == 'adalora':
        model = model.to(model.dtype)
    model.generation_config.max_new_tokens = 512
    model.generation_config.do_sample = False

    template = get_template(template_type, tokenizer)

    return model, template

model, template = get_model_template()

label_convert_liar2 = {'pants-fire': 0, 'false': 1, 'barely-true': 2, 'half-true': 3, 'mostly-true': 4, 'true': 5}

search_engine = "brave"

dataset = "liar2" # liar2, covmis
data_type = "valid" # train, test, valid
search_date = None

if dataset == "covmis":
    data = covmis.load_data(data_type)
    data_search =  covmis.load_data_search(data_type, search_date, search_engine=search_engine)
    data_search_llm = covmis.load_data_llm(data_type, search_date, search_engine=search_engine)
    claim_key = 'claim'
    claimant_key = 'None'
    LABEL_TRUE = 2
    LABEL_FALSE = 0
    true_labels_original = [LABEL_TRUE]
    false_labels_original = [LABEL_FALSE]

    save_data = lambda data: covmis.save_train(data)
    save_search = lambda data: covmis.save_data_search(
        data, data_type, search_date, search_engine=search_engine)
    save_search_llm = lambda data: covmis.save_data_llm(
        data, data_type, search_date, search_engine=search_engine)
elif dataset == "liar2":
    data = liar2.load_data(data_type)
    data_search = liar2.load_data_search(data_type, search_engine)
    data_search_llm = liar2.load_data_llm(data_type, search_engine)
    claim_key = 'statement'
    claimant_key = 'None'
    true_labels_original = [
        label_convert_liar2['true'], 
        # label_convert_liar2['mostly-true'],
        # label_convert_liar2['half-true']
    ]
    
    false_labels_original = [
        # label_convert_liar2['barely-true'], 
        label_convert_liar2['false'],
        label_convert_liar2['pants-fire']
    ]

    save_data = lambda data: liar2.save_data(data, data_type)
    save_search = lambda data: liar2.save_data_search(
        data, data_type, search_engine)
    save_search_llm = lambda data: liar2.save_data_llm(
        data, data_type, search_engine)
else:
    raise Exception("dataset error")

for i in tqdm(range(len(data_search_llm))):
    item = data_search_llm[i]

    if data[i]["id"] != item["id"]:
        raise Exception("id error！")
    
    if int(data[i]["label"]) not in (true_labels_original + false_labels_original):
        prompt = prompt_rag.get_prompt_with_prior_knowledge(
            data[i][claim_key], 
            search_engine,
            data_search[i][f"{search_engine}_search_results"], 
            item[f"prior_knowledge_{model_name}_v{prior_knowledge_version}_K={K}"], 
            K=K,
            claim_date=data[i]["date"],
            justification=data[i].get('justification'),
            known_info=True, 
            rag_info=True,
            justify_info=False,
            ids=None
        )
        pred_raw = inference(model, template, prompt)[0].strip()

        if pred_raw.startswith("TRUE"):
            data[i]["label2"] = true_labels_original[0]
        elif pred_raw.startswith("FALSE"):
            data[i]["label2"] = false_labels_original[0]
        else:
            raise Exception(f"Error label: {pred_raw}")

    else:
        if data[i]["label"] in true_labels_original:
            data[i]["label2"] = true_labels_original[0]
        elif data[i]["label"] in false_labels_original:
            data[i]["label2"] = false_labels_original[0]
        else:
            raise Exception()
        
save_data(data)
