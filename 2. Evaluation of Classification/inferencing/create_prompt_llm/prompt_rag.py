import json
from tqdm import tqdm
from datetime import datetime
import jsonlines

import sys
import covmis, liar2


def get_claim_with_date(claim, claim_date=None, claimant=None):
    if claim_date is None and claimant is None:
        return claim
    
    res = claim
    if claimant is not None:
        res += "\nClaimant: " + claimant
    if claim_date is not None:
        res += "\nPublication date: " + claim_date
    return res

def get_bing_snippet(bing_search_results, K=5, claim_date=None):
    if bing_search_results.get("webPages") is None:
        return ""
    
    results = bing_search_results["webPages"]["value"]
    snippets = [i["snippet"] for i in results][:K]

    snippet = ""

    for i, item in enumerate(snippets):
        snippet += f"{i+1}. " + item + '\n'

    return snippet

def get_bing_snippet_v2(bing_search_results, K, claim_date, sort):
    def get_snippets_dates(results, K=K):
        date_format = "%Y-%m-%d"
        date_claim = datetime.strptime(claim_date, date_format)
        
        if not sort:
            snippets = [i["snippet"] for i in results[:K]]
            dates = [i["datePublished"][:10] if i.get("datePublished") else "None" for i in results[:K]]
            return snippets, dates
        else:
            max_k = 20
            if K > max_k:
                raise Exception(f"K exceeds {max_k}")
            id_delta_list = []
            results_10 = []
            need_supplement = False
            for i in results:
                if i.get("datePublished") is not None:
                    results_10.append(i)
                
                if len(results_10) == max_k:
                    break
            if len(results_10) < K:
                need_supplement = True

            for id, item in enumerate(results_10):
                date_item = datetime.strptime(item["datePublished"][:10], date_format)
                delta = abs((date_item - date_claim).days)
                id_delta_list.append((id, delta))

            id_delta_list_sorted = sorted(id_delta_list, key=lambda x: x[1])
            snippets = [results_10[i[0]]["snippet"] for i in id_delta_list_sorted[:K]]
            dates = [results_10[i[0]]["datePublished"][:10] for i in id_delta_list_sorted[:K]]
            
            if need_supplement:
                for i, item in enumerate(results):
                    if i in [x[0] for x in id_delta_list_sorted]:
                        continue
                    
                    snippets.append(item["snippet"])
                    dates.append("None")
                    if len(snippets) == K:
                        break

            return snippets, dates

    if bing_search_results.get("webPages") is None:
        return ""
    
    results = bing_search_results["webPages"]["value"]
    snippets, dates = get_snippets_dates(results)

    snippet = ""

    for i, item in enumerate(snippets):
        snippet += f"{i+1}.\n" + "Publication date: " + dates[i] + '\n' + "Content: " + item + '\n'

    return snippet

def get_brave_snippet(search_results, ids: slice | list, ret_type='str'):
    results = search_results["web"]["results"]
    snippet = ""
    snippets = []
    if isinstance(ids, slice):
        results_filtered = results[ids]
        id_start = ids.start
    elif isinstance(ids, list):
        results_filtered = [results[i] for i in ids]
        id_start = 0
    else:
        raise Exception()
    
    for i, item in enumerate(results_filtered):

        date = item.get("page_age", "None")
        if date != "None":
            date = date[:10]

        extra_snippets = item.get("extra_snippets")
        if extra_snippets:
            content = "\n".join(extra_snippets)
        else:
            content = item["description"]

        one_info = f"Information {id_start + i + 1}:\n" + "Publication date: " + date + '\n' + \
            "Title: " + item["title"] + '\n' + "Content:\n" + content + '\n'
        
        if ret_type == 'str':
            snippet += one_info
        elif ret_type == 'list':
            snippets.append(one_info.strip())
        else:
            raise Exception()
        
    if ret_type == "str":
        return snippet
    else:
        return snippets

def get_prompt_for_generating_prior_knowledge(
        claim, claim_date, search_engine, search_results, model_name,
        K=5, claimant=None, sort=False, ids=None, without_info=False, without_claim_date=False, n_truncate=0):
    """
    pre + info + text
    """

    claim = claim.strip()

    if model_name == "solar":
        pre = "Below is some INFORMATION searched online and a CLAIM. These pieces of INFORMATION are relevant to the CLAIM. This CLAIM and all INFORMATION include their respective publication dates and contents. To classify the CLAIM more accurately (if the content described by the CLAIM is correct, it will be classified as TRUE; if the content described by the CLAIM is incorrect, it will be classified as FALSE), please first expand on the given INFORMATION and provide a detailed summary of it. Then analyze, reason, and provide reasonable evidence to judge the correctness of the CLAIM based on the available information and your knowledge, and finally generate prior knowledge that helps classify the CLAIM.\n\n"
    elif model_name == "mixtral":
        pre = "Below is some INFORMATION searched online and a CLAIM. These pieces of INFORMATION are relevant to the CLAIM. This CLAIM and all INFORMATION include their respective publication dates and contents. To classify the CLAIM more accurately (if the content described by the CLAIM is correct, it will be classified as TRUE; if the content described by the CLAIM is incorrect, it will be classified as FALSE), please first provide a clear summary of the given INFORMATION, and provide reasonable evidence to judge the correctness of the CLAIM based on the available information and your knowledge.\n\n"    
    elif model_name == "llama3":
        pre = "Below is some INFORMATION searched online and a <CLAIM>. These pieces of INFORMATION are relevant to the <CLAIM>. This <CLAIM> and all INFORMATION include their respective publication dates and contents. To classify the <CLAIM> more accurately (if the content described by the <CLAIM> is correct, it will be classified as TRUE; if the content described by the <CLAIM> is incorrect, it will be classified as FALSE), please first provide a clear summary of the given INFORMATION, and provide reasonable evidence to judge the correctness of the <CLAIM> based on the available information and your knowledge.\n\n"
    else:
        raise Exception()
    
    if without_claim_date:
        text = "CLAIM: " + claim
    else:
        if model_name == "solar":
            text = "CLAIM: "
        if model_name == "mixtral":
            text = "CLAIM: "
        elif model_name == "llama3":
            text = "<CLAIM>: "
        text += get_claim_with_date(claim, claim_date, claimant)

    if search_engine == "bing":
        snippet = get_bing_snippet_v2(search_results, K=K, claim_date=claim_date, sort=sort)
    elif search_engine == "brave":
        if ids is None:
            ids = slice(0, K)
        snippet = get_brave_snippet(search_results, ids=ids)
        if n_truncate > 0:
            snippet = snippet[:-n_truncate]
    else:
        raise Exception("Select search engines in [\"bing\", \"brave\"].")
    
    info = "INFORMATION:\n" + snippet + '\n'
    
    if without_info:
        return (pre + text).strip()
    else:
        return pre + info + text

def get_claim_id(claim, data_search):
    for i in range(len(data_search)):
        if claim.strip() in data_search[i]["claim"].strip():
            return i

def get_prompt_with_prior_knowledge(
        claim, search_engine, search_results, 
        prior_knowledge, K=5, claim_date=None, claimant=None, justification=None,
        known_info=True, rag_info=True, justify_info=False, 
        ids=None):

    if justify_info and justification is None:
        raise Exception()
    
    claim = claim.strip()
    pre = "Below is a CLAIM and the PRIOR KNOWLEDGE associated with it. Please classify the CLAIM as TRUE or FALSE based on the PRIOR KNOWLEDGE. If the content described by the CLAIM is correct, then classify it as TRUE; if the content described by the CLAIM is incorrect, then classify it as FALSE.\n\n"
    text = "CLAIM: " + get_claim_with_date(claim, claim_date, claimant) +'\n\n'

    if search_engine == "brave":
        if ids is None:
            ids = slice(0, K)
        snippet = get_brave_snippet(search_results, ids=ids)
    else:
        raise Exception("Select search engines in [\"brave\"].")

    if justify_info:
        assert not(known_info or rag_info), "Error"

        return pre + text + "PRIOR KNOWLEDGE:\n" + justification.strip()
    else:
        if known_info and rag_info:
            return pre + text + "PRIOR KNOWLEDGE:\n" + snippet + '\n' + prior_knowledge.strip()
        elif rag_info:
            return pre + text + "PRIOR KNOWLEDGE:\n" + prior_knowledge.strip()
        elif known_info:
            return pre + text + "PRIOR KNOWLEDGE:\n" + snippet
        else:
            raise Exception()


def save_search_llm_tmp(x, dataset, data_type, search_date):
    with open(f"data_search_llm_tmp_{dataset}_{data_type}_{search_date}.json", "w") as f:
        json.dump(x, f, indent=4)

def update_train_search_llm(
        model_name, get_resp_list, search_engine, dataset, prior_knowledge_version,
        data_type, search_date, K=5, sort=False, use_random=False, wsc_version='none'):
    data_llm_tmp = []
    
    if dataset == "covmis":
        data = covmis.load_data(data_type)
        data_search =  covmis.load_data_search(data_type, search_date, search_engine=search_engine)
        data_llm =  covmis.load_data_llm(data_type, search_date, search_engine=search_engine)
        claim_key = 'claim'
        claimant_key = 'None'
    elif dataset == "liar2":
        data = liar2.load_data(data_type)
        data_search = liar2.load_data_search(data_type, search_engine)
        data_llm = liar2.load_data_llm(data_type, search_engine)
        claim_key = 'statement'
        claimant_key = 'None'
    else:
        raise Exception("dataset error")

    prompt_list = []
    id_list = []
    data_llm_key = f"prior_knowledge_{model_name}_v{prior_knowledge_version}_K={K}"

    for i in range(len(data_search)):
        if data[i]["id"] != data_search[i]["id"]:
            raise Exception("id error")
        if data_llm[i].get(data_llm_key) is not None:
            continue

        if use_random:
            if K != 5 or sort:
                raise Exception()
            ids = data_search[i]["random_ids"]
        else:
            ids = None

        prompt = get_prompt_for_generating_prior_knowledge(
            data[i][claim_key], data[i]["date"], 
            search_engine, data_search[i][f"{search_engine}_search_results"], 
            model_name, K=K, claimant=data[i].get(claimant_key), sort=sort, ids=ids
        )
        prompt_list.append(prompt)
        id_list.append(data[i]["id"])

    request_list = [{'query': prompt} for prompt in prompt_list]
    resp_list = get_resp_list(request_list)
    resp_list = [i["response"] for i in resp_list]

    i_resp = 0
    for i in range(len(data_search)):
        if data_llm[i].get(data_llm_key) is not None:
            continue
        
        if id_list[i_resp] != data_search[i]["id"]:
            raise Exception()
        
        item_llm = {}
        item_llm["id"] = data_search[i]["id"]
        item_llm[f"prior_knowledge_{model_name}"] = resp_list[i_resp].strip()
        data_llm_tmp.append(item_llm)
        i_resp += 1
    save_search_llm_tmp(data_llm_tmp, dataset, data_type, search_date)
