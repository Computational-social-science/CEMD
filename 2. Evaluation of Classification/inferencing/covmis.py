import json

version_list = ["2", "original"]

type_list = ["train", "valid", "test", "entire"]
def get_data_dir(data_type):
    if data_type == "entire":
        return "../../data/COVMIS2/data.json"
    else:
        return f"../../data/COVMIS2/training/{data_type}.json"

def get_data_llm_dir(data_type, search_date, search_engine="brave"):
    if data_type == "entire":
        return f"../../data/COVMIS2/data_{search_engine}_search_llm.json"
    else:
        return f"../../data/COVMIS2/training/{data_type}_{search_engine}_search_llm/{search_date}.json"

def get_data_search_dir(data_type, search_date, search_engine="brave"):
    if data_type == "entire":
        return f"../../data/COVMIS2/data_{search_engine}_search.json"
    else:
        return f"../../data/COVMIS2/training/{data_type}_{search_engine}_search/{search_date}.json"


def load_data(data_type, version="2"):
    """
    type: train, valid, test, entire
    """
    assert data_type in type_list, f"data_type is not in {type_list}"

    if version == "2":
        with open(get_data_dir(data_type), "r") as f:
            return json.load(f)
    elif version == "original":
        if data_type != 'entire':
            raise Exception()
        with open("../../data/COVMIS2/data_original.json", "r") as f:
            return json.load(f)
    else:
        raise Exception(f"version is not in {version_list}")

def save_data(data, data_type, version="2"):
    assert data_type in type_list, f"data_type is not in {type_list}"

    if version == "2":
        with open(get_data_dir(data_type), "w") as f:
            json.dump(data, f, indent=4)
    else:
        raise Exception()
    
def load_data_llm(data_type, search_date, version="2", search_engine="brave"):
    assert data_type in type_list, f"data_type is not in {type_list}"

    if version == "2" and search_engine == "brave":
        with open(get_data_llm_dir(data_type, search_date, search_engine), "r") as f:
            return json.load(f)
    else:
        raise Exception()

def save_data_llm(data, data_type, search_date, version="2", search_engine="brave"):
    assert data_type in type_list, f"data_type is not in {type_list}"

    if version == "2" and search_engine == "brave":
        with open(get_data_llm_dir(data_type, search_date, search_engine), "w") as f:
            json.dump(data, f, indent=4)
    else:
        raise Exception()

def load_data_search(data_type, search_date, version="2", search_engine="brave"):
    assert data_type in type_list, f"data_type is not in {type_list}"

    if version == "2" and search_engine == "brave":
        with open(get_data_search_dir(data_type, search_date, search_engine), "r") as f:
            return json.load(f)
    else:
        raise Exception()

def save_data_search(data, data_type, search_date, version="2", search_engine="brave"):
    assert data_type in type_list, f"data_type is not in {type_list}"

    if version == "2" and search_engine == "brave":
        with open(get_data_search_dir(data_type, search_date, search_engine), "w") as f:
            json.dump(data, f, indent=4)
    else:
        raise Exception()
    