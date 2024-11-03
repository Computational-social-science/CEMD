import subprocess

train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
test_sizes = [0.5, 0.4, 0.3, 0.2, 0.1]

data_version = "1"

class SFTModels:
	llama_3_8b_instruct = "Meta-Llama-3-8B-Instruct"
class DatasetName:
	covmis = "covmis"
	liar2 = "liar2"


def run_lora(sft_model, lr, dataset_name, device, rank="8", rag_model="llama3", data_version=data_version, num_epochs="1"):
	subprocess.run(
		["bash", f"my_tuning/{sft_model}/lora/lora.sh", dataset_name,
			"0.2", "1.0", "lora", rank, lr, f"with_{rag_model}_info/brave", data_version, num_epochs, device])

def run_pissa(sft_model, lr, device, rank="8", rag_model="llama3", data_version=data_version):
	subprocess.run(
		["bash", f"my_tuning/{sft_model}/lora/pissa.sh", 
   			"0.2", "1.0", "lora", rank, lr, f"with_{rag_model}_info/brave", data_version, device])

def run_vera(sft_model, lr, device, vera_rank="256", rag_model="llama3", data_version=data_version):
	subprocess.run(
		["bash", f"my_tuning/{sft_model}/lora/vera.sh", 
   			"0.2", "1.0", "vera", vera_rank, lr, f"with_{rag_model}_info/brave", data_version, device])

def run_rslora(sft_model, lr, device, rank="8", rag_model="llama3", data_version=data_version):
	subprocess.run(
		["bash", f"my_tuning/{sft_model}/lora/rslora.sh", 
   			"0.2", "1.0", "lora", rank, lr, f"with_{rag_model}_info/brave", data_version, device])

def run_lora_plus(sft_model, lr, device, rank="8", rag_model="llama3", data_version=data_version):
	subprocess.run(
		["bash", f"my_tuning/{sft_model}/lora/lora_plus.sh",
			"0.2", "1.0", "lora", rank, lr, f"with_{rag_model}_info/brave", data_version, device])

def run_dora(sft_model, lr, dataset_name, device, rank="8", rag_model="llama3", data_version=data_version, num_epochs="1"):
	subprocess.run(
		["bash", f"my_tuning/{sft_model}/lora/dora.sh", dataset_name,
   			"0.2", "1.0", "lora", rank, lr, f"with_{rag_model}_info/brave", data_version, num_epochs, device])

def run_dora_with_info_or_not(sft_model, lr, dataset_name, device, with_info, rank="8", data_version=data_version, num_epochs="1"):
	if with_info:
		with_info_or_not = "with_info"
	else:
		with_info_or_not = "without_info"
	subprocess.run(
		["bash", f"my_tuning/{sft_model}/lora/dora.sh", dataset_name,
   			"0.2", "1.0", "lora", rank, lr, with_info_or_not, data_version, num_epochs, device])
