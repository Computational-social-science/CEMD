from run_base import SFTModels, DatasetName, run_lora, run_vera, run_dora, run_pissa, run_lora_plus, run_rslora, run_dora_with_info_or_not

DEVICE = "0"

lr = '1e-4'
run_dora(SFTModels.llama_3_8b_instruct, lr, DatasetName.covmis_wsc2, DEVICE, data_version="1")
