from datasets import load_dataset

dataset = load_dataset("hongrui/mimic_chest_xray_v_1")
  
save_path = "/home/yz979/ggf/ChatXRay/data"  
dataset.save_to_disk(save_path)  