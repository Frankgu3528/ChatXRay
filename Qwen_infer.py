from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)
from peft import AutoPeftModelForCausalLM
import json
from modelscope import snapshot_download
# path_to_adapter = "/data/ggf/ChatXRay/output_qwen"
# 请注意：分词器默认行为已更改为默认关闭特殊token攻击防护。
model_dir = snapshot_download('qwen/Qwen-VL-Chat')
tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map = "cuda:5",trust_remote_code=True)

# 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# 使用CPU进行推理，需要约32GB内存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# 默认gpu进行推理，需要约24GB显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
model = AutoModelForCausalLM.from_pretrained(
    model_dir, # path to the output directory
    device_map="cuda:5",
    trust_remote_code=True
).eval()
# 可指定不同的生成长度、top_p等相关超参（transformers 4.32.0及以上无需执行此操作）
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)


file_path = '/home/ggf/code/ChatXRay/mimic_100.json'  
with open(file_path, 'r') as file:  
    data = json.load(file)  

for entry in data:
    query = tokenizer.from_list_format([
    {'image': entry["image"][0]}, # Either a local path or an url
    {'text': 'Write a report about this Xray picture.'},])   
    response, history = model.chat(tokenizer, query=query, history=None)
    print(response)
    entry["QWen-VL"] = response

with open(file_path, 'w') as file:  
    json.dump(data, file, indent=4)  