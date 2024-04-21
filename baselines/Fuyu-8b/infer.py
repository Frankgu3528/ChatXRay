from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
import requests
import json
# load model and processor
model_id = "adept/fuyu-8b"
processor = FuyuProcessor.from_pretrained("/data/ggf/ChatXRay/baselines/fuyu-8b", local_files_only=True)
model = FuyuForCausalLM.from_pretrained("/data/ggf/ChatXRay/baselines/fuyu-8b", device_map="cuda:5",local_files_only=True )

# prepare inputs for the model
text_prompt = "write a medical report about the image.\n"

  
file_path = '/home/ggf/code/ChatXRay/mimic_100.json'  
with open(file_path, 'r') as file:  
    data = json.load(file)  

for entry in data:   
    image = Image.open(entry["image"][0]).convert("RGB")
    inputs = processor(text=text_prompt, images=image, return_tensors="pt").to("cuda:5")
    generation_output = model.generate(**inputs, max_new_tokens=100)
    generation_text = processor.batch_decode(generation_output[:, -120:], skip_special_tokens=True)
    print("=============")
    print(generation_text[0])
    desired_output = generation_text[0].split("\x04")[1]  
    entry["fuyu-8b"] = desired_output 

with open(file_path, 'w') as file:  
    json.dump(data, file, indent=4)  