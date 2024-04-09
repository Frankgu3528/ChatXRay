import PIL.ImageShow
from datasets import load_from_disk  
from PIL import Image  
dataset = load_from_disk("/home/yz979/ggf/ChatXRay/data")
from tqdm import tqdm  
import json
json_entries = []

for i in tqdm(range(len(dataset["train"]))):  
    pic = dataset["train"][i]['image']
    #text = dataset["train"][i]['text']
    report = dataset["train"][i]['report']
    # pic.save(f"/home/yz979/ggf/ChatXRay/data/pic/image_{i}.jpeg")  
    conversation = [  
        {  
            "from": "user",  
            "value": f"Picture 1: <img>/home/yz979/ggf/ChatXRay/data/pic/image_{i}.jpeg</img>\n write a report based on the X-Ray image"  
        },  
        {  
            "from": "assistant",  
            "value": report
        }
        # {  
        #     "from": "user",  
        #     "value": "Give a short report about the image."
        # } ,  
        # {  
        #     "from": "assistant",  
        #     "value": report 
        # }
    ]  
  
    # 创建一个包含id和对话历史的条目  
    entry = {  
        "id": f"identity_{i}",  
        "conversations": conversation  
    }  
  
    # 将条目添加到JSON条目列表中  
    json_entries.append(entry)  
    
json_data = json.dumps(json_entries, ensure_ascii=False, indent=4) 

with open('/home/yz979/ggf/ChatXRay/data/mimic.json', 'w', encoding='utf-8') as f:  
    f.write(json_data)  
  
print("JSON saved at /home/yz979/ggf/ChatXRay/data/mimic.json") 