import PIL.ImageShow
from datasets import load_from_disk ,Dataset
from PIL import Image  
# dataset = load_from_disk("/home/yz979/ggf/ChatXRay/data")
from tqdm import tqdm  
import json
json_entries = []
import io
import pandas as pd
 
data_0 = pd.read_parquet('/data/ggf/ChatXRay/data/train-00000-of-00005-2e64b288713e3244.parquet')  
data_1 = pd.read_parquet('/data/ggf/ChatXRay/data/train-00001-of-00005-758d41c16caf8567.parquet')  
data_2 = pd.read_parquet('/data/ggf/ChatXRay/data/train-00002-of-00005-eadec50cc9adb94f.parquet')  
data_3 = pd.read_parquet('/data/ggf/ChatXRay/data/train-00003-of-00005-80986d4d125d721b.parquet')  
data_4 = pd.read_parquet('/data/ggf/ChatXRay/data/train-00004-of-00005-cda2efa058cd60f9.parquet')  

merged_data = pd.concat([data_0, data_1, data_2, data_3, data_4])  
dataset = Dataset.from_pandas(merged_data)  
from tqdm import tqdm  
  

for i in tqdm(range(len(dataset["image"]))):  
    pic = dataset[i]['image']
    #text = dataset["train"][i]['text']
    report = dataset[i]['report']
  
    # 创建一个包含id和对话历史的条目  
    entry = {  
        "image": [f"/data/ggf/ChatXRay/img/image_{i}.jpeg"],
        "ground_truth": report,
    }  
  
    # 将条目添加到JSON条目列表中  
    json_entries.append(entry)  
    if i == 100:
        break
    
json_data = json.dumps(json_entries, ensure_ascii=False, indent=4) 

with open('/home/ggf/code/ChatXRay/mimic_100.json', 'w', encoding='utf-8') as f:  
    f.write(json_data)  
