import PIL.ImageShow
from datasets import load_from_disk  
from PIL import Image  
dataset = load_from_disk("/home/yz979/ggf/ChatXRay/data")
from tqdm import tqdm  
  
for i in tqdm(range(len(dataset["train"]))):  
    pic = dataset["train"][i]['image']
    # text = dataset["train"][i]['text']
    # report = dataset["train"][i]['report']
    pic.save(f"/home/yz979/ggf/ChatXRay/data/pic/image_{i}.jpeg")  
