from datasets import load_from_disk  
  
dataset = load_from_disk("/home/yz979/ggf/ChatXRay/data")

pic = dataset["train"][0]['text']
print(pic)
