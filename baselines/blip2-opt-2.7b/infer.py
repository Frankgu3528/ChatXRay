# pip install accelerate
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")

processor = Blip2Processor.from_pretrained("/data/ggf/ChatXRay/baselines/blip2opt", local_files_only=True)  
model = Blip2ForConditionalGeneration.from_pretrained("/data/ggf/ChatXRay/baselines/blip2opt", device_map = 'cuda', local_files_only=True)  

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

question = "how many dogs are in the picture?"
inputs = processor(raw_image, question, return_tensors="pt").to("cuda")

generation_output = model.generate(**inputs, max_new_tokens=30)
generation_text = processor.batch_decode(generation_output[:, -30:], skip_special_tokens=True)
print(generation_text)