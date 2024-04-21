import json 
from openai import OpenAI
import httpx

client = OpenAI(
    base_url="https://api.xty.app/v1", 
    api_key="YOUR_OPENAI_API_KEY",
    http_client=httpx.Client(
        base_url="https://api.xty.app/v1",
        follow_redirects=True,
    ),
)

with open('/home/ggf/code/ChatXRay/mimic_100.json') as f: # put the json_file contain two models output and ground truth here
    data = json.load(f)

k = 0
m = 0
for entry in data:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "I will give you three reports. The first is the true one, the next two are made by two models. Please tell me which one can point out the symptom,(chatxray or Qwen-VL) only give me the result in its name and don;texplain."},
            {"role": "user", "content": "Truth: "+entry["ground_truth"]},
            {"role": "user", "content": "Qwen-VL: "+entry["QWen-VL"]},
            {"role": "user", "content": "chatxray: "+entry["ChatXRay"]}
        ]
        )
    print(completion.choices[0].message.content)
    if 'Qwen-VL' in completion.choices[0].message.content:
        entry["result"] = 0
        print(0)
    elif 'chatxray' or 'ChatXray' in completion.choices[0].message.content:
        entry["result"] = 1
        print(1)
        m+=1
    k+=1
    
    
print(m)

