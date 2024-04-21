<img src="./img/logo.png" style="zoom: 25%;" />

<u>ChatXRay</u> is a Multimodel chatbot that input chest radiographs and generate medical report. You can find my paper [here](./final.pdf).

<img src="./img/demo.jpg" alt="img" style="zoom: 6%;" />

## Quick-Start

Of  course, you need to modify all the path in the repo to suit your own datapath.
### Set up
```
pip install -r requirements.txt
```
### generate finetuning dataset

```
./script/down_mimic.py
python generate_QWen_json.py
```

### train
```
python finetune.py
```

### inference
```
python infer.py         # use ChatXRay to infer
python Qwen_infer.py    # use Qwen-VL to infer
python ./baselines/Fuyu-8b/infer.py # use Fuyu-8b to infer
```

### generate inference result
```
python test_baseline.py
```