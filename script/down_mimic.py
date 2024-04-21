from datasets import load_dataset

import os
# 注意os.environ得在import huggingface库相关语句之前执行。
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download


def download_multi(local_dir,repo_id,token):
    print(f'开始下载\n仓库：{repo_id}\n如超时不用管，会自定继续下载，直至完成。中途中断，再次运行将继续下载。')
    while True:
        try:
            snapshot_download(
            local_dir=local_dir,
            repo_id=repo_id,
            token=token,
            repo_type="dataset",
            local_dir_use_symlinks=False,
            resume_download=True,
            # allow_patterns=["*.zst", "*.json", "*.bin",
            # "*.py", "*.md", "*.txt"],
            # ignore_patterns=["*.safetensors", "*.msgpack",
            # "*.h5", "*.ot",],
            )
        except Exception as e :
            print(e)
            # time.sleep(5)
        else:
            print('下载完成')
            break



download_multi("/data/ggf/ChatXRay","hongrui/mimic_chest_xray_v_1",'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')
# dataset = load_dataset("hongrui/mimic_chest_xray_v_1")
  
# save_path = "/data/ggf/ChatXRay"  
# dataset.save_to_disk(save_path)  