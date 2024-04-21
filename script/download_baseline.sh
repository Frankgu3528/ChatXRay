export HF_ENDPOINT=https://hf-mirror.com

# Fuyu-8b
huggingface-cli download --resume-download adept/fuyu-8b --local-dir /data/ggf/ChatXRay/baselines/fuyu-8b --local-dir-use-symlinks False

# blip2-opt-2.7b
huggingface-cli download --resume-download Salesforce/blip2-opt-2.7b --local-dir /data/ggf/ChatXRay/baselines/blip2opt --local-dir-use-symlinks False

# Yi-VL-6B
huggingface-cli download --resume-download 01-ai/Yi-VL-6B --local-dir /data/ggf/ChatXRay/baselines/Yi6b --local-dir-use-symlinks False