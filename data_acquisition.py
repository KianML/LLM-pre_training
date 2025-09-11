# It gets the datasets for an LLM training

from datasets import load_dataset

# PubMed. Web address: https://huggingface.co/datasets/devanshamin/PubMedDiabetes-LLM-Predictions
ds_pub = load_dataset("devanshamin/PubMedDiabetes-LLM-Predictions", split="train")
# Web Crawl. Web address: https://huggingface.co/datasets/kaifahmad/allenai-complex-web-QnA
ds_web = load_dataset("kaifahmad/allenai-complex-web-QnA", split="train")
# GitHub codes. Web address: https://huggingface.co/datasets/ed001/ds-coder-instruct-v1
ds_git = load_dataset("ed001/ds-coder-instruct-v1", split="train")
# Wikipedia. Web address: https://huggingface.co/datasets/agentlans/wikipedia-paragraph-sft
ds_wiki = load_dataset("agentlans/wikipedia-paragraph-sft", split="train")

# Converts datasets to json file and stores locally
local_path = ""
ds_pub.to_json(local_path + "PubMed.json")
ds_web.to_json(local_path + "WebCrawl.json")
ds_git.to_json(local_path + "GitHub.json")
ds_wiki.to_json(local_path + "Wikipedia.json")
