# LLM pre-training pipeline

This is a simple pre-training pipeline project designed for LLM models. You need to have python 3.1+

Execute data_acquisition.py first to download datasets from huggingface in json format. You need to install 'datasets' package and change 'local_path' variable to your own local path you wish to store datasets into.

Then execute data_prep.py for data preperation. It requires 'json', 're', 'unicodedata',  'matplotlib 3.10.6', 'random' and 'tensorflow 2.20.0' packages. Change the 'path' variable in line 49 to your own local path which the json files are stored.

It is recommended to set the variable 'n' in line 64 to a small number (e.g. 10), uncomment the block and test the execution of the code first. When runs smoothly comment the block again and run it for the complete dataset.

If you need to upscale the model and improve the execution time you can lower the number of most frequent words in lines 108 and 109. You can also raise the batch size in line 144.
