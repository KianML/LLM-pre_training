# LLM pre-training pipeline

This is a simple pre-training pipeline project designed for LLM models. You need to have python 3.1+

Execute data_acquisition.py first to download datasets from huggingface in json format. You need to install 'datasets' package and change 'local_path' variable to your own local path you wish to store datasets into.

Then execute data_prep.py for data preperation. It requires 'json', 're', 'unicodedata',  'matplotlib 3.10.6', 'random' and 'tensorflow 2.20.0' packages. Change the 'path' variable in line 49 to your own local path which the json files are stored.

It is recommended to set the variable 'n' in line 64 to a small number (e.g. 10), uncomment the block and test the execution of the code first. When runs smoothly comment the block again and run it for the complete dataset.

The output_mode in text vectorization is set to 'tf-idf' (lines 118 and 126) till adapt() (lines 132 and 133) learn the document frequencies of each token in the input dataset. When scaling up the model the vocabulary size can be reduced. The sentence length also can be limited to a few words (e.g. 50) but in this case the output_mode needs to be set to 'int'. Uncomment lines 119, 120, 127 and 128.

For larger datasets you may increase the batch size (line 146) and the size of the text bundle in shuffle (line 151).
