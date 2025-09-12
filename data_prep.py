import json
import re
import unicodedata
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

# ======= Normalize and aggregate datasets =======================================================
# Normalize a line of text
def normalize(line):
    # Uses a unicode normalization form. See https://unicode.org/reports/tr15/
    try:
        line = unicodedata.normalize("NFKC", line.strip().lower())
        # searches for a non-alphanumeric, non-space character at the beginning of the line string,
        # but only if that character is not immediately followed by a whitespace character.
        # If such a match is found, it inserts a space immediately after that character.
        line = re.sub(r"^([^ \w])(?!\s)", r"\1 ", line)
        line = re.sub(r"(\s[^ \w])(?!\s)", r"\1 ", line)
        # searches for a single non-whitespace, non-word character (e.g., punctuation) that appears at the very end of the line string,
        # and is not preceded by a space. If such a character is found, it inserts a space before that character.
        line = re.sub(r"(?!\s)([^ \w])$", r" \1", line)
        # inserts a space before any punctuation mark or symbol that is immediately followed by a space,
        # unless that punctuation mark or symbol is already preceded by a space.
        line = re.sub(r"(?!\s)([^ \w]\s)", r" \1", line)
    except:
        print('Warning: not able to normalize this line: <', line, '>!')
    return line

# Takes a list of raw dictionaries and returns a dictionary list with normalized and tokenized 'prompt' and 'response' keys
def prmpt_rspns_split(raw_dic_lst, prmpt_key, rspns_key, is_code):
    prmpt_rspns = list()
    for elmnt in raw_dic_lst:
        # Checks for None type objects
        if elmnt[prmpt_key] is None: elmnt[prmpt_key] = ''
        if elmnt[rspns_key] is None: elmnt[rspns_key] = ''
        prmpt = normalize(elmnt[prmpt_key])
        # If the text is code then ignors normalization
        if not is_code:
            rspns = normalize(elmnt[rspns_key])
        else:
            rspns = elmnt[rspns_key]
        rspns = "[start] " + rspns + " [end]"
        prmpt_rspns.append((prmpt, rspns))
    return prmpt_rspns

# The local path of stored raw datasets
# Change it into your local path
path = "/home/runner/work/LLM-pre_training/LLM-pre_training/Dataset-Samples/"

# Loads the json files
raw_pub = [json.loads(line) for line in open(path + 'PubMed.json', 'r')]
raw_web = [json.loads(line) for line in open(path + 'WebCrawl.json', 'r')]
raw_git = [json.loads(line) for line in open(path + 'GitHub.json', 'r')]
raw_wiki = [json.loads(line) for line in open(path + 'Wikipedia.json', 'r')]

# Adds the right answer (among many) element as a key for web crawl dataset
for elmnt in raw_web:
    elmnt['rght_answer'] = elmnt['answers'][0]['answer']

# Takes the first n elements of each dataset
# This is for testing the code. Ignore it during implementation
"""
n = 50000
raw_pub = raw_pub[:n]
raw_web = raw_web[:n]
raw_git = raw_git[:n]
raw_wiki = raw_wiki[:n]
"""

# Calls prompt/response split function by passing in the right key names for each dataset.
prmpt_rspns_pub = prmpt_rspns_split(raw_pub, 'title', 'explanation', 0)
prmpt_rspns_web = prmpt_rspns_split(raw_web, 'question', 'rght_answer', 0)
prmpt_rspns_git = prmpt_rspns_split(raw_git, 'instruction', 'output', 1)
prmpt_rspns_wiki = prmpt_rspns_split(raw_wiki, 'input', 'output', 0)

# Aggregates all four text sources and makes one dataset
prmpt_rspns = prmpt_rspns_pub + prmpt_rspns_web + prmpt_rspns_git + prmpt_rspns_wiki

#===== Plot =====================================================================
# Takes the length of sentences in prompt and response parts
prmpt_len = [len(prmpt.split()) for prmpt, rspns in prmpt_rspns]
rspns_len = [len(rspns.split()) for prmpt, rspns in prmpt_rspns]

# The histogram illustrates the distribution sentences length 
plt.hist(prmpt_len, label="prompt", color="g", alpha=0.33)
plt.hist(rspns_len, label="response", color="c", alpha=0.33)
plt.yscale("log")     # logarithm vertical axis for better view
plt.ylim(plt.ylim())  # y-axis adjust for both plots
plt.plot([max(prmpt_len), max(prmpt_len)], plt.ylim(), color="g")
plt.plot([max(rspns_len), max(rspns_len)], plt.ylim(), color="c")
plt.legend()
plt.title("Count vs sentence length")
plt.show()

# ==== Train, Test and validation split ==============================================
# train, test, validation sets of shuffled sentence pairs
random.shuffle(prmpt_rspns)
val_size = int(0.15 * len(prmpt_rspns))
train_size = len(prmpt_rspns) - 2 * val_size
train_set = prmpt_rspns[:train_size]
val_set = prmpt_rspns[train_size : train_size + val_size]
test_set = prmpt_rspns[train_size + val_size:]

# ==== Model Scalability ===========================================================
# Change vocabulary size to adjust the scale of the model
prmpt_vocab_size = 30000
rspns_vocab_size = 100000
# the maximum sentence lenght in the dataset
sentence_len = 50

# ==== Vectorizing ==================================================================
# Create vectorizer
prmpt_vec = TextVectorization(
    max_tokens=prmpt_vocab_size,
    standardize=None,
    split="whitespace",
    output_mode="tf_idf",
    # output_mode="int", # use this when limitting the sentence_len
    # output_sequence_length = sentence_len # use this when limitting the sentence_len
)
rspns_vec = TextVectorization(
    max_tokens=rspns_vocab_size,
    standardize=None,
    split="whitespace",
    output_mode="tf_idf",
    # output_mode="int", # use this when limitting the sentence_len
    # output_sequence_length = sentence_len + 1 # use this when limitting the sentence_len
)

# training the vectorizer
prmpt_vec.adapt([item[0] for item in train_set])
rspns_vec.adapt([item[1] for item in train_set])

# Making dataset ready for training LLM
# This function Takes an prompt and response sentence pair and convert them into source and target.
# The input is a dictionary with keys `enc_in` and `dec_in`, each is a vector, corresponding to prompt and response respectively.
def shape_ds(prmpt, rspns):
    prmpt = prmpt_vec(prmpt)
    rspns = rspns_vec(rspns)
    source = {"enc_in": prmpt, "dec_in": rspns[:, :-1]}
    target = rspns[:, 1:]
    return (source, target)

# Creates TensorFlow Dataset
def make_ds(pairs, batch_size = 64):
    # aggregate sentences
    prmpt_text, rspns_text = zip(*pairs)
    # convert into list and create tensors
    dataset = tf.data.Dataset.from_tensor_slices((list(prmpt_text), list(rspns_text)))
    return dataset.shuffle(2048).batch(batch_size).map(shape_ds).prefetch(16).cache()

# Makes training and validation datasets
train_dataset = make_ds(train_set)
val_dataset = make_ds(val_set)

# test the shape of dataset
for src, trgt in train_dataset.take(1):
    print(f'inputs["enc_in"].shape: {src["enc_in"].shape}')
    print(f'inputs["enc_in"][0]: {src["enc_in"][0]}')
    print(f'inputs["dec_in"].shape: {src["dec_in"].shape}')
    print(f'inputs["dec_in"][0]: {src["dec_in"][0]}')
    print(f"targets.shape: {trgt.shape}")
    print(f"targets[0]: {trgt[0]}")








