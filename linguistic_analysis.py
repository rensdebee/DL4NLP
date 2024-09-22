import re
import random
# get dataset
from datasets import load_dataset, concatenate_datasets, DatasetDict

ds_train = load_dataset("DanteZD/HC3_plus_llama70B", split="train")
ds_reddit_test = load_dataset("DanteZD/HC3_plus_llama70B", split="reddit_test")
ds_reddit = concatenate_datasets([ds_train, ds_reddit_test])

'''
Vocabulary Features on reddit only:
L: The average length
N: The number of unique words
The density: D = 100 x V/(L x N)
N = number of answers
'''

random.seed(42)

# average length of answers
def vocabulary_features(dataset, type="human_answers"):
    average_len = 0
    unique_words = set()
    for answer in dataset[type]:
        sample = random.choice(answer)
        no_punctuation = re.sub(r'[^\w\s]', '', sample)
        words = no_punctuation.split()
        unique_words = unique_words | set(words)
        average_len += len(words)
    num_unique = len(unique_words)
    average_len = average_len / len(dataset[type])
    density = 100 * num_unique / (average_len * len(dataset[type]))
    print(type)
    print("The average length of is: ", average_len)
    print("The number of unique words is: ", num_unique)
    print('The density is: ', density)

# print the vocabulary features for all types
# vocabulary_features(ds_reddit)
# vocabulary_features(ds_reddit, type="chatgpt_answers")
# vocabulary_features(ds_reddit, type="llama_answers")

# '''
# Part-of-Speech on reddit only:
# Proportion
# '''

# import nltk
# # nltk.download()
# from nltk.tokenize import word_tokenize

# def pos(dataset, type="human_answers"):
#     desired_tags = {'NOUN', 'PUNCT', 'VERB', 'ADP', 'DET', 'PRON', 'ADJ', 
#                 'AUX', 'ADV', 'CCONJ', 'PROPN', 'PART', 'SCONJ', 
#                 'NUM', 'SPACE', 'SYM', 'X', 'INTJ'}
#     for answer in dataset[type]:
#         sample = random.choice(answer)
#         tokens = word_tokenize(sample)
#         tagged_tokens = nltk.pos_tag(tokens)   
#         print(tagged_tokens) 
#         filtered_tokens = [(word, tag) for word, tag in tagged_tokens if tag in desired_tags]
   
#         print("Filtered Tokens:", filtered_tokens)
#         break

# pos(ds_reddit)

'''
Sentiment Analysis
https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment
'''

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
# pip install sentencepiece
# pip install protobuf

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)



def semantics(dataset, type="human_answers"):
    negative = 0
    positive = 0
    neutral = 0

    for answer in dataset[type]:
        sample = random.choice(answer)
        text = preprocess(sample)
        encoded_input = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        best_index = np.argmax(scores)

        best_label = config.id2label[best_index]

        if best_label == 'positive':
            positive += 1
        elif best_label == 'negative':
            negative += 1
        else:
            neutral += 1
    return neutral, positive, negative

import numpy as np
import matplotlib.pyplot as plt


def semantics_plot(human_data, chatgpt_data, llama_data):

    categories = ['neutral', 'positive', 'negative']
    x = np.arange(len(categories))
    bar_width = 0.25

    bar1 = plt.bar(x, human_data, width=bar_width, label='Human', color='mediumslateblue', edgecolor='black')
    bar2 = plt.bar(x + bar_width, chatgpt_data, width=bar_width, label='ChatGPT', color='gold', edgecolor='black')
    bar3 = plt.bar(x + 2 * bar_width, llama_data, width=bar_width, label='Llama', color='lightpink', edgecolor='black')

    plt.ylabel('Proportion (%)')
    plt.title('Sentiment Distribution')
    plt.xticks(x + bar_width, categories)

    plt.legend()

    def add_value_labels(bars):
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}', 
                    ha='center', va='bottom')

    add_value_labels(bar1)
    add_value_labels(bar2)
    add_value_labels(bar3)

    plt.show()

# run semantics:
# MODEL = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"

# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# config = AutoConfig.from_pretrained(MODEL)

# # PT
# model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# model.save_pretrained(MODEL)

# neutral, positive, negative = semantics(ds_reddit, type="human_answers")
# human_data = [neutral / ds_reddit * 100, positive / ds_reddit * 100]
# neutral, positive, negative = semantics(ds_reddit, type="chatgpt_answers")
# human_data = [neutral / ds_reddit * 100, positive / ds_reddit * 100]
# neutral, positive, negative = semantics(ds_reddit, type="llama_answers")
# human_data = [neutral / ds_reddit * 100, positive / ds_reddit * 100]
# semantics_plot(human_data, chatgpt_data, llama_data)


# run locally and stored in comments
# human_data = [1426 / 3000 * 100, 135 / 3000 * 100, 1439 / 3000 * 100]
# chatgpt_data = [1983 / 3000 * 100, 90 / 3000 * 100, 927 / 3000 * 100]
# llama_data = [1468 / 3000 * 100, 105 / 3000 * 100, 1427 / 3000 * 100]
# semantics_plot(human_data, chatgpt_data, llama_data)

