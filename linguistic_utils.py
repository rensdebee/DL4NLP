# imports
from datasets import load_dataset, concatenate_datasets, DatasetDict
import random
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax


def get_dataset(names):
    '''
    Retrieve and combine n datasets
    Names: list of names of dataset splits
    '''
    dataset = load_dataset("DanteZD/HC3_plus_llama70B", split=names[0])
    if len(names) == 1:
        return dataset
    else:
        for name in names[1:]:
            ds = load_dataset("DanteZD/HC3_plus_llama70B", split=name)
            dataset = concatenate_datasets([dataset, ds])
        return dataset
    


def vocabulary_features(dataset, type="human_answers"):
    '''
    Compute vocabulary features
    Dataset: dataset
    Type: column name of dataset
    '''
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
    return average_len, num_unique, density



def semantics(model, tokenizer, config, dataset, type="human_answers"):
    '''
    Compute sentiment of dataset
    Model: to use for sentiment prediction
    Tokenizer: tokenizer that comes with model
    Config: config that comes with model
    Dataset: dataset
    Type: column name of dataset
    '''
    negative = 0
    positive = 0
    neutral = 0

    for answer in dataset[type]:
        sample = random.choice(answer)
        encoded_input = tokenizer(sample, return_tensors='pt', max_length=512, truncation=True)
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



def add_value_labels(bars):
    '''
    Add values on top of bar in barplot
    '''
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}', 
                ha='center', va='bottom')



def semantics_plot(datas, labels, colours):
    '''
    Plot the sementiment data
    Datas: list with lists of neutral, positive, negative proportions
    Labels: list of names corresponding with the dataset
    Colours: list of desired colours
    '''

    categories = ['neutral', 'positive', 'negative']
    x = np.arange(len(datas[0]))
    n_bars = len(categories)
    bar_width = 0.8 / n_bars

    for i, data in enumerate(datas):
        bar = plt.bar(x + i * bar_width, data, width=bar_width, label=labels[i], color=colours[i], edgecolor='black')
        add_value_labels(bar)

    plt.ylabel('Proportion (%)')
    plt.title('Sentiment Distribution')
    plt.xticks(x + bar_width, categories)

    plt.legend()
    plt.show()



def calculate_perplexity(text, model, tokenizer):
    '''
    Compute perplexity of piece of text
    Text: piece of text to compute complexity from
    Model: model to use
    Tokenizer: tokenizer that comes with model
    '''
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        try:
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    return perplexity.item()



def perplexity(model, tokenizer, dataset, type="human_answers"):
    '''
    Get perplexity of column in dataset
    Dataset: dataset to compute perplexity of
    Type: column in dataset
    Model: model to use
    Tokenizer: tokenizer that comes with model
    '''
    perplexity = []
    for answer in dataset[type][:5]:
        sample = random.choice(answer)
        ppl = calculate_perplexity(sample, model, tokenizer)
        if ppl is not None:
            perplexity.append(ppl)
    return np.array(perplexity)



def plot_ppl(datas, labels, colours):
    '''
    Plot perplexities
    Datas: list of arrays with ppl values
    Labels: list of names corresponding with the dataset
    Colours: list of desired colours
    '''

    plt.figure(figsize=(6, 4))

    for i, data in enumerate(datas):
        sns.kdeplot(data, label=labels[i], color=colours[i], fill=True)

    plt.xlabel("Perplexity")
    plt.ylabel("Proportion")
    plt.title("Perplexity Distributions: ChatGPT, Human, and llama")
    plt.legend()

    plt.tight_layout()
    plt.show()




    # '''
# Part-of-Speech on reddit only:
# Proportion
# '''

# import nltk
# # nltk.download()
# from nltk.tokenize import word_tokenize

# def pos(dataset, type="human_answers"):
#     tag_dict = {'NOUN': 0, 'PUNCT':0, 'VERB':0, 'ADP':0, 'DET':0, 'PRON':0, 'ADJ':0, 
#                 'AUX':0, 'ADV':0, 'CCONJ':0, 'PROPN':0, 'PART':0, 'SCONJ':0, 
#                 'NUM':0, 'SYM':0, 'X':0, 'INTJ':0}
#     for answer in dataset[type]:
#         sample = random.choice(answer)
#         tokens = word_tokenize(sample)
#         tagged_tokens = nltk.pos_tag(tokens)  
#         for _, token in  tagged_tokens:
#             if token in {'NN', 'NNS', 'NNP', 'NNPS'}:
#                 tag_dict['NOUN'] += 1
#             elif token in {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}:
#                 tag_dict['VERB'] += 1
#             elif token == 'IN':
#                 tag_dict['ADP'] += 1
#             elif token in {'DT', 'PDT', 'WDT'}:
#                 tag_dict['DET'] += 1
#             elif token in {'PRP', 'PRPS', 'WP', 'WPS'}:
#                 tag_dict['PRON'] += 1
#             elif token in {'JJ', 'JJR', 'JJS'}:
#                 tag_dict['ADJ'] += 1
#             elif token == 'MD':
#                 tag_dict['AUX'] += 1
#             elif token in {'RB', 'RBR', 'RBS', 'WRB'}:
#                 tag_dict['ADV'] += 1
#             elif token == 'CC':
#                 tag_dict['CCONJ'] += 1
#             elif token in {'NNP', 'NNPS'}:
#                 tag_dict['PROPN'] += 1
#             elif token in {'RP', 'TO'}:
#                 tag_dict['PART'] += 1
#             elif token == 'IN':
#                 tag_dict['SCONJ'] += 1
#             elif token in {'CD', 'LS'}:
#                 tag_dict['NUM'] += 1 
#             elif token == 'SYM':
#                 tag_dict['SYM'] += 1
#             elif token == 'UH':
#                 tag_dict['INTJ'] += 1
#             elif token in {'.', ',', ':', ';', '?', '!', '-', '_', '(', ')', '[', ']', '{', '}', '"', "'", '\\', '/', '|', '@', '#', '$', '%', '^', '&', '*', '=', '+', '<', '>', '~', '`'}:
#                 tag_dict['PUNCT'] += 1
#             else:
#                 tag_dict['X'] += 1   
#     return tag_dict        
        

# tag_dict_human = pos(ds_reddit)
# tag_dict_chatgpt = pos(ds_reddit, type='chatgpt_answers')
# tag_dict_llama = pos(ds_reddit, type='llama_answers')
# print('human: ', tag_dict_human)
# print('chatgpt: ', tag_dict_chatgpt)
# print('llama: ', tag_dict_llama)

# human = {'NOUN': 94642, 'PUNCT': 47957, 'VERB': 67869, 'ADP': 42012, 'DET': 41514, 'PRON': 22750, 'ADJ': 30076, 'AUX': 6447, 'ADV': 26754, 'CCONJ': 12827, 'PROPN': 0, 'PART': 12165, 'SCONJ': 0, 'NUM': 5806, 'SYM': 96, 'X': 10897, 'INTJ': 158}
# chatgpt = {'NOUN': 141903, 'PUNCT': 57365, 'VERB': 105086, 'ADP': 66478, 'DET': 68057, 'PRON': 25760, 'ADJ': 49239, 'AUX': 13113, 'ADV': 29532, 'CCONJ': 26209, 'PROPN': 0, 'PART': 22889, 'SCONJ': 0, 'NUM': 4391, 'SYM': 1, 'X': 13918, 'INTJ': 151}
# llama = {'NOUN': 108862, 'PUNCT': 63442, 'VERB': 97146, 'ADP': 47615, 'DET': 52185, 'PRON': 39319, 'ADJ': 39402, 'AUX': 8404, 'ADV': 34268, 'CCONJ': 21050, 'PROPN': 0, 'PART': 16137, 'SCONJ': 0, 'NUM': 3846, 'SYM': 5, 'X': 15209, 'INTJ': 340}
# def plot_pos(human_dict, chatgpt_dict, llama_dict):

#     def normalize_dict(dict):
#         total = sum(dict.values())
#         return {key: value / total * 100 for key, value in dict.items()}
    
#     # normalise the dictionaries
#     human_dict = normalize_dict(human_dict)
#     chatgpt_dict = normalize_dict(chatgpt_dict)
#     llama_dict = normalize_dict(llama_dict)

#     # sort the dictionaries
#     human = dict(sorted(human_dict.items(), key=lambda x: x[1], reverse=True))
#     chatgpt =  {key: chatgpt_dict[key] for key in human}
#     llama = {key: llama_dict[key] for key in human}

#     categories = human.keys()
#     x = np.arange(len(categories))
#     bar_width = 0.25

#     plt.bar(x, human.values(), width=bar_width, label='Human', color='mediumslateblue', edgecolor='black')
#     plt.bar(x + bar_width, chatgpt.values(), width=bar_width, label='ChatGPT', color='gold', edgecolor='black')
#     plt.bar(x + 2 * bar_width, llama.values(), width=bar_width, label='Llama', color='lightpink', edgecolor='black')

#     plt.ylabel('Proportion (%)')
#     plt.title('Part-of-Speech Comparison')
#     plt.xticks(x + bar_width, categories)

#     plt.legend()

#     plt.show()

# def pos_results():
#     tag_dict_human = pos(ds_reddit)
#     tag_dict_chatgpt = pos(ds_reddit, type='chatgpt_answers')
#     tag_dict_llama = pos(ds_reddit, type='llama_answers')

#     plot_pos(tag_dict_human, tag_dict_chatgpt, tag_dict_llama)

# pos_results()