# imports
import random
from linguistic_utils import get_dataset, vocabulary_features, semantics, semantics_plot, perplexity, plot_ppl

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# pip install sentencepiece #install if needed
# pip install protobuf #install if needed

# set seed
random.seed(42)

def get_vocabulary_features(names, types):
    '''
    Vocabulary Features:
    L: The average length
    N: The number of unique words
    The density: D = 100 x V/(L x N)
    N = number of answers
    Names: list of names for dataset splits
    Types: list of column names
    '''
    dataset = get_dataset(names)
    for type in types:
        average_len, num_unique, density = vocabulary_features(dataset, type)
        print(type)
        print("The average length of is: ", average_len)
        print("The number of unique words is: ", num_unique)
        print('The density is: ', density)



def get_semantics_plot(names, types, labels, colours):
    '''
    Sentiment Analysis: neural, positive, neutral
    https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment
    Names: list of names for dataset splits
    Types: list of column names
    Labels: list of label names for the plot
    Colours: list of colours for the plot
    '''
    MODEL = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.save_pretrained(MODEL)

    dataset = get_dataset(names)
    datas = []
    for type in types:
        neutral, positive, negative = semantics(model, tokenizer, config, dataset, type)
        datas.append([neutral / len(dataset) * 100, positive / len(dataset) * 100, negative / len(dataset) * 100])
    semantics_plot(datas, labels, colours)



def get_perplexity_plot(names, types, labels, colours):
    '''
    Perplexity
    https://huggingface.co/openai-community/gpt2
    Names: list of names for dataset splits
    Types: list of column names
    Labels: list of label names for the plot
    Colours: list of colours for the plot
    '''
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()

    dataset = get_dataset(names)
    datas = []
    for type in types:
        ppl = perplexity(model, tokenizer, dataset, type)
        datas.append(ppl)
    plot_ppl(datas, labels, colours)

# names: reddit_test, train
# types: human_answers, chatgpt_answers, llama_answers
# get_vocabulary_features(["reddit_test", "train"], ['human_answers', 'chatgpt_answers', 'llama_answers'])
# get_semantics_plot(["reddit_test", "train"], ['human_answers', 'chatgpt_answers', 'llama_answers'], 
#                    ['Human', 'ChatGPT', 'Llama'], ['mediumslateblue', 'gold', 'lightpink'])
# get_perplexity_plot(["reddit_test", "train"], ['human_answers', 'chatgpt_answers', 'llama_answers'], 
#                    ['Human', 'ChatGPT', 'Llama'], ['mediumslateblue', 'gold', 'lightpink'])
