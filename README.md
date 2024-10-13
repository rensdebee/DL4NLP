# Exploring the Cross-Domain and Cross-Model Capabilities of Supervised Machine-Generated Text Detectors

Due to the rapid improvements in the quality of texts generated by Large Language Models, their usage has vastly increased. This has raised concerns regarding issues such as plagiarism, acadamic integrity and the spread of misinformation. Additionally, the combination of high-quality output and sheer volume has made manual detection impossible, prompting a shift towards machine-based detection systems.
This project investigates the generalizability of machine detectors across different models and domains. Specifically, it examines how well RoBERTa, when trained on one model or domain, can generalize to another with minimal fine-tuning on the new target. 

* It builds upon work by [Guo et al. 2023](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection?tab=readme-ov-file) by extending the experiments with different generator models beyond ChatGPT.
* We add to the [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) dataset, a human-ChatGPT comparison corpus, by generating additional responses with the Llama 3.1 70B and 8B Instruct models. The dataset used throughout this project can be found [here](https://huggingface.co/datasets/DanteZD/HC3_plus_llama70B).

## Adding to the dataset
Additional samples for text generated by the Llama-70B model can be obtained as follows:
```bash
python ./dataset/create_dataset.py --data_split reddit_eli5 --start_idx 6000 --stop_idx 7000
```
Which would generated responses to Reddit ELI5 question prompts for entry 6000 to 7000, and upload them to a huggingface dataset.

A similar command can be run for Llama-8B (which requires a different setup for high quality responses):
```bash
python ./dataset/create_8B_dataset.py --data_split reddit_eli5 --start_idx 6000 --stop_idx 7000
```

## Performing the linguistic analysis
Linguistic analysis can be run with the linguistic_analysis.py file.
The instruction are given inside the file with examples of the types and names to run.
The three functions that can be run are: get_vocabulary_features, get_semantics_plot and get_perplexity_plot

The topic extraction can be performed by simply running:
```bash
python topic_extraction.py
```
This will create the embedding distribution plot, and topics plot. Note that you need a folder named: 'topic_ex'

## Experiments
The experiments can be reproduced by running the following commands:

### Cross-Domain and Cross-Model Evaluation
All commands can be found in the 'commands.txt' file under TABLE 2, When all of these are trained, you can go to Evaluation.ipynb to evaluate and print all found results, ensuring that the model paths are set correctly.<br>
EXAMPLE:
```bash
python train.py -traind reddit_train -traing chatgpt -testd reddit_test -testg chatgpt -b 4
```

### In-depth Cross-Domain Evaluation
All commands can be found in the 'commands.txt' file under TABLE 3, When all of these are trained, you can go to Evaluation.ipynb to evaluate and print all found results, ensuring that the model paths are set correctly.<br>
EXAMPLE:
```bash
python train.py -traind reddit_train -traing llama -testd reddit_test -testg llama -b 4 --train_multiple wiki_csai,open_qa,finance,medicine
```

### Data Ablation Study
Use data_ablation.ipynb for generating the results
```bash
# REDDIT BASE MODEL TRAINING:
python train.py -traind reddit_train -traing llama -testd reddit_test -testg llama -b 4 --eval_steps 1000
# FINE-TUNING:
# NON REDDIT:
python train.py -m models\train_conf_reddit_train_llama\checkpoint-2340 -traind non_reddit_test -traing llama -testd non_reddit_test,reddit_test -testg llama -b 4 --eval_steps 1 --split_ratio 0.8
# WIKI CS/AI
python train.py -m models\train_conf_reddit_train_llama\checkpoint-2340 -traind wiki_csai -traing llama -testd medicine,reddit_test,wiki_csai -testg llama -b 4 --eval_steps 1 --split_ratio 0.9


```

### Authors: *H.C. van den Bos, R. den Braber, A.J. van Breda, D. Zegveld*