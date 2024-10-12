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
TODO Weet niet of je hier nog ff wil laten weten hoe ze het moeten runnen?

## Experiments
The experiments can be reproduced by running the following commands:

### Cross-Domain and Cross-Model Evaluation
```bash
python RUN RELEVANT SCRIPT + PARAMETERS 
```

### In-depth Cross-Domain Evaluation
```bash
python RUN RELEVANT SCRIPT + PARAMETERS 
```

### Data Ablation Study
```bash
python RUN RELEVANT SCRIPT + PARAMETERS 
```

### Authors: *H.C. van den Bos, R. den Braber, A.J. van Breda, D. Zegveld*










