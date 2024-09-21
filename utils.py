import numpy as np
import evaluate
from scipy.special import softmax
import random
import torch
import pandas as pd
from datasets import Dataset

def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def convert_dataset(dataset, bot):
    ds = []
    for row in dataset:
        q = row["question"]
        human = row["human_answers"]
        model = row[f"{bot}_answers"]
        for h in human:
            ds.append([q, h, 0])
        for m in model:
            ds.append([q, m, 1])

    ds = Dataset.from_pandas(pd.DataFrame(ds, columns=["question", "answer", "label"]))

    return ds

def compute_metrics(eval_preds):
    metrics = {}
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    probs = softmax(logits, axis=-1)

    roc_auc = evaluate.load("roc_auc")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")
    accuracy = evaluate.load("accuracy")

    metrics["accuracy"] = accuracy.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    metrics["precision"] = precision.compute(
        predictions=predictions, references=labels
    )["precision"]
    metrics["recall"] = recall.compute(predictions=predictions, references=labels)[
        "recall"
    ]
    metrics["f1"] = f1.compute(predictions=predictions, references=labels)["f1"]
    metrics["roc_auc"] = roc_auc.compute(
        prediction_scores=probs[np.arange(len(labels)), labels], references=labels
    )["roc_auc"]
    return metrics