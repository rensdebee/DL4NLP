import numpy as np
from collections import Counter
import evaluate
from scipy.special import softmax
import random
import torch
import pandas as pd
from datasets import Dataset, load_dataset


def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_datasets(
    train_domain=None,
    train_generator=None,
    test_domain=None,
    test_generator=None,
    ratio=0.2,
    seed=None,
):
    dataset_url = "DanteZD/HC3_plus_llama70B"
    # Filter dataset based on the source
    if train_domain and train_generator:
        ds = load_dataset(dataset_url, split=train_domain)
        ds = convert_dataset(ds, train_generator)
        ds = ds.class_encode_column("label")
        ds = ds.train_test_split(test_size=ratio, seed=seed, stratify_by_column="label")
        print("Train set:")
        ds_train = ds["train"]
        print(Counter(ds_train["label"]))
        ds_val = ds["test"]
        print("Eval set")
        print(Counter(ds_val["label"]))
    else:
        ds_train = None
        ds_val = None

    if test_domain and test_generator:
        ds_test = load_dataset(dataset_url, split=test_domain)
        ds_test = convert_dataset(ds_test, test_generator)
        print("Test set:")
        print(Counter(ds_test["label"]))
    else:
        ds_test = None

    return ds_train, ds_val, ds_test


def convert_dataset(dataset, generator):
    ds = []
    for row in dataset:
        q = row["question"]
        human = row["human_answers"]
        model = row[f"{generator}_answers"]
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
        predictions=predictions, references=labels, average=None
    )["precision"].tolist()
    metrics["recall"] = recall.compute(
        predictions=predictions, references=labels, average=None
    )["recall"].tolist()
    metrics["f1"] = f1.compute(
        predictions=predictions, references=labels, average=None
    )["f1"].tolist()
    metrics["roc_auc"] = roc_auc.compute(
        prediction_scores=probs[np.arange(len(labels)), labels], references=labels
    )["roc_auc"]
    return metrics
