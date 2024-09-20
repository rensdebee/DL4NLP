import argparse
from datasets import load_dataset, Dataset
import random
import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import evaluate
from scipy.special import softmax


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


def get_datasets(train_domain, test_domain, train_model, test_model, ratio=0.2):

    # Filter dataset based on the source
    ds = load_dataset("Hello-SimpleAI/HC3", "reddit_eli5", split=train_domain)
    ds = convert_dataset(ds, train_model)
    ds = ds.train_test_split(test_size=ratio)
    ds_train = ds["train"]
    ds_val = ds["test"]

    ds_test = load_dataset("Hello-SimpleAI/HC3", "finance", split=test_domain)
    ds_test = convert_dataset(ds_test, test_model)
    return ds_train, ds_val, ds_test


def main(args):
    seed(args.seed)

    train_dataset, eval_dataset, test_dataset = get_datasets(
        args.train_domain,
        args.test_domain,
        args.train_model,
        args.test_model,
        ratio=0.2,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, clean_up_tokenization_spaces=True
    )
    kwargs = dict(max_length=args.max_length, truncation=True)
    if args.pair:

        def tokenize_fn(example):
            return tokenizer(example["question"], example["answer"], **kwargs)

    else:

        def tokenize_fn(example):
            return tokenizer(example["answer"], **kwargs)

    print("Tokenizing and mapping...")
    train_dataset = train_dataset.map(tokenize_fn)
    eval_dataset = eval_dataset.map(tokenize_fn)
    test_dataset = test_dataset.map(tokenize_fn)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    output_dir = f"./models/{args.train_domain}_{args.test_domain}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=args.seed,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="epoch",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.eval_dataset(test_dataset)


if __name__ == "__main__":
    args = argparse.ArgumentParser("OOD AI detector trainer")
    args.add_argument(
        "-m",
        "--model-name",
        type=str,
        help="model name",
        default="FacebookAI/roberta-base",
    )
    args.add_argument(
        "-traind",
        "--train_domain",
        type=str,
        help="Split used for training",
        default="train_reddit_eli5",
    )
    args.add_argument(
        "-trainm",
        "--train_model",
        type=str,
        help="Split used for training",
        default="chatgpt",
    )
    args.add_argument(
        "-testd",
        "--test_domain",
        type=str,
        help="Split used for testing at the end",
        default="test_reddit_eli5",
    )
    args.add_argument(
        "-testm",
        "--test_model",
        type=str,
        help="Split used for training",
        default="chatgpt",
    )
    args.add_argument("-b", "--batch-size", type=int, default=16, help="batch size")
    args.add_argument("-e", "--epochs", type=int, default=2, help="batch size")
    args.add_argument(
        "--cuda", "-c", type=str, default="0", help="gpu ids, like: 1,2,3"
    )
    args.add_argument("--seed", "-s", type=int, default=42, help="random seed.")
    args.add_argument("--max-length", type=int, default=512, help="max_length")
    args.add_argument("--pair", action="store_true", default=False, help="paired input")

    args = args.parse_args()
    main(args)
