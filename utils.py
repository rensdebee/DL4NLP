import re
import numpy as np
from collections import Counter
import evaluate
from scipy.special import softmax
import random
import torch
import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets
from sklearn.metrics import roc_auc_score
import json
from transformers import TrainerCallback, Trainer
import os


def remove_special_characters(text):
    """Removes newline, tab, and double quote characters from a string."""

    # Use regular expressions to replace occurrences with empty strings
    cleaned_text = re.sub(r"[\n\t\"\*#]", "", text)

    return cleaned_text


def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def filter_dataset(ds, included_sources):
    "filters correct sources and ensures equal representation"

    datasets = [ds[included] for included in included_sources]
    counts = [len(d) for d in datasets]
    min_count = min(counts)
    datasets = [d.shuffle().select(range(min_count)) for d in datasets]

    return concatenate_datasets(datasets)


def get_datasets(
    train_domain=None,
    train_generator=None,
    test_domain=None,
    test_generator=None,
    ratio=0.2,
    seed=None,
    train_multiple=False,
    preprocess=True,
):
    dataset_url = "DanteZD/HC3_plus_llama70B"
    # Filter dataset based on the source
    ["reddit_train", "wiki_csai", "open_qa", "finance", "medicine"]

    if test_domain:
        test_domain = test_domain.strip().split(",")
    else:
        test_domain = []

    if train_domain and train_generator:
        if not train_multiple:
            ds = load_dataset(dataset_url, split=train_domain)
        else:
            ds = load_dataset(dataset_url)
            included_sources = train_multiple.split(",")
            ds = filter_dataset(ds, included_sources)
            print(ds)
        ds = convert_dataset(ds, train_generator, preprocess=preprocess)
        ds = ds.train_test_split(test_size=ratio, seed=seed, stratify_by_column="label")
        print(f"Train set {train_domain}:")
        ds_train = ds["train"]
        print(Counter(ds_train["label"]))
        ds_val = {}
        ds_val[f"{train_domain}_{train_generator}"] = ds["test"]
        for k, v in ds_val.items():
            print(f"Eval set {k}")
            print(Counter(v["label"]))

    else:
        ds_train = None
        ds_val = None

    if len(test_domain) > 0 and test_generator:
        ds_test = {}
        for domain in test_domain:
            ds = load_dataset(dataset_url, split=domain)
            ds = convert_dataset(ds, test_generator, preprocess=preprocess)
            name = "Test"
            if len(test_domain) > 1 and f"{domain}_{test_generator}" not in ds_val:
                ds_val[f"{domain}_{test_generator}"] = ds
                name = "Test/Eval"
            if f"{domain}_{test_generator}" in ds_val:
                ds_test[f"{domain}_{test_generator}"] = ds_val[
                    f"{domain}_{test_generator}"
                ]
            else:
                ds_test[f"{domain}_{test_generator}"] = ds
            print(f"{name} set {domain}_{test_generator}")
            print(Counter(ds_test[f"{domain}_{test_generator}"]["label"]))
    else:
        ds_test = None

    return ds_train, ds_val, ds_test


def convert_dataset(
    dataset,
    generator,
    preprocess=False,
):
    ds = []
    for row in dataset:
        q = row["question"]
        if preprocess:
            q = remove_special_characters(q)
        human = row["human_answers"]
        model = row[f"{generator}_answers"]
        for h in human[:1]:
            if preprocess:
                h = remove_special_characters(h)
            ds.append([q, h, 0])
        for m in model[:1]:
            if preprocess:
                m = remove_special_characters(m)
            ds.append([q, m, 1])

    ds = Dataset.from_pandas(pd.DataFrame(ds, columns=["question", "answer", "label"]))
    ds = ds.class_encode_column("label")

    return ds


def compute_metrics(eval_preds, steps=None, domain=None, model=None, output_dir=None):
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
        prediction_scores=probs[:, 1], references=labels
    )["roc_auc"]
    metrics["sk_roc_auc"] = roc_auc_score(labels, probs[:, 1])
    if steps is not None and output_dir and domain:
        eval_output_dir = output_dir + f"/eval/{domain}_{model}"
        os.makedirs(eval_output_dir, exist_ok=True)
        with open(eval_output_dir + f"/step_{steps}.json", "w") as f:
            json.dump(metrics, f)

    return metrics


def test_all_domains(
    model_name="models/train_conf_reddit_train_open_qa_finance_medicine_chatgpt_head_only_False/checkpoint-1516",
    test_generator="chatgpt",
):
    from custom_test import test

    domains = ["reddit_test", "wiki_csai", "open_qa", "finance", "medicine"]
    for domain in domains:
        test(
            model_name,
            test_domain=domain,
            test_generator=test_generator,
            out=model_name,
        )


def colorize(value):
    RED = "\033[91m"
    RESET = "\033[0m"
    try:
        float_value = float(value)
        if float_value < 0.8:
            return f"{RED}{float_value:.4f}{RESET}"
        return f"{float_value:.4f}"
    except (ValueError, TypeError):
        return value


def print_results(model_name, test_generator, not_trained_on):
    domains = ["reddit_test", "wiki_csai", "open_qa", "finance", "medicine"]
    domains_print = ["reddit_train", "wiki_csai", "open_qa", "finance", "medicine"]

    results = []
    for domain in domains:
        with open(
            f"{model_name}/eval/{domain}_{test_generator}/test_results.json"
        ) as f:
            results.append(json.load(f))

    header = "Domain\t\tAcc\tAUC\tpH\tpAI\trH\trAI\tf1H\tf1AI"
    print(f"NTO: {not_trained_on}")
    print(header)
    print("=" * 78)

    for domain, result in zip(domains_print, results):
        accuracy = colorize(result.get("eval_accuracy", "N/A"))
        roc_auc = colorize(result.get("eval_roc_auc", "N/A"))
        p = [colorize(val) for val in result.get("eval_precision", ["N/A", "N/A"])]
        r = [colorize(val) for val in result.get("eval_recall", ["N/A", "N/A"])]
        f1 = [colorize(val) for val in result.get("eval_f1", ["N/A", "N/A"])]

        print(
            f"{domain:<15}\t{accuracy}\t{roc_auc}\t{p[0]}\t{p[1]}\t{r[0]}\t{r[1]}\t{f1[0]}\t{f1[1]}"
        )
    print("\n\n")


class StepCallback(TrainerCallback):
    def __init__(self, eval_steps, trainer, test_model, output_dir):
        self.step = 0
        self.eval_steps = eval_steps
        self.trainer = trainer  # Reference to the trainer to call evaluation
        self.eval_datasets = None  # This will be set in the trainer
        self.model = test_model
        self.outpur_dir = output_dir

    def on_step_end(self, args, state, control, **kwargs):
        self.step = state.global_step

        # Trigger evaluation on multiple datasets every `eval_steps`
        if self.step % self.eval_steps == 0:
            print(f"Evaluating at step {self.step}...")
            self.trainer.evaluate_multiple_datasets(self.eval_datasets, self)


class MultiDatasetTrainer(Trainer):
    def evaluate_multiple_datasets(self, eval_datasets, step_callback):
        results = {}
        for domain, dataset in eval_datasets.items():
            print(f"Eval on {domain} at step {step_callback.step}")
            # Evaluate on each dataset
            data_loader = self.get_eval_dataloader(dataset)
            data_loader.batch_size = self.args.per_device_eval_batch_size
            output = self.prediction_loop(
                data_loader,
                description=f"Evaluation on {domain}",
                metric_key_prefix=f"{domain}",
            )

            # Call custom compute_metrics function and pass step + dataset_name
            metrics = compute_metrics(
                (output.predictions, output.label_ids),
                step_callback.step,
                domain,
                step_callback.model,
                step_callback.outpur_dir,
            )
            results[domain] = metrics
        return results
