import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from utils import compute_metrics, seed, convert_dataset
import json

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
        save_strategy="steps",
        save_steps=1000
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
    test_results = trainer.evaluate(test_dataset)
    print(test_results, "w")
    with open(output_dir+"/test_results.json") as f:
        json.dump(test_results, f)

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
        help="Split used for testing",
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
