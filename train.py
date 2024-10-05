import argparse
import json
import os
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
)
from utils import compute_metrics, seed, get_datasets, MultiDatasetTrainer, StepCallback
import time
import warnings

# Filters out warning of Transformer update that we don't need to worry about
warnings.filterwarnings("ignore", category=FutureWarning)


def main(args):
    start = time.time()
    seed(args.seed)
    train_dataset, eval_dataset, test_dataset = get_datasets(
        train_domain=args.train_domain,
        train_generator=args.train_generator,
        test_domain=args.test_domain,
        test_generator=args.test_generator,
        ratio=args.split_ratio,
        seed=args.seed,
        train_multiple=args.train_multiple,
        preprocess=args.preprocess,
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
    for k, v in eval_dataset.items():
        eval_dataset[k] = v.map(tokenize_fn)
    for k, v in test_dataset.items():
        test_dataset[k] = v.map(tokenize_fn)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    if args.head_only:
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

    if not args.out:
        if args.train_multiple:

            output_dir = f"./models/train_conf_{('_'.join(args.train_multiple.split(',')))}_{args.train_generator}_head_only_{args.head_only}"
        else:
            output_dir = (
                f"./models/train_conf_{args.train_domain}_{args.train_generator}"
            )
        if args.head_only:
            output_dir += "_head_only"
        if args.preprocess:
            output_dir += "_cleaned"
    else:
        output_dir = args.out

    eval_output_dir = output_dir + f"/eval/"

    os.makedirs(eval_output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=args.seed,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=200,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    step_callback = StepCallback(
        eval_steps=training_args.eval_steps,
        trainer=None,
        test_model=args.test_generator,
        output_dir=output_dir,
    )

    trainer = MultiDatasetTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[step_callback],
    )

    step_callback.trainer = trainer
    step_callback.eval_datasets = eval_dataset

    test_results = trainer.evaluate_multiple_datasets(test_dataset, step_callback)
    print(test_results)

    trainer.train()

    test_results = trainer.evaluate_multiple_datasets(test_dataset, step_callback)
    print(test_results)

    print("Total train time:", time.time() - start)


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
        default="train",
    )
    args.add_argument(
        "-traing",
        "--train_generator",
        type=str,
        help="Split used for training",
        default="chatgpt",
    )
    args.add_argument(
        "-testd",
        "--test_domain",
        type=str,
        help="Split used for testing at the end",
        default="reddit_test",
    )
    args.add_argument(
        "-testg",
        "--test_generator",
        type=str,
        help="Split used for testing",
        default="chatgpt",
    )
    args.add_argument("-b", "--batch_size", type=int, default=16, help="batch size")
    args.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs")
    args.add_argument(
        "--cuda", "-c", type=str, default="0", help="gpu ids, like: 1,2,3"
    )
    args.add_argument("--seed", "-s", type=int, default=42, help="random seed.")
    args.add_argument("--max-length", type=int, default=512, help="max_length")
    args.add_argument("--pair", action="store_true", default=False, help="paired input")
    args.add_argument(
        "--head_only",
        action="store_true",
        default=False,
        help="Only train classification head",
    )
    args.add_argument(
        "--preprocess",
        action="store_true",
        default=False,
        help="Remove special characters from text",
    )
    args.add_argument(
        "--out", type=str, default=None, help="Path to store trainend model"
    )
    args.add_argument(
        "--train_multiple", type=str, default=False, help="Path to store trainend model"
    )
    args.add_argument(
        "--eval_steps", type=int, default=200, help="Number of steps between evaluation"
    )
    args.add_argument(
        "--split_ratio",
        type=float,
        default=0.1,
        help="Number of steps between evaluation",
    )

    args = args.parse_args()
    main(args)
