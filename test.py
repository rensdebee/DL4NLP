import argparse
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from utils import compute_metrics, seed, get_datasets
import json
import os


def main(args):
    seed(args.seed)

    _, _, test_dataset = get_datasets(
        test_domain=args.test_domain,
        test_generator=args.test_generator,
        ratio=0.1,
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
    test_dataset = test_dataset.map(tokenize_fn)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    if not args.out:
        output_dir = f"./models/eval_{args.test_domain}_{args.test_generator}"
    else:
        output_dir = args.out + f"/eval/{args.test_domain}_{args.test_generator}"

    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=args.seed,
        per_device_eval_batch_size=args.batch_size,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model,
        training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    test_results = trainer.evaluate(test_dataset)
    print(test_results)
    with open(output_dir + "/test_results.json", "w") as f:
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
        "-testd",
        "--test_domain",
        type=str,
        help="Split used for testing at the end",
        default="test_reddit_eli5",
    )
    args.add_argument(
        "-testg",
        "--test_generator",
        type=str,
        help="Split used for testing",
        default="chatgpt",
    )
    args.add_argument("-b", "--batch-size", type=int, default=16, help="batch size")
    args.add_argument(
        "--cuda", "-c", type=str, default="0", help="gpu ids, like: 1,2,3"
    )
    args.add_argument("--seed", "-s", type=int, default=42, help="random seed.")
    args.add_argument("--max-length", type=int, default=512, help="max_length")
    args.add_argument("--pair", action="store_true", default=False, help="paired input")
    args.add_argument(
        "--out", type=str, default=None, help="Path to store trainend model"
    )
    args = args.parse_args()
    main(args)
