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


def test(
    model_name,
    test_domain="test_reddit_eli5",
    test_generator="chatgpt",
    batch_size=16,
    cuda="0",
    seed_value=42,
    max_length=512,
    pair=False,
    out=None,
    preprocess=False,
):
    seed(seed_value)

    _, _, test_dataset = get_datasets(
        test_domain=test_domain,
        test_generator=test_generator,
        ratio=0.1,
        preprocess=preprocess,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, clean_up_tokenization_spaces=True
    )
    kwargs = dict(max_length=max_length, truncation=True)
    if pair:

        def tokenize_fn(example):
            return tokenizer(example["question"], example["answer"], **kwargs)

    else:

        def tokenize_fn(example):
            return tokenizer(example["answer"], **kwargs)

    print("Tokenizing and mapping...")
    test_dataset = test_dataset.map(tokenize_fn)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    if not out:
        output_dir = f"./models/eval_{test_domain}_{test_generator}"
    else:
        output_dir = out + f"/eval/{test_domain}_{test_generator}"

    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=seed_value,
        per_device_eval_batch_size=batch_size,
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
    print(output_dir + "/test_results.json")
    with open(output_dir + "/test_results.json", "w") as f:
        json.dump(test_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("OOD AI detector trainer")
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        help="model name",
        default="FacebookAI/roberta-base",
    )
    parser.add_argument(
        "-testd",
        "--test_domain",
        type=str,
        help="Split used for testing at the end",
        default="test_reddit_eli5",
    )
    parser.add_argument(
        "-testg",
        "--test_generator",
        type=str,
        help="Split used for testing",
        default="chatgpt",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="batch size")
    parser.add_argument(
        "--cuda", "-c", type=str, default="0", help="gpu ids, like: 1,2,3"
    )
    parser.add_argument("--seed", "-s", type=int, default=42, help="random seed.")
    parser.add_argument("--max-length", type=int, default=512, help="max_length")
    parser.add_argument("--pair", action="store_true", default=False, help="paired input")
    parser.add_argument(
        "--out", type=str, default=None, help="Path to store trained model"
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        default=False,
        help="Remove special characters from text",
    )

    args = parser.parse_args()

    # Call the test function with individual arguments
    test(
        model_name=args.model_name,
        test_domain=args.test_domain,
        test_generator=args.test_generator,
        batch_size=args.batch_size,
        cuda=args.cuda,
        seed_value=args.seed,
        max_length=args.max_length,
        pair=args.pair,
        out=args.out,
        preprocess=args.preprocess,
    )
