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

def get_datasets(test_domain,test_model):
    # Filter dataset based on the source
    ds_test = load_dataset("Hello-SimpleAI/HC3", "finance", split=test_domain)
    ds_test = convert_dataset(ds_test, test_model)
    return ds_test


def main(args):
    seed(args.seed)

    test_dataset = get_datasets(
        args.test_domain,
        args.test_model,
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

    output_dir = f"./models/{args.test_domain}/eval"
    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=args.seed,
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
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    test_results = trainer.evaluate(test_dataset)
    print(test_results)
    with open(output_dir+"/test_results.json", "w") as f:
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
        "-testm",
        "--test_model",
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
    args = args.parse_args()
    main(args)
