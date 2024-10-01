import argparse
import json
import os
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from utils import compute_metrics, seed, get_datasets
import time


def main(args):
    start = time.time()
    seed(args.seed)
    train_dataset, eval_dataset, test_dataset = get_datasets(
        train_domain=args.train_domain,
        train_generator=args.train_generator,
        test_domain=args.test_domain,
        test_generator=args.test_generator,
        ratio=0.1,
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
    eval_dataset = eval_dataset.map(tokenize_fn)
    test_dataset = test_dataset.map(tokenize_fn)

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

    eval_output_dir = output_dir + f"/eval/{args.test_domain}_{args.test_generator}"

    os.makedirs(eval_output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=args.seed,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Custom callback to capture the step
    class StepCallback(TrainerCallback):
        def __init__(self):
            self.step = 0

        def on_step_end(self, args, state, control, **kwargs):
            self.step = state.global_step

    step_callback = StepCallback()

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(
            eval_pred, step_callback.step, eval_output_dir
        ),
        callbacks=[step_callback],
    )

    test_results = trainer.evaluate(test_dataset)
    print(test_results)
    with open(eval_output_dir + "/test_results_beforetraining.json", "w") as f:
        json.dump(test_results, f)

    trainer.train()

    test_results = trainer.evaluate(test_dataset)
    print(test_results)
    with open(eval_output_dir + "/test_results.json", "w") as f:
        json.dump(test_results, f)

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

    args = args.parse_args()
    main(args)
