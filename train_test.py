"""
pip install datasets evaluate scikit-learn torch==1.12.1 transformers
"""

"""
module load 2022; module load Anaconda3/2022.05; source activate DL4NLP
"""

import argparse
import os
import random
from datasets import load_dataset


_PARSER = argparse.ArgumentParser("dl detector")
_PARSER.add_argument("-i", "--input", type=str, help="input file path", default="en")
_PARSER.add_argument(
    "-m", "--model-name", type=str, help="model name", default="FacebookAI/roberta-base"
)
_PARSER.add_argument("-b", "--batch-size", type=int, default=16, help="batch size")
_PARSER.add_argument("-e", "--epochs", type=int, default=2, help="batch size")
_PARSER.add_argument("--cuda", "-c", type=str, default="0", help="gpu ids, like: 1,2,3")
_PARSER.add_argument("--seed", "-s", type=int, default=42, help="random seed.")
_PARSER.add_argument("--max-length", type=int, default=512, help="max_length")
_PARSER.add_argument("--pair", action="store_true", default=False, help="paired input")
_PARSER.add_argument(
    "--all-train", action="store_true", default=False, help="use all data for training"
)


_ARGS = _PARSER.parse_args()

if len(_ARGS.cuda) > 1:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # if cuda >= 10.2
os.environ["CUDA_VISIBLE_DEVICES"] = _ARGS.cuda


def main(args: argparse.Namespace):
    import numpy as np
    from datasets import Dataset, concatenate_datasets
    import evaluate
    import pandas as pd
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    def read_train_test():
        from datasets import load_dataset
        import pandas as pd
        from datasets import Dataset

        # Filter dataset based on the source
        ds = load_dataset("Hello-SimpleAI/HC3", "all")
        train_dataset = []
        test_dataset = []

        for row in ds["train"]:
            human = row["human_answers"]
            chat = row["chatgpt_answers"]

            if row["source"] == "reddit_eli5":
                # train
                for h in human:
                    print(h)
                    train_dataset.append([h, 0])
                for c in chat:
                    train_dataset.append([c, 1])
            else:
                # test
                for h in human:
                    test_dataset.append([h, 0])
                for c in chat:
                    test_dataset.append([c, 1])
        ds_train = Dataset.from_pandas(
            pd.DataFrame(train_dataset, columns=["answer", "label"])
        )
        ds_test = Dataset.from_pandas(
            pd.DataFrame(test_dataset, columns=["answer", "label"])
        )
        return ds_train, ds_test

    # if 'mix' in args.input:
    #     data = [read_train_test(args.input.replace('mix', m)) for m in ('text', 'sent')]
    #     train_dataset = concatenate_datasets([data[0][0], data[1][0]])
    #     test_dataset = concatenate_datasets([data[0][1], data[1][1]])
    # else:
    #     train_dataset, test_dataset = read_train_test(args.input)

    train_dataset, test_dataset = read_train_test()

    if args.all_train:
        train_dataset = concatenate_datasets([train_dataset, test_dataset])
        print("Using all data for training..")
        print(train_dataset)
        test_dataset = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    kwargs = dict(max_length=args.max_length, truncation=True)
    if args.pair:

        def tokenize_fn(example):
            return tokenizer(example["question"], example["answer"], **kwargs)

    else:

        def tokenize_fn(example):
            return tokenizer(example["answer"], **kwargs)

    print("Tokenizing and mapping...")
    train_dataset = train_dataset.map(tokenize_fn)
    if test_dataset is not None:
        test_dataset = test_dataset.map(tokenize_fn)

    # remove unused columns
    names = ["id", "question", "answer", "source"]
    tokenized_train_dataset = train_dataset
    if test_dataset is not None:
        tokenized_test_dataset = test_dataset
    else:
        tokenized_test_dataset = None
    print(tokenized_train_dataset)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=predictions, references=labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    output_dir = "./models/all/" + args.input  # checkpoint save path
    if args.pair:
        output_dir += "-pair"
    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=args.seed,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="no" if test_dataset is None else "steps",
        eval_steps=2000 if "sent" in args.input else 500,
        save_strategy="epoch",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main(_ARGS)
