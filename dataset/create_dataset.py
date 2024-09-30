import transformers
import torch
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import numpy as np
import argparse
from datetime import timedelta
import time

def main(args):
    # Function for adding the llama answers to a batch of the dataset object
    def add_llama_answers(entry, idx, llama_answers):
        entry['llama_answers'] = llama_answers[idx]
        return entry

    # Function for adding the dataset source as a column to the dataset
    def add_source(entry):
        entry['source'] = args.data_split
        return entry

    # Load dataset
    ds_train = load_dataset("Hello-SimpleAI/HC3", args.data_split, split="train")

    # Use only specified number of entries from the dataset
    ds_train = ds_train.select(range(args.start_idx, args.stop_idx))

    # Load model
    model = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model)

    batch_size = 32

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype = torch.float16,
        device_map="auto",
    )

    # Meassure Runtime
    starttime = time.perf_counter()

    # Full loop for generating and adding llama answers to their corresponding questions in the hc3 dataset
    # Loop over the dataset with batch_size sized steps
    for i in tqdm(range(0, len(ds_train), batch_size)):
        # Select the corresponding dataset entries (rows) to this batch
        # And create a duplicate dataset object for it
        ds_batch = ds_train.select(range(i, min(i + batch_size, len(ds_train))))
        
        # Create a list of all the questions for this batch
        questions = ds_batch['question']
        # Process the questions
        processed_questions = []
        for question in questions:
            # Add "xxx" to prompt for a more concise answer
            processed_questions.append(question + " (max. 170 words)")

        # Run the llama model for inference for the answers
        sequences = pipeline(
            processed_questions,
            do_sample = True,
            top_k = 10,
            num_return_sequences = 1,
            eos_token_id = tokenizer.eos_token_id,
            truncation = True,
            max_length = 400,
        )
        
        llama_answers = []
        # Remove the question part of the llama answer (output of llama prepends the prompt to the answer)
        for question, answer in zip(processed_questions, sequences):
            only_answer = answer[0]['generated_text'].split(question)[-1].strip()
            llama_answers.append([only_answer])
        
        # Add the llama answers only to the corresponding batch
        # Updating the current batch with the answers
        ds_batch = ds_batch.map(
            add_llama_answers,
            with_indices=True,
            fn_kwargs={'llama_answers': llama_answers}		
            )
        
        # Add the source (dataset source (wiki, reddit_eli5, etc)) aswell
        ds_batch = ds_batch.map(add_source)

        # Create a single dataset holding all data by concatenating all the batches
        if i == 0:
            new_dataset = ds_batch
        else:
            new_dataset = concatenate_datasets([new_dataset, ds_batch])

    # Report runtime
    duration = timedelta(seconds=time.perf_counter()-starttime)
    print('Job took: ', duration)

    # Push dataset with extra llama column to huggingface
    new_dataset.push_to_hub(f"DanteZD/HC3_plus_llama70B_{args.data_split}_start{args.start_idx}_stop{args.stop_idx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Dataset settings.
    parser.add_argument("--data_split", default="reddit_eli5", type=str)
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--stop_idx", default=200, type=int)

    args = parser.parse_args()
    main(args)
