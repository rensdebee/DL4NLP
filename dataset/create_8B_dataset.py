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
    def add_llama_8B_answers(entry, idx, llama_8B_answers):
        entry['llama_8B_answers'] = llama_8B_answers[idx]
        return entry

    # Load dataset
    ds_train = load_dataset("DanteZD/HC3_plus_llama70B", split=args.data_split)

    # Use only specified number of entries from the dataset
    ds_train = ds_train.select(range(args.start_idx, args.stop_idx))

    # Load model
    model = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    batch_size = 32

    pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    )

    # Meassure runtime
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
            # For this model we specify a specific context to stop the model from over-generating
            message = [
                {"role": "system", "content": "You are a question answering machine, that answers in max 200 words."},
                {"role": "user", "content": question},
            ]
            processed_questions.append(message)

        # Run the llama model for inference for the answers
        sequences = pipeline(
            processed_questions,
            max_new_tokens=400,
        )
        
        llama_8B_answers = []
        # Remove the question part of the llama answer (output of llama prepends the prompt to the answer)
        for answer in sequences:
            only_answer = answer[0]['generated_text'][2]['content']
            llama_8B_answers.append([only_answer])
        
        # Add the llama answers only to the corresponding batch
        # Updating the current batch with the answers
        ds_batch = ds_batch.map(
            add_llama_8B_answers,
            with_indices=True,
            fn_kwargs={'llama_8B_answers': llama_8B_answers}		
            )
        
        # Create a single dataset holding all data by concatenating all the batches
        if i == 0:
            new_dataset = ds_batch
        else:
            new_dataset = concatenate_datasets([new_dataset, ds_batch])

    # Report Runtime
    duration = timedelta(seconds=time.perf_counter()-starttime)
    print('Job took: ', duration)

    # Push dataset with extra llama column to huggingface
    new_dataset.push_to_hub(f"DanteZD/HC3_plus_llama_{args.data_split}_start{args.start_idx}_stop{args.stop_idx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Dataset settings.
    parser.add_argument("--data_split", default="reddit_train", type=str)
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--stop_idx", default=200, type=int)

    args = parser.parse_args()
    main(args)
