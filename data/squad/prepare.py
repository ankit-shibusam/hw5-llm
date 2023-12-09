import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

num_proc = 8

num_proc_load_dataset = num_proc

PAD_TOKEN = '<|pad|>'
SEP_1_TOKEN = '<|sep1|>'
SEP_2_TOKEN = '<|sep1|>'

enc = tiktoken.get_encoding("gpt2")
enc = tiktoken.Encoding(
    name="gpt2_with_sp_tokens",
    pat_str=enc._pat_str,
    mergeable_ranks=enc._mergeable_ranks,
    special_tokens={
        **enc._special_tokens,
        SEP_1_TOKEN : 50257,
        SEP_2_TOKEN : 50258
    }
)

if __name__ == '__main__':
    dataset = load_dataset("squad", num_proc=num_proc_load_dataset)

    def process(example):
        context_ids = enc.encode_ordinary(example['context']) # encode_ordinary ignores any special tokens
        context_ids.append(50258) 

        question_ids = enc.encode_ordinary(example['question'])
        question_ids.append(50258)

        answers_ids = enc.encode_ordinary(example['answers']['text'][0])
        answers_ids.append(enc.eot_token)
        
        data = context_ids + question_ids + answers_ids
        if len(data) > 1024:
            text = [0]
        else:
            # Pad input sequences to length of 1024
            text = [enc.eot_token]*1024
            text[:len(data)] = data

        out = {'data': text, 'data_len': len(data), 'context_lens': [len(context_ids)+ len(question_ids)]}

        return out

    # tokenize the dataset
    tokenized = dataset.map(
        process,
        remove_columns=['id', 'title', 'context', 'question', 'answers'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # Drop examples with input lengths greater than 1024
    tokenized['train'] = tokenized['train'].filter(lambda data: len(data['data']) == 1024)
    tokenized['validation'] = tokenized['validation'].filter(lambda data: len(data['data']) == 1024)

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        filename = os.path.join(os.path.dirname(__file__), f'{split}')
        print(np.array(dset['data']).shape)
        np.save(filename, np.array(dset['data']))


        filename = os.path.join(os.path.dirname(__file__), f'{split}_lens')
        np.save(filename, np.array(dset['context_lens']))
