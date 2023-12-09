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
    dataset = load_dataset("cnn_dailymail", '3.0.0', num_proc=num_proc_load_dataset)

    def process(example):
        article_ids = enc.encode_ordinary(example['article']) # encode_ordinary ignores any special tokens
        article_ids.append(50257) 

        highlights_ids = enc.encode_ordinary(example['highlights'])
        highlights_ids.append(enc.eot_token)

        data = article_ids + highlights_ids
        if len(data) > 1024:
            text = [0]
        else:
            # Pad sequences to length 1024
            text = [enc.eot_token]*1024
            text[:len(data)] = data

        out = {'data': text, 'data_len': len(data), 'article_lens': [len(article_ids)]}

        return out

    # tokenize the dataset
    tokenized = dataset.map(
        process,
        remove_columns=['article','highlights','id'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # Drop examples with input sequence lengths < 1024
    tokenized['train'] = tokenized['train'].filter(lambda data: len(data['data']) == 1024)
    tokenized['validation'] = tokenized['validation'].filter(lambda data: len(data['data']) == 1024)
    tokenized['test'] = tokenized['test'].filter(lambda data: len(data['data']) == 1024)

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        filename = os.path.join(os.path.dirname(__file__), f'{split}')
        np.save(filename, np.array(dset['data']))


        filename = os.path.join(os.path.dirname(__file__), f'{split}_lens')
        np.save(filename, np.array(dset['article_lens']))
