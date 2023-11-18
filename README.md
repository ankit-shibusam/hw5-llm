# HW 5 - Large Language Model

Team Members -
1) Ankit Shibusam = ashibusa@andrew.cmu.edu
2) Atharva Anand Joshi - atharvaa@andrew.cmu.edu
3) Ketan Ramaneti - kramanet@andrew.cmu.edu

## Start Training
0) Install the required dependencies on your local machine
```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

1) First generate the train and val data by running data/load_data.py. This script would fetch the OpenWebText data and perform a train-val split, followed by sub-word level tokenization using tiktoken. Finally it saves the process train and val data in the data/ folder.

```
$ python3 data/load_data.py
```

2) You can run the pretraining by simply running the train script. The configurations for the training can be set using the config.py file.

```
python3 train.py
```
## TODO
1) Create the dataloader

2) Setup support for distributed training

3) Write code to load from saved checkpoints.

4) Code for finetuning tasks

## References
NanaoGPT - https://github.com/karpathy/nanoGPT/tree/master 
This repository was referred to for creating the LLM model.