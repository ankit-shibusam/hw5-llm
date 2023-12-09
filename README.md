# HW 5 - Large Language Model

Team Members -
1) Ankit Shibusam = ashibusa@andrew.cmu.edu
2) Atharva Anand Joshi - atharvaa@andrew.cmu.edu
3) Ketan Ramaneti - kramanet@andrew.cmu.edu

## Start Training
0) Install the required dependencies on your local machine
```
pip install torch numpy transformers datasets tiktoken wandb tqdm pytorch-ignite
```

1) First generate the train and val data by running data/openwebtext/prepare.py. This script would fetch the OpenWebText data and perform a train-val split, followed by sub-word level tokenization using tiktoken. Finally it saves the process train and val data in the data/ folder.

```
$ python3 data/openwebtext/prepare.py
```

2) You can run the pretraining by simply running the train script. The configurations for the training can be set using the config.py file.

```
python3 train.py
```

3) For the finetuning tasks set up the data by running the below command -

```
python data/cnn_dailymail/prepare.py

python data/squad/prepare.py
```

4) Set the right file names and the required config variables in finetune_config.py and config.py. The other fields can be left untouched, but the file paths will need to be modified.

## TODO
1) Setup support for distributed training

2) Write code for sequential unfreezing.

## References
NanaoGPT - https://github.com/karpathy/nanoGPT/tree/master 
This repository was referred to for creating the LLM model.