CHK_PT_PATH = '/home/ankitshibusam/nanoGPT/pretraining-out/ckpt.pt'
DEVICE = 'cuda'
OUTPUT_DIR = '/home/ankitshibusam/nanoGPT/finetune-complete'
EPOCHS = 2
SUMMARY_ROOT = '/home/ankitshibusam/nanoGPT/data/cnn_dailymail'
SQUAD_ROOT = '/home/ankitshibusam/nanoGPT/data/squad'
BATCH_SIZE = 4
IGNORE_INDEX = -1

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
wandb_project = 'hw5'
wandb_run_name = 'finetune-complete'
scaler_enabled = True
dropout = 0.0
