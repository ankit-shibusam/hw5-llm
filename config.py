config = {
    # General training flags
    'eval_interval' : 100,
    'log_interval' : 1,
    'eval_iters' : 200,
    'out_dir' : 'result',
    'eval_only' : False, 
    'always_save_checkpoint' : True,
    'init_from' : 'scratch',

    # Wand flags
    'wandb_log' : True,
    'wandb_project' : 'hw5',
    'wandb_run_name' : 'run-10',

    # Model gflags
    'gradient_accumulation_steps' : 5 * 8,
    'batch_size' : 12,
    'block_size' : 1024,
    'n_layer' : 8,
    'n_head' : 12,
    'n_embd' : 768,
    'dropout' : 0.0,
    'bias' : False,

    # Optimizer gflags
    'learning_rate' : 6e-4,
    'max_iters' : 1000000,
    'weight_decay' : 1e-1,
    'beta1' : 0.9,
    'beta2' : 0.95,
    'grad_clip' : 1.0,

    # Learning rate scheduler gflags
    'decay_lr' : True,
    'warmup_iters' : 2000,
    'lr_decay_iters' : 30000,
    'min_lr' : 6e-5,
}