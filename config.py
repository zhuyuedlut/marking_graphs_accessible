class CFG:
    # General
    debug = False
    num_proc = 24
    num_workers = 24
    gpus = 2

    # Data
    max_length = 1024
    image_height = 560
    image_width = 560

    # Training
    epochs = 2
    val_check_interval = 1.0  # how many times we want to validate during an epoch
    check_val_every_n_epoch = 1
    gradient_clip_val = 1.0
    lr = 3e-5
    lr_scheduler_type = "cosine"
    num_warmup_steps = 100
    seed = 42
    warmup_steps = 300
    output_path = "output"
    log_steps = 200
    batch_size = 2
    use_wandb = True
