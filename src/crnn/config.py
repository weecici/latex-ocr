class Config:
    data_path = "/kaggle/input/im2latex-100k/dataset"
    vocab_path = "models/vocab.pkl"
    save_dir = "./Checkpoint"

    # Model args
    word_emb_dim = 80
    rnn_h_dim = 512
    rnn_o_dim = 512
    enc_out_dim = 512
    att_dim = 512

    # Training args
    dropout = 0.2
    cuda = True
    batch_size = 64
    accumulation_steps = 1
    epoches = 50
    lr = 2e-4
    min_lr = 2e-5
    label_smoothing = 0.1

    # Sampling & Decay
    sample_method = "inv_sigmoid"  # choices: 'teacher_forcing', 'exp', 'inv_sigmoid'
    decay_k = 4000
    lr_decay = 0.5
    lr_patience = 5
    clip = 2.0

    # Learning rate warpup
    warmup_lr = 1e-4
    warmup_epochs = 2

    # Early stopping
    es_patience = 10
    es_min_delta = 1e-3

    # Misc
    seed = 2025

    # Checkpoint settings
    from_check_point = False

    # Evaluation
    result_path = "./Result.txt"
    ref_path = "./Ref.txt"
    eval_batch_size = 32
    beam_size = 5
    max_len = 150
