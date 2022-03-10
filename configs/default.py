from configs.machine_config import machine_config_dict

# default experiment settings
# all experiments should modify these settings only as needed
default_config = dict(
    env_name='cheetah-non-stationary-vel',
    env_params=dict(
        n_train_tasks=100,  # number train tasks
        n_eval_tasks=30,  # number evaluation tasks tasks
        use_normalized_env=True,  # if normalized env should be used
        scripted_policy=False,  # if true, a scripted oracle policy will be used for data collection, only supported for metaworld
        use_state_decoder=True,
    ),
    path_to_weights=None, # path to pre-trained weights to load into networks
    train_or_showcase='train',  # 'train' for train new policy, 'showcase' to load trained policy and showcase
    showcase_itr=1000,  # training epoch from which to use weights of policy to showcase
    util_params=dict(
        base_log_dir=machine_config_dict['base_log_dir'],  # name of output directory
        temp_dir=machine_config_dict['temp_dir'],  # temp directory for quicksaving the encoder and decoder during training
        use_gpu=machine_config_dict['use_gpu'],  # set True if GPU available and should be used
        use_multiprocessing=machine_config_dict['use_multiprocessing'],  # set True if data collection should be parallelized across CPUs
        num_workers=machine_config_dict['num_workers'],  # number of CPU workers for data collection
        gpu_id=machine_config_dict['gpu_id'],  # number of GPU if machine with multiple GPUs
        debug=False,  # debugging triggers printing and writes logs to debug directory
        debug_encoding=False,
        plot=False  # plot figures of progress for reconstruction and policy training
    ),

    algo_params=dict(
        policy_mode='sac_single',
        num_policy_nets=1,
        use_relabeler=False, # if data should be relabeled
        use_combination_trainer=False,  # if combination trainer (gradients from Decoder and SAC should be used, currently broken
        use_data_normalization=True,  # if data become normalized, set in correspondence to use_combination_trainer
        use_sac_data_normalization=False,
        use_parametrized_alpha=False,  # alpha conditioned on z
        encoder_type='TimestepMLP',
        encoder_merge_mode="add",  # if encoder_type==TimestepMLP: how to merge y infos: "add", "add_softmax", "multiply", "linear", "mlp"
        encoder_omit_input=None,  # Debug parameter: Should actions be ignored by VAE
        decoder_omit_input=None,
        use_fixed_seeding=True,  # seeding, make comparison more robust
        seed=0,  # seed for torch and numpy
        batch_size_reconstruction=256,  # batch size reconstruction trainer
        batch_size_validation=4096,  # batch size for the encoder-decoder validation
        batch_size_combination=256,  # batch size combination trainer
        batch_size_policy=256,  # batch size policy trainer
        batch_size_relabel=1024,  # batch size relabeler
        time_steps=30,  # timesteps before current to be considered for determine current task
        latent_size=1,  # dimension of the latent context vector z
        sac_layer_size=300,  # layer size for SAC networks, value 300 taken from PEARL
        max_replay_buffer_size=10000000,  # write as integer!
        sampling_mode=[None, 'linear'][0],  # determines how to choose samples from replay buffer
        num_last_samples=10000000,  # if data_usage_sac == 'cut, number of previous samples to be used
        permute_samples=False,  # if order of samples from previous timesteps should be permuted (avoid learning by heart)
        num_train_epochs=250,  # number of overall training epochs
        snapshot_gap=20,  # interval to store weights of all networks like encoder, decoder, policy etc.
        num_reconstruction_steps=5000,  # number of training steps in reconstruction trainer per training epoch
        num_policy_steps=3000,  # number of training steps in policy trainer per training epoch
        num_train_tasks_per_episode=100,  # number of training tasks from which data is collected per training epoch
        num_transitions_initial=200,  # number of overall transitions per task while initial data collection
        num_transitions_per_episode=200,  # number of overall transitions per task while each epoch's data collection
        num_eval_trajectories=3,  # number evaluation trajectories per test task
        showcase_every=0,  # interval between training epochs in which trained policy is showcased
        num_showcase_deterministic=1,  # number showcase evaluation trajectories per test task, encoder deterministic
        num_showcase_non_deterministic=1,  # number showcase evaluation trajectories per test task, encoder deterministic
        max_path_length=200,  # maximum number of transitions per episode in the environment
        target_entropy_factor=1.0,  # target entropy from SAC
        sac_alpha=1.0,  # fixed alpha value in SAC when not using automatic entropy tuning
        lr_policy=3e-4,  # learning rate for the policy network optimizer
        lr_qf=3e-4  # learning rate for the q-function network optimizer
    ),

    reconstruction_params=dict(
        reconstruction_model='VAE',
        use_state_diff=False,  # determines if decoder uses state or state difference as target
        use_next_state_for_reward_decoder=True,  # whether next_state is part of the reward decoder input
        reconstruct_all_timesteps=True,
        component_constraint_learning=False,  # enables providing base class information to the class encoder
        prior_mode='fixedOnY',  # options: 'fixedOnY' and 'network, determine if prior comes from a linear layer or is fixed on y
        prior_sigma=0.5,  # simga on prior when using fixedOnY prior
        num_classes=1,  # number of base classes in the class encoder
        lr_encoder=3e-4,  # learning rate decoder (ADAM) 3e-4 when combine with combination trainer,
        lr_decoder=3e-4,  # learning rate decoder (ADAM) 3e-4 when combine with combination trainer,
        alpha_kl_z=1e-3,  # weighting factor KL loss of z distribution
        beta_kl_y=1e-3,  # # weighting factor KL loss of y distribution
        net_complex_enc_dec=10.0,  # determines overall net complextity in encoder and decoder (shared_dim = net_complex_enc_dec * input_dim)
        factor_qf_loss=1.0,  # weighting of state and reward loss compared to Qfunction in combination trainer
        train_val_percent=0.8,  # percentage of train samples vs. validation samples
        eval_interval=50,  # interval for evaluation with validation data and possible early stopping
        early_stopping_threshold=500,  # minimal epochs before early stopping after new minimum was found
        optim_first_order=False,
        optim_z_update_steps=1
    )
)
