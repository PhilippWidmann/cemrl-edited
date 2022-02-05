analysis_config = dict(
    env_name='cheetash-stationary-dir',
    path_to_weights='./output/cheetah-stationary-vel/2022_01_31_14_06_27',  # CONFIGURE! path to experiment folder
    save_dir='./output_analysis/cheetah-stationary-vel/2022_01_31_14_06_27',
    save_prefix='',  # CONFIGURE! Prefix for all saved files during analysis run
    showcase_itr=150,  # CONFIGURE! epoch for which analysis is performed
    util_params=dict(
        base_log_dir='.temp',  # name of output directory
        use_gpu=True,  # set True if GPU available and should be used
        use_multiprocessing=False,  # set True if data collection should be parallelized across CPUs
        num_workers=8,  # number of CPU workers for data collection
        gpu_id=0,  # number of GPU if machine with multiple GPUs
        debug=False,  # debugging triggers printing and writes logs to debug directory
        plot=False  # plot figures of progress for reconstruction and policy training
    ),
    analysis_params=dict(
        train_example_cases=[],
        example_cases=[6, 7, 8, 9, 24, ],  # CONFIGURE! choose an array of test tasks
        single_episode_plots=[],
        multiple_episode_plots=[['time_vs_task_indicators', 'time_vs_rewards', 'time_vs_velocity_const_specification'],],
        log_and_plot_progress=False,  # CONFIGURE! If True: experiment will be logged to the experiment_database.json and plotted, If already logged: plot only
        save=True,  # CONFIGURE! If True: plots of following options will be saved to the experiment folder
        show=False,
        visualize_run=False,  # CONFIGURE! If True: learned policy of the showcase_itr will be played on example_case
        plot_time_response=False,  # CONFIGURE! If True: plot of time response
        plot_time_encoding=False,  # CONFIGURE! If True: plot the latent task and base task encodings over an episode
        plot_velocity_multi=False,  # CONFIGURE! If True: (only for velocity envs) plots time responses for multiple tasks
        plot_encoding=False,  # CONFIGURE! If True: plots encoding for showcase_itr
        produce_video=False,  # CONFIGURE! If True: produces a video of learned policy of the showcase_itr on example_case
        manipulate_time_steps=False,  # CONFIGURE! If True: uses time_steps different recent context, as specified below
        time_steps=10,  # CONFIGURE! time_steps for recent context if manipulated is enabled
        manipulate_change_trigger=False,  # CONFIGURE! for non-stationary envs: change the trigger for task changes
        change_params=dict(  # CONFIGURE! If manipulation enabled: set new trigger specification
            change_mode="time",
            change_prob=1.0,
            change_steps=100,
        ),
        manipulate_max_path_length=False,  # CONFIGURE! change max_path_length to value specified below
        max_path_length=800,
        manipulate_test_task_number=False,  # CONFIGURE! change test_task_number to value specified below
        test_task_number=10,
    ),
    env_params=dict(
        scripted_policy=False,
    ),
    algo_params=dict(
        merge_mode="mlp",
        use_fixed_seeding=True,
        seed=0,
    ),
    reconstruction_params=dict()
)
