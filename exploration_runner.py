import os
import click
import rlkit.torch.pytorch_util as ptu


@click.command()
@click.argument('config', default="configs/cheetah-target/cheetah-stationary-target-exploration.json")#toy-goal/toy-goal-line.json")
def showcase_exploration(config):
    import json
    from configs.default import default_config
    from philipp_runner import deep_update_dict
    variant = default_config

    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)

    ### Do one dummy training to have something to load for analysis
    # Results don't matter exploration agent will be trained later
    variant['algo_params']['exploration_agent'] = None
    variant['algo_params']['num_train_epochs'] = 1
    variant['algo_params']['num_policy_steps'] = 50
    variant['algo_params']['num_reconstruction_steps'] = 50
    variant['algo_params']['num_train_tasks_per_episode'] = 1
    variant['algo_params']['num_eval_trajectories'] = 1


    from philipp_runner import setup_environment, initialize_networks
    env, experiment_log_dir = setup_environment(variant)
    algorithm, networks, *_ = initialize_networks(variant, env, experiment_log_dir)
    if ptu.gpu_enabled():
        algorithm.to()

    algorithm.train()

    ### Plot exploration trajectories
    urlb_methods = ['smm']#['icm', 'proto', 'diayn', 'icm_apt', 'ind_apt', 'aps', 'smm', 'rnd', 'disagreement']
    methods = ['ensemble_urlb_' + m for m in urlb_methods]  # + ['line']
    for exploration_method in methods:
        import configs
        import shutil
        from analysis_runner import prepare_variant_file, analysis
        # Remove snapshot of previous exploration agent
        try:
            shutil.rmtree(os.path.join(experiment_log_dir, 'exploration'))
        except FileNotFoundError:
            print('No exploration directory found. Is this the first run?')

        variant_analysis = configs.analysis_config.analysis_config
        variant_analysis['path_to_weights'] = experiment_log_dir
        variant_analysis['showcase_itr'] = 0
        variant_analysis['save_dir'] = './output_analysis/' + experiment_log_dir.split('/')[-2] + '/exploration'
        variant_analysis['save_prefix'] = exploration_method
        variant_analysis = prepare_variant_file(variant_analysis)

        # At last, overwrite exploration_agent ith the desired one
        # This will then be loaded and initialized in analysis()
        variant_analysis['algo_params']['exploration_agent'] = exploration_method

        analysis(variant_analysis)


if __name__ == '__main__':
    showcase_exploration()
