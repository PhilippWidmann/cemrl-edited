# Task Inference for Meta-Reinforcement Learning in Broad and Non-Parametric  Environments
by Philipp Widmann (Technical University Munich)

This is the reference implementation of the algorithm presented in this thesis, which is itself based on CEMRL. 
Our code is also available on https://github.com/PhilippWidmann/cemrl-edited

--------------------------------------

## Installation

### Mujoco
For our experiments we use MuJoCo200, which is free as of July 2022. To install it, do the following:
- Get both the free license key and installation files from the [Mujoco download page](http://www.roboti.us/download.html).
- Put the key file in `~/.mujoco`.
- Download mujoco200 and put the files in `~/.mujoco`.
- Set `LD_LIBRARY_PATH` to point to the MuJoCo binaries (`~/.mujoco/mujoco200/bin`).

Our experiments use only Mujoco200. However, parts/environments of the CEMRL legacy code may require older MuJoCo versions. 
If this is the case, they can be installed by repeating the above steps for the required version.

### Conda environment
For the remaining dependencies, we recommend using [miniconda](https://docs.conda.io/en/latest/miniconda.html).
Use the `environment.yml` file to [set up](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#) a conda environment.
The file lists all our installed packages on an Ubuntu 20.04 machine; no other configuration was tested. 
For your installation, make sure the correct GPU driver is installed and you use a matching version of CUDA toolkit for your GPU.

#### Dependencies
Our code depends on three submodules, which have to be installed into the conda environment.
After installing all required packages from the `environment.yml` file, use the following code to install the submodules `metaworld`, `rand_param_envs` and `url_benchmark`, respectively:

```
cd /path/to/submodules/<name_of_submodule>/
conda activate cemrl-edited
pip install -e .
```

## Run experiments
To reproduce an experiment, run:
```
conda activate cemrl-edited
python philipp_runner.py configs/thesis/[EXP].json
```
A quick, working starting example is `python philipp_runner.py configs/thesis/toy-goal-line.json`.

Output files will be written to `./output/[CONFIG_NAME]/[EXP_NAME]` where the experiment name is uniquely generated based on the date.
The file `progress.csv` contains statistics logged over the course of training, `variant.json` documents the used parameters and further files contain pickled data for specific epochs like weights or encodings.
The `exploration` subdirectory contains training progress logged by the exploration agent.
  

## Analyse experiments
With the script `cemrl/analysis_runner.py` you can generate analysis_plots as specified in `configs/analysis_config.py`. 
To simply generate the plots used in the thesis, use the scripts `analysis/thesis_plot_episodes.py` and `analysis/thesis_plot_progress.py` 


## Things to notice
- Most relevant code for the `cemrl-edited` algorithm itself is in the folder `./cemrl_edited`.
- Experiments are configured via `json` configuration files located in `./configs`, include a basic default configuration `./configs/default.py`.
- Adjust the `max_replay_buffer_size` according to the amount of collected data and supported memory.
