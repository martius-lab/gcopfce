# Goal-conditioned Offline Planning from Curious Exploration

This codebase accompanies an anonymous submission to NeurIPS 2023, and the project [website](https://sites.google.com/view/gcopfce).
The codebase borrows its structure, as well as several classes, from [mbrl-lib](https://github.com/facebookresearch/mbrl-lib), to which credit is due.

## Installation

We recommend to install dependencies in a virtual environment with Python 3.8.10:

    virtualenv --python=3.8 v_mbrl
    source v_mbrl/bin/activate
    pip install -r requirements.txt
    pip install -e .

## Execution

Unsupervised exploration can be executed with 

    python -m mbrl.run.main overrides=pets_maze_dis_icem

where ``maze`` can be replaced by ``pinpad``, ``fetch`` or ``kitchen``.
Results, including a replay buffer and a dynamics model, will be saved by default in a folder named ``exp``.

In order to train a value function and evaluate the agent, simply run

    python -m mbrl.run.main overrides=pets_maze_eval_dis_mbold overrides.model_load_dir=/path/to/exploration/results

By default, this trains a value function with TD3, and evaluates its actor network (``policy_distance``), as well as model-based planning
on the learned value function w/o (``plan_distance``) and w/ (``graph_distance``) graph-based value aggregation.
Results can once again be found in ``exp``; in particular, success rates are found in ``final_eval_metrics.json``.

To select a different value learning algorithm:

MBPO

    python -m mbrl.run.main overrides=pets_maze_eval_dis_mbold overrides.additional_modules.distance.enable_mbpo=True

MOPO

    python -m mbrl.run.main overrides=pets_maze_eval_dis_mbold overrides.additional_modules.distance.enable_mbpo=True overrides.additional_modules.distance.mbpo_lambda=0.1

CRR

    python -m mbrl.run.main overrides=pets_maze_eval_dis_mbold overrides.additional_modules.distance.enable_crr=True

Further hyperparameters are documented in ``run/conf/overrides/pets_maze_eval_dis_mbold``.

## Figure 5, 7, 9

A snippet that can be used to generate datapoints for Figures 5,7,9 can be found in ``run/plot.py``.
