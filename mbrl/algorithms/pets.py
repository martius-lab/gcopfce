import os
from typing import Optional
import time
from types import SimpleNamespace
import json
import gym
import numpy as np
import omegaconf
import torch

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math
from mbrl.util.plotting import VideoRecorder
from mbrl.util.env import _get_term_and_reward_fn, _get_goal_proj_fn

EVAL_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT
MAX_RUNTIME = 60*60*64


def train(
    env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    reward_fn: mbrl.types.RewardFnType,
    additional_modules: dict,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    goal_shape = env.goal_space.shape if hasattr(env, 'goal_space') else (0,)

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    work_dir = work_dir or os.getcwd()
    print(f"Results will be saved at {work_dir}.")

    if silent:
        logger = None
    else:
        logger = mbrl.util.Logger(work_dir)
        logger.register_group(
            mbrl.constants.RESULTS_LOG_NAME, EVAL_LOG_FORMAT, color="green"
        )
    video_recorder = VideoRecorder(work_dir, enabled=cfg.overrides.save_video)

    # -------- Create initial env dataset --------
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    grow_buffer = cfg.overrides.grow_buffer
    _, relabeling_fn = _get_term_and_reward_fn(cfg)
    goal_proj_fn = _get_goal_proj_fn(cfg)
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        goal_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        goal_type=dtype,
        reward_type=dtype,
        goal_proj_fn=goal_proj_fn,
        collect_trajectories=True,
    )

    # ---------------------------------------------------------
    # ---------- Create model environment and agent -----------
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, reward_fn, relabeling_fn, generator=torch_generator
    )
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=logger,
        additional_modules=additional_modules,
    )
    agent = mbrl.planning.create_trajectory_optim_agent_for_model(
        model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles,
        cost_discount=cfg.overrides.cost_discount, cost_aggregation=cfg.overrides.cost_aggregation,
        seed=cfg.seed
    )
    if cfg.overrides.acting_module in additional_modules:
        agent.act = additional_modules[cfg.overrides.acting_module].act
    [m.observe(generator=torch_generator, model=dynamics_model, replay_buffer=replay_buffer, env=env,
     work_dir=work_dir, agent=agent, model_env=model_env, logger=logger, relabeling_fn=relabeling_fn,
     additional_modules=additional_modules) for _, m in additional_modules.items()]
    [m.init() for _, m in additional_modules.items()]

    # ---------------------------------------------
    # -------- Handle resuming and loading --------

    # sticky variables that carry over across runs
    sv = SimpleNamespace(env_steps=0, current_trial=0, max_total_reward=-np.inf, n_offline_epochs=0)
    objects_to_save = ['model_trainer', 'model_env', 'torch_generator', 'agent', 'env', 'logger']
    
    def save_all(variables):
        replay_buffer.save(work_dir)
        dynamics_model.save(work_dir)
        [m.save(str(work_dir)) for m in additional_modules.values()]
        saved_states = {'rng': rng.__getstate__(), **sv.__dict__}
        saved_states.update({k: variables[k].get_state() for k in objects_to_save})
        torch.save(saved_states, os.path.join(work_dir,'saved_states.pt'))
        print(sv)
        print('Saving before exit.')

    start_time = time.time()
    check_time_and_save_and_exit = lambda x : [save_all(x), exit(0)] if time.time() > start_time + MAX_RUNTIME else []

    if os.path.exists(os.path.join(work_dir, 'saved_states.pt')):  # resume
        # resume
        replay_buffer.load(work_dir)
        dynamics_model.load(work_dir)
        [m.load(work_dir) for m in additional_modules.values()]
        saved_states = torch.load(os.path.join(work_dir, 'saved_states.pt'))
        for k in objects_to_save: locals()[k].set_state(saved_states[k])
        sv = SimpleNamespace(**{k: saved_states[k] for k in sv.__dict__})
        rng.__setstate__(saved_states['rng'])
        print(sv)
        print('Resuming saved states.')
    else:  # load if not resuming
        if cfg.overrides.model_load_dir is None:
            model_load_dir = None
        elif os.path.exists(cfg.overrides.model_load_dir):
            model_load_dir = cfg.overrides.model_load_dir
        replay_buffer.load(model_load_dir)
        dynamics_model.load(model_load_dir)
        [m.load(model_load_dir) for _, m in additional_modules.items()]

    # --------------------------------
    # -------- Prefill buffer --------

    # since we collect full trajectories, the second argument refers to the number of trajs
    if not len(replay_buffer):
        mbrl.util.common.rollout_agent_trajectories(
            env,
            cfg.algorithm.initial_exploration_steps // cfg.overrides.trial_length,
            mbrl.planning.RandomAgent(env),
            {},
            replay_buffer=replay_buffer,
            collect_full_trajectories=True,
            grow_buffer=grow_buffer
        )

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------

    if cfg.overrides.training_reward_fn in additional_modules.keys():
        on_step_callback = lambda x: additional_modules[cfg.overrides.training_reward_fn].on_step(x)
        reset_callback = lambda x: additional_modules[cfg.overrides.training_reward_fn].on_reset(x)
    else:
        on_step_callback, reset_callback = None, None

    while sv.env_steps < cfg.overrides.num_steps:
        obs = env.reset()
        agent.reset()
        video_recorder.init()
        kwargs_from_module = reset_callback(obs) if reset_callback else {}
        done = False
        total_reward = 0.0
        steps_trial = 0
        while not done:
            # --------------- Model Training -----------------
            if sv.env_steps % cfg.algorithm.freq_train_model == 0:
                online_modules = {k: v for k, v in additional_modules.items() if v.should_train_online}
                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    replay_buffer,
                    work_dir=None,  # models are saved only on exit_for_resume, or at end
                    train_model=cfg.overrides.train_model_online,
                    additional_modules=online_modules,
                    relabeling_fn=relabeling_fn,
                )

            # --- Doing env step using the agent and adding to model dataset ---
            next_obs, reward, done, _, kwargs_from_module = mbrl.util.common.step_env_and_add_to_buffer(
                env, obs, agent, kwargs_from_module, replay_buffer, callback=on_step_callback, grow_buffer=grow_buffer
            )

            obs = next_obs
            total_reward += reward
            steps_trial += 1
            sv.env_steps += 1
            video_recorder.record(env)
            if not (sv.env_steps % cfg.overrides.checkpoint_model_every):
                new_folder = work_dir+'/'+str(sv.env_steps)
                [c(new_folder) for c in [os.mkdir, replay_buffer.save, dynamics_model.save]]

            if debug_mode:
                print(f"Step {sv.env_steps}: Reward {reward:.3f}.")
        [m.log(sv.current_trial) for _, m in additional_modules.items()]
        if logger is not None:
            logger.log_data(
                mbrl.constants.RESULTS_LOG_NAME,
                {"env_step": sv.env_steps, "episode_reward": total_reward},
            )
        video_recorder.save(f'online_{sv.current_trial}')
        sv.current_trial += 1
        if debug_mode:
            print(f"Trial: {sv.current_trial }, reward: {total_reward}.")

        sv.max_total_reward = max(sv.max_total_reward, total_reward)

        check_time_and_save_and_exit(locals())

    print('Online training completed.')
    # ---------------------------------------------------------
    # ------------------- Offline Training --------------------

    # --- Reset reward_fn for offline phase ---
    _, reward_fn = mbrl.util.env._get_term_and_reward_fn(cfg)
    if cfg.overrides.get("learned_rewards", True):
        reward_fn = None
    if cfg.overrides.eval_reward_fn is not None:
        assert cfg.overrides.eval_reward_fn in additional_modules.keys()
        print(f'Overriding evaluation reward signal: {cfg.overrides.eval_reward_fn}.')
        reward_fn = additional_modules[cfg.overrides.eval_reward_fn].reward_fn
    model_env.set_reward_fn(reward_fn)

    if cfg.overrides.eval_reward_fn in additional_modules.keys():
        on_step_callback = lambda x: additional_modules[cfg.overrides.eval_reward_fn].on_step(x)
        reset_callback = lambda x: additional_modules[cfg.overrides.eval_reward_fn].on_reset(x)
    else:
        on_step_callback, reset_callback = None, None

    offline_modules = {k: v for k, v in additional_modules.items() if v.should_train_offline}

    while sv.n_offline_epochs < cfg.overrides.offline_epochs:
        mbrl.util.common.train_model_and_save_model_and_data(
            dynamics_model,
            model_trainer,
            cfg.overrides,
            replay_buffer,
            work_dir=None,  # models are saved only on exit_for_resume, or at end
            train_model=cfg.overrides.train_model_offline,
            additional_modules=offline_modules,
            relabeling_fn=relabeling_fn,
        )
        [m.log(sv.current_trial, sv.n_offline_epochs) for _, m in additional_modules.items()]
        sv.n_offline_epochs += 1
        if not ((sv.n_offline_epochs) % cfg.overrides.offline_evaluate_every):
            eval_metrics = mbrl.util.common.evaluate_final(env, agent, \
                offline_epochs=sv.n_offline_epochs, callback=on_step_callback, \
                reset_callback=reset_callback, work_dir=work_dir, video_recorder=video_recorder)
            json.dump(eval_metrics, open(work_dir+'/eval_metrics.json', 'a'))
        check_time_and_save_and_exit(locals())
        if sv.n_offline_epochs >= cfg.overrides.offline_epochs:
            save_all(locals())

    print('Offline training completed.')
    save_all(locals())

    # ---------------------------------------------------------
    # ------------------- Final Evaluation --------------------
    eval_fn = lambda prefix: mbrl.util.common.evaluate_final(env, agent,
            offline_epochs=cfg.overrides.offline_epochs, callback=on_step_callback,
            reset_callback=reset_callback, work_dir=work_dir, video_recorder=video_recorder, prefix=prefix)
    all_eval_metrics = {}

    if not cfg.overrides.acting_module:
        for eval_reward_fn in cfg.overrides.eval_reward_fns:
            if eval_reward_fn is None:
                _, reward_fn = mbrl.util.env._get_term_and_reward_fn(cfg)
                record_key = 'plan_default'
            else:
                assert eval_reward_fn in additional_modules.keys()
                print(f'Overriding evaluation reward signal: {eval_reward_fn}.')
                reward_fn = additional_modules[eval_reward_fn].reward_fn
                on_step_callback = lambda x: additional_modules[eval_reward_fn].on_step(x)
                reset_callback = lambda x: additional_modules[eval_reward_fn].on_reset(x)
                record_key = 'plan_'+eval_reward_fn
            model_env.set_reward_fn(reward_fn)
            eval_metrics = eval_fn(record_key)
            all_eval_metrics.update({record_key +'_'+k: v for k, v in eval_metrics.items()})

    model_env.set_reward_fn(None)
    on_step_callback, reset_callback = None, None
    for actor in cfg.overrides.eval_acting_modules:
        agent.act = additional_modules[actor].act
        record_key = 'policy_'+actor
        eval_metrics = eval_fn(record_key)
        all_eval_metrics.update({record_key +'_'+k: v for k, v in eval_metrics.items()})

    json.dump(all_eval_metrics, open(work_dir+'/final_eval_metrics.json', 'a'))
    return all_eval_metrics
