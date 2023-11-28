import argparse
import hydra
from hydra import compose, initialize
import numpy as np
import torch

import mbrl
from mbrl.models.util import to_tensor
from mbrl.util.env import _get_term_and_reward_fn, _get_goal_proj_fn
from mbrl.util.common import create_one_dim_tr_model, create_replay_buffer
from mbrl.env.reward_fns import fetch, maze, kitchen, pinpad


# This script provides an example of how the occurrence of artifacts
# in learned value functions can be estimated.
# We assume that the first inequality for T-local optima is satisfied.
# The second inequality is estimated via sampling with
N_TRAJS = 50
HORIZON = 15


def plot(override, path):

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path="conf", job_name="example")
    overrides = [f'overrides={override}']
    cfg = compose(config_name="main", overrides=overrides)
    env, termination_fn, reward_fn, additional_modules = mbrl.util.env.make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    goal_shape = env.goal_space.shape if hasattr(env, 'goal_space') else (0,)
    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)
    dynamics_model = create_one_dim_tr_model(cfg, obs_shape, act_shape)
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    _, relabeling_fn = _get_term_and_reward_fn(cfg)
    goal_proj_fn = _get_goal_proj_fn(cfg)
    replay_buffer = create_replay_buffer(
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
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, reward_fn, relabeling_fn, generator=torch_generator
    )
    agent = mbrl.planning.create_trajectory_optim_agent_for_model(
        model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles,
        cost_discount=cfg.overrides.cost_discount, cost_aggregation=cfg.overrides.cost_aggregation,
        seed=cfg.seed
    )
    [m.observe(generator=torch_generator, model=dynamics_model, replay_buffer=replay_buffer, env=env,
    work_dir='/tmp', agent=agent, model_env=model_env, relabeling_fn=relabeling_fn,
    additional_modules=additional_modules) for _, m in additional_modules.items()]
    [m.init() for _, m in additional_modules.items()]

    replay_buffer.load(path)
    dynamics_model.load(path)
    [m.load(path) for _, m in additional_modules.items()]
    module = additional_modules['distance']

    @torch.no_grad()
    def compute_lm_occurrence(module, obs, goals):
        BS = min(5000, len(obs))
        all_obs = to_tensor(obs).to(module.device)
        goals = to_tensor(goals).to(module.device)

        population = mbrl.util.math.powerlaw_psd_gaussian(2.0, size=(N_TRAJS, module.act_dim, HORIZON),
            device=module.device, generator=module.generator).transpose(1, 2).clip(-1., 1.).cpu()
        module.model_env.old_reward_fn = module.model_env.reward_fn
        module.model_env.set_reward_fn(None)

        n_local_optima = 0
        for i in range(0, len(all_obs), BS):
            batch = all_obs[i:i+BS]
            initial_obs_batch = torch.tile(batch, (N_TRAJS, 1))
            model_state = module.model_env.reset(initial_obs_batch.cpu().numpy(), return_as_np=False)
            for time_step in range(HORIZON):
                endpoints, _, _, model_state = module.model_env.step(population[:, time_step, :].unsqueeze(1).repeat((1, len(batch), 1)).reshape(-1, module.act_dim), model_state, None, sample=True)

            for goal in goals:
                initial_dist = -module.reward_fn(None, None, batch, goal).reshape(-1)
                final_dist = -module.reward_fn(None, None, endpoints, goal)
                min_final_dist = final_dist.reshape((N_TRAJS, len(batch), -1)).min(0).values.reshape(-1)
                local_optima = ((initial_dist < min_final_dist).flatten())
                n_local_optima += local_optima.sum().item()

        module.model_env.set_reward_fn(module.model_env.old_reward_fn)
        return n_local_optima / (len(all_obs) * len(goals)), local_optima


    # load trajectories
    trajs = np.load(path+'/plan_distance_trajs.npy', allow_pickle=True)
    obs = np.array([[tt['observation'] for tt in t]for t in trajs])
    dg = np.array([[tt['desired_goal'] for tt in t]for t in trajs])

    # trim trajectories
    sorted_trajs, sorted_goals, sorted_succ = [], [], []
    for traj, goal in zip(obs, dg):
        traj = to_tensor(traj).to(module.device)
        goal = to_tensor(goal).to(module.device)
        gt_r_fns = {'fetch': fetch, 'maze': maze, 'kitchen': kitchen, 'pinpad': pinpad}
        r_fn = [v for k, v in gt_r_fns.items() if cfg.overrides.env.startswith(k)][0]
        rews = r_fn(None, None, traj, goal)
        if rews[0]:
            # skip trivial tasks
            continue
        rews = torch.cumsum(rews, 0)
        traj = traj[(rews == 0).squeeze()]
        goal = goal[(rews == 0).squeeze()]
        sorted_trajs.append(traj)
        sorted_goals.append(goal)
        sorted_succ.append((rews.sum() > 0).float().item())
    
    for traj, goal, succ in zip(sorted_trajs, sorted_goals, sorted_succ):
        values = module.reward_fn(None, None, traj, goal).squeeze()
        if len(values) <= HORIZON:
            non_monotonicities = 0
        else:
            non_monotonicities = ((values[HORIZON:] - values[:-HORIZON]) < 0).float().mean().item()
        _, local_optima = compute_lm_occurrence(module, traj, goal[[0]])
        local_optima[:-HORIZON] &= ((values[HORIZON:] - values[:-HORIZON]) < 0)
        print()
        print(f'Goal: {goal[0]}') 
        print(f'Success: {succ}')
        print(f'Occurrence of T-local optima: {local_optima.float().mean().item()}')
        print(f'Occurrence of non-monotonicities: {non_monotonicities}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--override', help='Config used for evaluation run.')
    parser.add_argument('--path', help='Path containing results for evaluation run.')
    args = parser.parse_args()
    plot(args.override, args.path)
