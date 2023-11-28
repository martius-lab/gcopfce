from typing import Dict, Optional, Tuple, Union
from functools import partial
import gym
import gym.wrappers
import omegaconf

from mbrl.modules import create_module
import mbrl.planning
import mbrl.types


def _get_term_and_reward_fn(
    cfg: Union[omegaconf.ListConfig, omegaconf.DictConfig],
) -> Tuple[mbrl.types.TermFnType, Optional[mbrl.types.RewardFnType]]:
    """Returns a specified reward function."""
    import mbrl.env

    term_fn = getattr(mbrl.env.termination_fns, cfg.overrides.term_fn)
    if hasattr(cfg.overrides, "reward_fn") and cfg.overrides.reward_fn is not None:
        reward_fn = getattr(mbrl.env.reward_fns, cfg.overrides.reward_fn)
    else:
        reward_fn = getattr(mbrl.env.reward_fns, cfg.overrides.term_fn, None)

    return term_fn, reward_fn


def _get_goal_proj_fn(
    cfg: Union[omegaconf.ListConfig, omegaconf.DictConfig],
) -> mbrl.types.GoalProjFnType:
    """Returns function projecting observations to goals."""
    import mbrl.env
    return None if cfg.overrides.goal_proj_fn is None else \
        partial(getattr(mbrl.env.goal_proj_fns, cfg.overrides.goal_proj_fn), \
                compact_goal_space=cfg.overrides.env_params.compact_goal_space)


def _handle_learned_rewards_and_seed(
    cfg: Union[omegaconf.ListConfig, omegaconf.DictConfig],
    env: gym.Env,
    reward_fn: mbrl.types.RewardFnType,
    additional_modules: dict
) -> Tuple[gym.Env, mbrl.types.RewardFnType]:
    if cfg.overrides.get("learned_rewards", True):
        reward_fn = None
    
    if cfg.overrides.get("training_reward_fn", None) is not None:
        assert cfg.overrides.training_reward_fn in additional_modules.keys()
        print(f'Overriding reward signal: {cfg.overrides.training_reward_fn}.')
        reward_fn = additional_modules[cfg.overrides.training_reward_fn].reward_fn

    if cfg.seed is not None:
        env.seed(cfg.seed)
        env.observation_space.seed(cfg.seed + 1)
        env.action_space.seed(cfg.seed + 2)

    return env, reward_fn


def make_env(
    cfg: Union[Dict, omegaconf.ListConfig, omegaconf.DictConfig],
) -> Tuple[gym.Env, mbrl.types.TermFnType, Optional[mbrl.types.RewardFnType], dict]:
    """Creates an environment from a given OmegaConf configuration object.

    This method expects the configuration, ``cfg``,
    to have the following attributes (some are optional):

        - ``cfg.overrides.env``: a string description of the environment
        - ``cfg.overrides.term_fn``: a string indicating the environment's termination
        function to use when simulating the environment with the model. It should
        correspond to the name of a function in :mod:`mbrl.env.termination_fns`.
        - ``cfg.overrides.reward_fn``: a string indicating the environment's reward
        function to use when simulating the environment with the model. It should
        correspond to the name of a function in :mod:`mbrl.env.reward_fns`.
        - ``cfg.overrides.learned_rewards``: (optional) if present indicates that
        the reward function will be learned, in which case the method will return
        a ``None`` reward function.
        - ``cfg.overrides.trial_length``: (optional) if presents indicates the maximum length
        of trials.

    Args:
        cfg (omegaconf.DictConf): the configuration to use.

    Returns:
        (tuple of env, termination function, reward function, additional modules): returns the
        new environment, the termination function to use, and the reward function to use
        (or ``None`` if ``cfg.learned_rewards == True``).
    """
    # Handle the case where cfg is a dict
    cfg = omegaconf.OmegaConf.create(cfg)

    if "maze___" in cfg.overrides.env:
        from mbrl.env.d4rl_envs import MazeEnv
        env = MazeEnv(cfg.overrides.env.split("___")[1], **cfg.overrides.env_params)
        term_fn, reward_fn = _get_term_and_reward_fn(cfg)
    elif "kitchen___" in cfg.overrides.env:
        from mbrl.env.d4rl_envs import KitchenEnv
        env = KitchenEnv(cfg.overrides.env.split("___")[1], **cfg.overrides.env_params)
        term_fn, reward_fn = _get_term_and_reward_fn(cfg)
    elif "fetch___" in cfg.overrides.env:
        from mbrl.env.fetch_envs import FetchEnv
        env = FetchEnv(cfg.overrides.env.split("___")[1], **cfg.overrides.env_params)
        term_fn, reward_fn = _get_term_and_reward_fn(cfg)
    elif "pinpad___" in cfg.overrides.env:
        from mbrl.env.d4rl_envs import PinpadEnv
        env = PinpadEnv(cfg.overrides.env.split("___")[1], **cfg.overrides.env_params)
        term_fn, reward_fn = _get_term_and_reward_fn(cfg)
    else:
        raise ValueError("Invalid environment string.")

    additional_modules = {} if 'additional_modules' not in cfg.overrides else \
    {k: create_module(env, v) for k, v in cfg.overrides.additional_modules.items()}
    env, reward_fn = _handle_learned_rewards_and_seed(cfg, env, reward_fn, additional_modules)
    return env, term_fn, reward_fn, additional_modules
