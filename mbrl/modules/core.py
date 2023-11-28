import pathlib
from typing import Tuple, Union, Optional, Callable
import hydra
import omegaconf
import numpy as np
import torch
import torch.nn as nn
import gym

from mbrl.types import ModelInput
from mbrl.util.replay_buffer import AdditionalIterator
from mbrl.models.model_env import ModelEnv
from mbrl.util.math import Normalizer
from mbrl.planning import Agent
import mbrl.models.util as model_util


def create_module(
    env: Union[gym.Env, ModelEnv], module_cfg: omegaconf.DictConfig
):
    """Creates a module after completing its configuration given information from the environment.

    It will check for and complete any of the following keys:

        - "obs_dim": set to env.observation_space.shape
        - "goal_dim": set to env.goal_space.shape
        - "act_dim": set to env.action_space.shape

    Note:
        If the user provides any of these values in the Omegaconf configuration object, these
        *will not* be overridden by this function.
    """

    if "obs_dim" in module_cfg.keys() and "obs_dim" not in module_cfg:
        module_cfg.obs_dim = env.observation_space.shape[0]
    if "goal_dim" in module_cfg.keys() and "goal_dim" not in module_cfg:
        module_cfg.goal_dim = env.goal_space.shape[0]
    if "act_dim" in module_cfg.keys() and "act_dim" not in module_cfg:
        module_cfg.act_dim = env.action_space.shape[0]

    return hydra.utils.instantiate(module_cfg)


class Module(nn.Module, Agent):
    """Base class for all additional (possibly trainable) modules.

    All classes derived from `Module` should implement the following methods:

        - ``update``: updates the module given a batch of data
        - ``save``: saves the module to a given path.
        - ``load``: loads the module from a given path.
        - ``reward_fn``: acts as a reward function.
        - ``get_iter``: returns iterator needed for training.
    
    and the following properties:

        - ``name``: unique name.
        - ``should_train_online``: whether the model should be trained online.
        - ``should_train_ffline``: whether the model should be trained offline.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        [setattr(self, k, kwargs.get(k)) for k in ['should_train_online', 'should_train_offline', 'device']]
        self.obs_normalizer: Optional[Normalizer] = None
        self.goal_normalizer: Optional[Normalizer] = None
        if kwargs['normalize']:
            self.obs_normalizer = Normalizer(
                kwargs['obs_dim'],
                self.device,
                dtype=torch.double if kwargs['normalize_double_precision'] else torch.float,
            )
            self.goal_normalizer = Normalizer(
                kwargs['goal_dim'],
                self.device,
                dtype=torch.double if kwargs['normalize_double_precision'] else torch.float,
            )

        super().__init__()

    @property
    def name(self):
        """ Returns an unique name for this module.
        """
        return 'module'

    def update(
        self,
        model_in: ModelInput
    ) -> float:
        """ Possibly updates the module's parameters from a single batch.

        Args:
            model_in (tensor or batch of transitions): the inputs to the model.

        Returns:
            float: loss for logging
        """
        pass

    def on_epoch(
        self,
        epoch: int,
    ):
        return {}

    def on_step(
        self,
        experience: Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool],
    ):
        """Processes the experience after each environment step.

        Args:
            experience (Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]): 
                tuple containing observation, action, next_observation, reward, done signal.
        """
        return {}

    def on_reset(
        self,
        obs: np.ndarray,
    ):
        """Processes environment resets.

        Args:
            obs (np.ndarray): starting observation.
        """
        return {}

    def reward_fn(
        self,
        obs: torch.Tensor, 
        act: torch.Tensor,
        next_obs: torch.Tensor,
        goal: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes a reward function.

        Args:
            obs (torch.Tensor): observation preceding the action
            act (torch.Tensor): action chosen by the agent
            next_obs (torch.Tensor): observation following the action
            goal (torch.Tensor, optional): goal for goal-conditioned planning

        Returns:
            torch.Tensor: reward scalar
        """
        return torch.zeros((act.shape[0], 1), device=self.device)
    
    def get_iter(
        self,
    ) -> Callable:
        """Returns iterator to be used for training.

        Returns:
            Callable: buffer iterator
        """
        return AdditionalIterator

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves the model to the given directory. Skip model attribute."""
        state = self.get_state()
        torch.save(state, pathlib.Path(save_dir) / (self.name+'.pth'))
    
    def get_state(self):
        """Returns the state of the model as a dict."""
        state_dict = {k: v for k, v in self.state_dict().items() if not k.startswith('model.')}
        state = {'state_dict': state_dict}
        if self.obs_normalizer:
            state['obs_normalizer'] = self.obs_normalizer.get_state()
        if self.goal_normalizer:
            state['goal_normalizer'] = self.goal_normalizer.get_state()
        return state

    def init(self):
        """Initializes the module."""
        pass
    
    def load(self, load_dir: Union[str, pathlib.Path]):
        """Loads the model from the given path."""
        if load_dir is None: return
        if (pathlib.Path(load_dir) / (self.name+'.pth')).exists():
            state = torch.load(pathlib.Path(load_dir) / (self.name+'.pth'), map_location=self.device)
            self.set_state(state)
            print(f'Loaded {self.name}.')

    def set_state(self, state):
        """Sets the state of the model to the provided dict."""
        state_dict = self.state_dict()
        state_dict.update(state['state_dict']) 
        self.load_state_dict(state_dict)
        if self.obs_normalizer:
            self.obs_normalizer.set_state(state['obs_normalizer'])
        if self.goal_normalizer:
            self.goal_normalizer.set_state(state['goal_normalizer'])

    def observe(
        self,
        **kwargs
    ):
        """Gives the module a chance to save a reference to a few objects."""
        allowed_keys = ["generator", "model", "replay_buffer", "work_dir", "agent", "model_env", "logger", "relabeling_fn", "env"]
        [setattr(self, k, v) for k, v in kwargs.items() if k in allowed_keys]
        if 'logger' in kwargs:
            self.logger.register_group(self.name, self.EVAL_LOG_FORMAT, color="red")

    def log(
        self,
        trial_n: int,
        offline_epochs: Optional[int] = None,
    ):
        """Logs some statistics or images."""
        pass

    def get_module_input(self, batch):
        """Processes and normalizes a batch for training."""
        obs, act, next_obs, rew, done, goal = (model_util.to_tensor(e).to(self.device).float() for e in batch.astuple())
        obs, next_obs, goal = self.normalize_obs(obs), self.normalize_obs(next_obs), self.normalize_goal(goal)
        return obs, act, next_obs, rew, done, goal
    
    def normalize_obs(self, obs):
        return self.obs_normalizer.normalize(obs).float() if self.obs_normalizer else obs
    
    def normalize_goal(self, goal):
        return self.goal_normalizer.normalize(goal).float() if self.goal_normalizer else goal

    def update_normalizer(self, batch):
        # FIXME: deal with care, if we load the module, but the replay buffer is not
        # the same that the model was trained on, everything breaks.
        if self.obs_normalizer:
            self.obs_normalizer.update_stats(batch.obs)
        if self.goal_normalizer:
            self.goal_normalizer.update_stats(batch.desired_goals)
