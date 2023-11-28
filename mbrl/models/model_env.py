from typing import Dict, Optional, Tuple
import gym
import numpy as np
import torch

import mbrl.types
from . import Model


class ModelEnv:
    """Wraps a dynamics model into a gym-like environment.

    This class can wrap a dynamics model to be used as an environment. The only requirement
    to use this class is for the model to use this wrapper is to have a method called
    ``predict()``
    with signature `next_observs, rewards = model.predict(obs,actions, sample=, rng=)`

    Args:
        env (gym.Env): the original gym environment for which the model was trained.
        model (:class:`mbrl.models.Model`): the model to wrap.
        termination_fn (callable): a function that receives actions and observations, and
            returns a boolean flag indicating whether the episode should end or not.
        reward_fn (callable, optional): a function that receives actions and observations
            and returns the value of the resulting reward in the environment.
            Defaults to ``None``, in which case predicted rewards will be used.
        relabeling_fn (callable, optional): a relabeling function, which returns whether
            a given state acheives a given goal.
        generator (torch.Generator, optional): a torch random number generator (must be in the
            same device as the given model). If None (default value), a new generator will be
            created using the default torch seed.
    """

    def __init__(
        self,
        env: gym.Env,
        model: Model,
        termination_fn: mbrl.types.TermFnType,
        reward_fn: Optional[mbrl.types.RewardFnType] = None,
        relabeling_fn = None,
        generator: Optional[torch.Generator] = None,
    ):
        self.dynamics_model = model
        self.termination_fn = termination_fn
        self.reward_fn = reward_fn
        self.env_relabeling_fn = relabeling_fn
        self.device = model.device

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.goal_space = env.goal_space

        self._current_obs: torch.Tensor = None
        self._propagation_method: Optional[str] = None
        self._model_indices = None
        if generator:
            self._rng = generator
        else:
            self._rng = torch.Generator(device=self.device)
        self._return_as_np = True

    def reset(
        self, initial_obs_batch: np.ndarray, return_as_np: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Resets the model environment.

        Args:
            initial_obs_batch (np.ndarray): a batch of initial observations. One episode for
                each observation will be run in parallel. Shape must be ``B x D``, where
                ``B`` is batch size, and ``D`` is the observation dimension.
            return_as_np (bool): if ``True``, this method and :meth:`step` will return
                numpy arrays, otherwise it returns torch tensors in the same device as the
                model. Defaults to ``True``.

        Returns:
            (dict(str, tensor)): the model state returned by `self.dynamics_model.reset()`.
        """
        if isinstance(self.dynamics_model, mbrl.models.OneDTransitionRewardModel):
            assert len(initial_obs_batch.shape) == 2  # batch, obs_dim
        with torch.no_grad():
            model_state = self.dynamics_model.reset(
                initial_obs_batch.astype(np.float32), rng=self._rng
            )
        self._return_as_np = return_as_np
        return model_state if model_state is not None else {}

    def step(
        self,
        actions: mbrl.types.TensorType,
        model_state: Dict[str, torch.Tensor],
        goal: Optional[mbrl.types.TensorType] = None,
        sample: bool = False,
        return_distributions: Optional[bool] = False,
        compute_rewards: Optional[bool] = True
    ) -> Tuple[mbrl.types.TensorType, mbrl.types.TensorType, np.ndarray, Dict]:
        """Steps the model environment with the given batch of actions.

        Args:
            actions (torch.Tensor or np.ndarray): the actions for each "episode" to rollout.
                Shape must be ``B x A``, where ``B`` is the batch size (i.e., number of episodes),
                and ``A`` is the action dimension. Note that ``B`` must correspond to the
                batch size used when calling :meth:`reset`. If a np.ndarray is given, it's
                converted to a torch.Tensor and sent to the model device.
            model_state (dict(str, tensor)): the model state as returned by :meth:`reset()`.
            goal (tensor, optional): a goal, in case of goal-conditioned planning.
            sample (bool): if ``True`` model predictions are stochastic. Defaults to ``False``.
            return_distribution (bool, optional): whether to return full predicted distributions.
            compute_rewards (bool, optional): whether to consider or ignore predicted rewards.

        Returns:
            (tuple): contains the predicted next observation, reward, done flag and metadata.
            The done flag is computed using the termination_fn passed in the constructor.
        """
        assert len(actions.shape) == 2  # batch, action_dim
        with torch.no_grad():
            # if actions is tensor, code assumes it's already on self.device
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).to(self.device)
            (
                next_observs,
                pred_rewards,
                pred_terminals,
                next_model_state,
                mus,
                stds
            ) = self.dynamics_model.sample(
                actions,
                model_state,
                deterministic=not sample,
                rng=self._rng,
                return_distributions=True
            )
            if not compute_rewards:
                rewards = torch.zeros((actions.shape[0], 1), device=self.device)
            elif self.reward_fn is None:
                rewards = pred_rewards
            else:
                if goal is not None:
                    rewards = self.reward_fn(model_state['obs'], actions, next_observs, goal)
                else:
                    rewards = self.reward_fn(model_state['obs'], actions, next_observs)
            dones = self.termination_fn(actions, next_observs)

            if pred_terminals is not None:
                raise NotImplementedError(
                    "ModelEnv doesn't yet support simulating terminal indicators."
                )

            if self._return_as_np:
                next_observs = next_observs.cpu().numpy()
                rewards = rewards.cpu().numpy()
                dones = dones.cpu().numpy()

            if return_distributions:
                mus = mus.cpu().numpy() if self._return_as_np else mus
                stds = stds.cpu().numpy() if self._return_as_np else stds
                return next_observs, rewards, dones, next_model_state, mus, stds

            return next_observs, rewards, dones, next_model_state

    def render(self, mode="human"):
        pass

    def evaluate_action_sequences(
        self,
        action_sequences: torch.Tensor,
        initial_state: np.ndarray,
        goal: Optional[torch.Tensor],
        num_particles: int,
        discount: float = 1.0,
        aggregation: str = 'sum_values',
    ) -> torch.Tensor:
        """Evaluates a batch of action sequences on the model.

        Args:
            action_sequences (torch.Tensor): a batch of action sequences to evaluate.  Shape must
                be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size, horizon,
                and action dimension, respectively.
            initial_state (np.ndarray): the initial state for the trajectories.
            goal (tensor, optional): a goal, in case of goal-conditioned planning.
            num_particles (int): number of times each action sequence is replicated. The final
                value of the sequence will be the average over its particles values.
            discount (float, optional): the environment's discout factor.
            aggregation (str, optional): how costs should be aggregated across steps.

        Returns:
            (torch.Tensor): the accumulated reward for each action sequence, averaged over its
            particles.
        """
        with torch.no_grad():
            assert len(action_sequences.shape) == 3
            population_size, horizon, action_dim = action_sequences.shape
            # either 1-D state or 3-D pixel observation
            assert initial_state.ndim in (1, 3)
            tiling_shape = (num_particles * population_size,) + tuple(
                [1] * initial_state.ndim
            )
            initial_obs_batch = np.tile(initial_state, tiling_shape).astype(np.float32)
            if isinstance(goal, np.ndarray):
                goal = torch.from_numpy(goal).to(self.device)
            model_state = self.reset(initial_obs_batch, return_as_np=False)
            batch_size = initial_obs_batch.shape[0]
            total_uncertainty = torch.zeros(population_size,).to(self.device)
            terminated = torch.zeros(batch_size, 1, dtype=bool).to(self.device)
            trajectories, all_rewards, all_labels = [], [], []
            for time_step in range(horizon):
                action_for_step = action_sequences[:, time_step, :]
                action_batch = torch.repeat_interleave(
                    action_for_step, num_particles, dim=0
                )
                compute_rewards = (aggregation != 'n_step') or (time_step == horizon-1)
                obs, rewards, dones, model_state = self.step(
                    action_batch, model_state, goal, sample=True, compute_rewards=compute_rewards
                )
                rewards *= (discount**time_step)
                rewards = torch.nan_to_num(rewards, neginf=-1000)
                labels = self.env_relabeling_fn(None, None, obs, goal)
                labels *= (discount**time_step)
                labels = torch.nan_to_num(labels, neginf=-1000)
                total_uncertainty += obs.reshape(population_size, num_particles, obs.shape[-1]).var(1).mean(-1)
                rewards[terminated] = 0
                labels[terminated] = 0
                terminated |= dones
                all_rewards.append(rewards.reshape(-1, num_particles))
                all_labels.append(labels.reshape(-1, num_particles))
                trajectories.append(obs.reshape(-1, num_particles, obs.shape[-1]))

            all_labels = torch.stack(all_labels, 1).mean(-1)
            all_rewards = torch.stack(all_rewards, 1).mean(-1)
            assert aggregation in ['sum_values', 'max_values', 'n_step', 'median_n_step', 'exp_n_step']
            if aggregation == 'sum_values':
                total_rewards = all_rewards.sum(1)
            elif aggregation == 'max_values':
                total_rewards = all_rewards.max(1)[0]
            elif aggregation == 'n_step':
                all_labels[:, -1] = all_rewards[:, -1]
                total_rewards = all_labels.sum(1)
            else:
                bs = all_rewards.shape[0]
                vv = torch.repeat_interleave(all_rewards, horizon, 0) * torch.eye(horizon).repeat(bs, 1).to(self.device)
                rr = torch.repeat_interleave(all_labels, horizon, 0) * (torch.tril(torch.ones(horizon, horizon)) - torch.eye(horizon)).repeat(bs, 1).to(self.device)
                n_step_rets = (vv + rr).reshape(bs, horizon, horizon).sum(-1)
                if aggregation == 'median_n_step':
                    total_rewards = n_step_rets.median(-1)[0]
                elif aggregation == 'exp_n_step':
                    lambda_ = 0.95
                    discount = (lambda_ ** torch.arange(horizon).to(self.device)).unsqueeze(0)
                    discount[0, :-1] *= (1-lambda_)
                    total_rewards = (n_step_rets * discount).sum(-1)
                else:
                    raise NotImplementedError(f'Unknown aggregation: {aggregation}.')

            ######################################################################
            # Uncertainty penalization could be used to penalize degenerate      #
            # trajectories. However, setting a hard threshold to remove them     #
            # grants better stability and easier tuning.                         #
            # The threshold of 100 works across environments as input/outputs    #
            # are normalized.                                                    #
            ######################################################################

            total_uncertainty = torch.nan_to_num(total_uncertainty, posinf=1e10).clip(0, 1e10)
            total_rewards[total_uncertainty > 1e2] = -1e10

            ##########################################################################################
            # For future reference, this snippet implements penalization                             #
            # if penalty is not None:                                                                #
            #     # uncertainty is quantified as variance of particles                               #
            #     total_uncertainty = torch.nan_to_num(total_uncertainty, posinf=1e15).clip(0, 1e15) #
            #     total_rewards = total_rewards - penalty * total_uncertainty                        #
            ##########################################################################################

            return total_rewards, torch.stack(trajectories, 1).mean(-2), all_rewards

    def set_reward_fn(
        self,
        reward_fn: Optional[mbrl.types.RewardFnType] = None,
    ):
        """Replaces the reward function in use for the model.

        Args:
            reward_fn (callable, optional): a function that receives actions and observations
                and returns the value of the resulting reward in the environment.
                Defaults to ``None``, in which case predicted rewards will be used.
        """
        self.reward_fn= reward_fn

    variables_to_save = ['_current_obs', '_propagation_method', '_model_indices', '_return_as_np']

    def get_state(self):
        return {**{k: getattr(self, k) for k in self.variables_to_save}, 'generator_state': self._rng.get_state()}

    def set_state(self, state):
        self._rng.set_state(state['generator_state'])
        [setattr(self, k, state[k]) for k in self.variables_to_save]
