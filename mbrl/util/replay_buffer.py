import pathlib
import warnings
from typing import Any, List, Optional, Sequence, Sized, Tuple, Type, Union, Callable
import numpy as np
import torch

from mbrl.types import TransitionBatch


def _consolidate_batches(batches: Sequence[TransitionBatch]) -> TransitionBatch:
    len_batches = len(batches)
    b0 = batches[0]
    obs = np.empty((len_batches,) + b0.obs.shape, dtype=b0.obs.dtype)
    act = np.empty((len_batches,) + b0.act.shape, dtype=b0.act.dtype)
    next_obs = np.empty((len_batches,) + b0.obs.shape, dtype=b0.obs.dtype)
    rewards = np.empty((len_batches,) + b0.rewards.shape, dtype=np.float32)
    dones = np.empty((len_batches,) + b0.dones.shape, dtype=bool)
    desired_goals = np.empty((len_batches,) + b0.desired_goals.shape, dtype=b0.desired_goals.dtype)
    for i, b in enumerate(batches):
        obs[i] = b.obs
        act[i] = b.act
        next_obs[i] = b.next_obs
        rewards[i] = b.rewards
        dones[i] = b.dones
        desired_goals[i] = b.desired_goals
    return TransitionBatch(obs, act, next_obs, rewards, dones, desired_goals)


class TransitionIterator:
    """An iterator for batches of transitions.

    The iterator can be used doing:

    .. code-block:: python

       for batch in batch_iterator:
           do_something_with_batch()

    Rather than be constructed directly, the preferred way to use objects of this class
    is for the user to obtain them from :class:`ReplayBuffer`.

    Args:
        transitions (:class:`TransitionBatch`): the transition data used to built
            the iterator.
        batch_size (int): the batch size to use when iterating over the stored data.
        shuffle_each_epoch (bool): if ``True`` the iteration order is shuffled everytime a
            loop over the data is completed. Defaults to ``False``.
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If None (default value), a new default generator will be used.
    """

    def __init__(
        self,
        transitions: TransitionBatch,
        batch_size: int,
        shuffle_each_epoch: bool = False,
        rng: Optional[np.random.Generator] = None,
    ):
        self.transitions = transitions
        self.num_stored = len(transitions)
        self._order: np.ndarray = np.arange(self.num_stored)
        self.batch_size = batch_size
        self._current_batch = 0
        self._shuffle_each_epoch = shuffle_each_epoch
        self._rng = rng if rng is not None else np.random.default_rng()

    def _get_indices_next_batch(self) -> Sized:
        start_idx = self._current_batch * self.batch_size
        if start_idx >= self.num_stored:
            raise StopIteration
        end_idx = min((self._current_batch + 1) * self.batch_size, self.num_stored)
        order_indices = range(start_idx, end_idx)
        indices = self._order[order_indices]
        self._current_batch += 1
        return indices

    def __iter__(self):
        self._current_batch = 0
        if self._shuffle_each_epoch:
            self._order = self._rng.permutation(self.num_stored)
        return self

    def __next__(self):
        return self[self._get_indices_next_batch()]

    def ensemble_size(self):
        return 0

    def __len__(self):
        return (self.num_stored - 1) // self.batch_size + 1

    def __getitem__(self, item):
        return self.transitions[item]


class BootstrapIterator(TransitionIterator):
    """A transition iterator that can be used to train ensemble of bootstrapped models.

    When iterating, this iterator samples from a different set of indices for each model in the
    ensemble, essentially assigning a different dataset to each model. Each batch is of
    shape (ensemble_size x batch_size x obs_size) -- likewise for actions, rewards, dones.

    Args:
        transitions (:class:`TransitionBatch`): the transition data used to built
            the iterator.
        batch_size (int): the batch size to use when iterating over the stored data.
        ensemble_size (int): the number of models in the ensemble.
        shuffle_each_epoch (bool): if ``True`` the iteration order is shuffled everytime a
            loop over the data is completed. Defaults to ``False``.
        permute_indices (boot): if ``True`` the bootstrap datasets are just
            permutations of the original data. If ``False`` they are sampled with
            replacement. Defaults to ``True``.
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If None (default value), a new default generator will be used.

    Note:
        If you want to make other custom types of iterators compatible with ensembles
        of bootstrapped models, the easiest way is to subclass :class:`BootstrapIterator`
        and overwrite ``__getitem()__`` method. The sampling methods of this class
        will then batch the result of of ``self[item]`` along a model dimension, where each
        batch is sampled independently.
    """

    def __init__(
        self,
        transitions: TransitionBatch,
        batch_size: int,
        ensemble_size: int,
        shuffle_each_epoch: bool = False,
        permute_indices: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(
            transitions, batch_size, shuffle_each_epoch=shuffle_each_epoch, rng=rng
        )
        self._ensemble_size = ensemble_size
        self._permute_indices = permute_indices
        self._bootstrap_iter = ensemble_size > 1
        self.member_indices = self._sample_member_indices()

    def _sample_member_indices(self) -> np.ndarray:
        member_indices = np.empty((self.ensemble_size, self.num_stored), dtype=int)
        if self._permute_indices:
            for i in range(self.ensemble_size):
                member_indices[i] = self._rng.permutation(self.num_stored)
        else:
            member_indices = self._rng.choice(
                self.num_stored,
                size=(self.ensemble_size, self.num_stored),
                replace=True,
            )
        return member_indices

    def __iter__(self):
        super().__iter__()
        return self

    def __next__(self):
        if not self._bootstrap_iter:
            return super().__next__()
        indices = self._get_indices_next_batch()
        batches = []
        for member_idx in self.member_indices:
            content_indices = member_idx[indices]
            batches.append(self[content_indices])
        return _consolidate_batches(batches)

    def toggle_bootstrap(self):
        """Toggles whether the iterator returns a batch per model or a single batch."""
        if self.ensemble_size > 1:
            self._bootstrap_iter = not self._bootstrap_iter

    @property
    def ensemble_size(self):
        return self._ensemble_size


def _sequence_getitem_impl(
    transitions: TransitionBatch,
    batch_size: int,
    sequence_length: int,
    valid_starts: np.ndarray,
    item: Any,
):
    start_indices = valid_starts[item].repeat(sequence_length)
    increment_array = np.tile(np.arange(sequence_length), len(item))
    full_trajectory_indices = start_indices + increment_array
    return transitions[full_trajectory_indices].add_new_batch_dim(
        min(batch_size, len(item))
    )


class AdditionalIterator(TransitionIterator):
    """    Iterator that is tailored for training additional modules.
    Additionally stores information on trajectories in order to
    allow goal relabeling and contrastive learning.

    Args:
        transitions (TransitionBatch): transitions from the replay buffer.
            They should not be shuffled.
        trajectory_indices (List[Tuple[int, int]]): trajectory indices
            from the buffer.
        batch_size (int): batch size for training.
        relabeling (Optional[str], optional): relabeling algorithm.
        relabeling_fn (Optional[Callable], optional): functoin computing
            rewards in goal space.
        goal_proj_fn (Optional[Callable], optional): function projecting
            observations to goal space.
        rng (Optional[np.random.Generator], optional): random number generator
    """

    def __init__(
        self,
        transitions: TransitionBatch,
        trajectory_indices: List[Tuple[int, int]],
        batch_size: int,
        relabeling: Optional[str] = None,
        relabeling_fn: Optional[Callable] = None,
        goal_proj_fn: Optional[Callable] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.trajectory_indices = trajectory_indices
        self.relabeling = relabeling
        self.relabeling_fn = relabeling_fn
        self.goal_proj_fn = goal_proj_fn
        self.id_last_obs_in_traj = np.zeros(len(transitions), dtype=int)
        for (a,b) in trajectory_indices:
            if b > a:
                self.id_last_obs_in_traj[a:b] = b-1
            else:
                self.id_last_obs_in_traj[a:-1] = b-1
                self.id_last_obs_in_traj[0:b] = b-1

        super().__init__(
            transitions,
            batch_size,
            shuffle_each_epoch=True,
            rng=rng,
        )


class GoalRelabelingIterator(AdditionalIterator):
    """
    Iterator that performs goal relabeling.
    Mainly used for distance learning.
    """

    def __init__(self, *args, **kwargs):
        self.p_rand_goals = kwargs.pop('p_rand_goals')
        self.p_geometric = kwargs.pop('p_geometric')
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        assert self.relabeling in ['mbold'], 'This iterator is meant to actually relabel trajectories.'
        if isinstance(item, slice):
            item = np.arange(item.stop)[item]
        batch = self.transitions[item]
        rand_goal_idxs = self._rng.integers(self.num_stored, size=len(item))
        future_goal_idxs = (item + self._rng.geometric(p=self.p_geometric, size=len(item)) - 1) % self.num_stored
        # if there is a wraparound, but the goal does not wraparound, do nothing, else trim
        future_goal_idxs = np.where((self.id_last_obs_in_traj[item] < item) * (item < future_goal_idxs),
                                    future_goal_idxs,
                                    np.minimum(future_goal_idxs, self.id_last_obs_in_traj[item]))
        sample_rand_goals = self._rng.uniform(size=len(item)) < self.p_rand_goals
        goal_idxs = np.where(sample_rand_goals, rand_goal_idxs, future_goal_idxs) % self.num_stored

        batch.desired_goals = self.goal_proj_fn(self.transitions[goal_idxs].next_obs)
        batch.rewards = (goal_idxs == item).astype(np.float32)
        batch.dones = (goal_idxs == item)
        return batch

class MOPOIterator(GoalRelabelingIterator):
    """
    Iterator that performs goal relabeling.
    Mainly used for distance learning.
    """

    def __getitem__(self, item):
        batch = super().__getitem__(item)
        if isinstance(item, slice):
            item = np.arange(item.stop)[item]
        uncertainty_term = self.transitions[item].rewards
        batch.rewards += uncertainty_term
        return batch


class ReplayBuffer:
    """A replay buffer with support for training/validation iterators and ensembles.

    This buffer can be pushed to and sampled from as a typical replay buffer.

    Args:
        capacity (int): the maximum number of transitions that the buffer can store.
            When the capacity is reached, the contents are overwritten in FIFO fashion.
        obs_shape (Sequence of ints): the shape of the observations to store.
        action_shape (Sequence of ints): the shape of the actions to store.
        goal_shape (Sequence of ints): the shape of the goals to store.
        obs_type (type): the data type of the observations (defaults to np.float32).
        action_type (type): the data type of the actions (defaults to np.float32).
        goal_type (type): the data type of the goals (defaults to np.float32).
        reward_type (type): the data type of the rewards (defaults to np.float32).
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If None (default value), a new default generator will be used.
        max_trajectory_length (int, optional): if given, indicates that trajectory
            information should be stored and that trajectories will be at most this
            number of steps. Defaults to ``None`` in which case no trajectory
            information will be kept. The buffer will keep trajectory information
            automatically using the done value when calling :meth:`add`.
        goal_proj_fn (Callable, optional): function projecting observations to 
            the goal space.

    .. warning::
        When using ``max_trajectory_length`` it is the user's responsibility to ensure
        that trajectories are stored continuously in the replay buffer.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: Sequence[int],
        action_shape: Sequence[int],
        goal_shape: Optional[Sequence[int]],
        obs_type: Type = np.float32,
        action_type: Type = np.float32,
        goal_type: Type = np.float32,
        reward_type: Type = np.float32,
        rng: Optional[np.random.Generator] = None,
        max_trajectory_length: Optional[int] = None,
        goal_proj_fn: Optional[Callable] = None,
    ):
        self.cur_idx = 0
        self.capacity = capacity
        self.num_stored = 0
        self.goal_proj_fn = goal_proj_fn

        self.trajectory_indices: Optional[List[Tuple[int, int]]] = None
        if max_trajectory_length:
            self.trajectory_indices = []
            capacity += max_trajectory_length
        self.obs = np.empty((capacity, *obs_shape), dtype=obs_type)
        self.next_obs = np.empty((capacity, *obs_shape), dtype=obs_type)
        self.action = np.empty((capacity, *action_shape), dtype=action_type)
        self.reward = np.empty(capacity, dtype=reward_type)
        self.done = np.empty(capacity, dtype=bool)
        self.desired_goal = np.empty((capacity, *goal_shape), dtype=goal_type)

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

        self._start_last_trajectory = 0

    @property
    def stores_trajectories(self) -> bool:
        return self.trajectory_indices is not None

    @staticmethod
    def _check_overlap(segment1: Tuple[int, int], segment2: Tuple[int, int]) -> bool:
        s1, e1 = segment1
        s2, e2 = segment2
        return (s1 <= s2 < e1) or (s1 < e2 <= e1)

    def remove_overlapping_trajectories(self, new_trajectory: Tuple[int, int]):
        cnt = 0
        for traj in self.trajectory_indices:
            if self._check_overlap(new_trajectory, traj):
                cnt += 1
            else:
                break
        for _ in range(cnt):
            self.trajectory_indices.pop(0)

    def _trajectory_bookkeeping(self, done: bool):
        self.cur_idx += 1
        if self.num_stored < self.capacity:
            self.num_stored += 1
        if self.cur_idx >= self.capacity:
            self.num_stored = max(self.num_stored, self.cur_idx)
        if done:
            self.close_trajectory()
        else:
            partial_trajectory = (self._start_last_trajectory, self.cur_idx + 1)
            self.remove_overlapping_trajectories(partial_trajectory)
        if self.cur_idx >= len(self.obs):
            warnings.warn(
                "The replay buffer was filled before current trajectory finished. "
                "The history of the current partial trajectory will be discarded. "
                "Make sure you set `max_trajectory_length` to the appropriate value "
                "for your problem."
            )
            self._start_last_trajectory = 0
            self.cur_idx = 0
            self.num_stored = len(self.obs)

    def close_trajectory(self):
        new_trajectory = (self._start_last_trajectory, self.cur_idx)
        self.remove_overlapping_trajectories(new_trajectory)
        self.trajectory_indices.append(new_trajectory)

        if self.cur_idx - self._start_last_trajectory > (len(self.obs) - self.capacity):
            warnings.warn(
                "A trajectory was saved with length longer than expected. "
                "Unexpected behavior might occur."
            )

        if self.cur_idx >= self.capacity:
            self.cur_idx = 0
        self._start_last_trajectory = self.cur_idx

    def add(
        self,
        obs: Union[np.ndarray, dict],
        action: np.ndarray,
        next_obs: Union[np.ndarray, dict],
        reward: float,
        done: bool,
    ):
        """Adds a transition (s, a, s', r, done) to the replay buffer.

        Args:
            obs (np.ndarray): the observation at time t.
            action (np.ndarray): the action at time t.
            next_obs (np.ndarray): the observation at time t + 1.
            reward (float): the reward at time t + 1.
            done (bool): a boolean indicating whether the episode ended or not.
        """

        if isinstance(obs, dict):
            self.desired_goal[self.cur_idx] = obs['desired_goal']
            obs = obs['observation']
            next_obs = next_obs['observation']

        self.obs[self.cur_idx] = obs
        self.next_obs[self.cur_idx] = next_obs
        self.action[self.cur_idx] = action
        self.reward[self.cur_idx] = reward
        self.done[self.cur_idx] = done

        if self.trajectory_indices is not None:
            self._trajectory_bookkeeping(done)
        else:
            self.cur_idx = (self.cur_idx + 1) % self.capacity
            self.num_stored = min(self.num_stored + 1, self.capacity)

    def add_batch(
        self,
        obs: Union[np.ndarray, dict],
        action: np.ndarray,
        next_obs: Union[np.ndarray, dict],
        reward: np.ndarray,
        done: np.ndarray,
    ):
        """Adds a transition (s, a, s', r, done) to the replay buffer.

        Expected shapes are:
            obs --> (batch_size,) + obs_shape
            act --> (batch_size,) + action_shape
            reward/done --> (batch_size,)

        Args:
            obs (np.ndarray): the batch of observations at time t.
            action (np.ndarray): the batch of actions at time t.
            next_obs (np.ndarray): the batch of observations at time t + 1.
            reward (float): the batch of rewards at time t + 1.
            done (bool): a batch of booleans terminal indicators.
        """

        goal_conditioned = False
        if isinstance(obs, dict):
            goal_conditioned = True
            desired_goal = obs['desired_goal']
            obs = obs['observation']
            next_obs = next_obs['observation']

        def copy_from_to(buffer_start, batch_start, how_many):
            buffer_slice = slice(buffer_start, buffer_start + how_many)
            batch_slice = slice(batch_start, batch_start + how_many)
            np.copyto(self.obs[buffer_slice], obs[batch_slice])
            np.copyto(self.action[buffer_slice], action[batch_slice])
            np.copyto(self.reward[buffer_slice], reward[batch_slice])
            np.copyto(self.next_obs[buffer_slice], next_obs[batch_slice])
            np.copyto(self.done[buffer_slice], done[batch_slice])
            if goal_conditioned:
                np.copyto(self.desired_goal[buffer_slice], desired_goal[batch_slice])

        _batch_start = 0
        buffer_end = self.cur_idx + len(obs)
        if buffer_end > self.capacity:
            copy_from_to(self.cur_idx, _batch_start, self.capacity - self.cur_idx)
            _batch_start = self.capacity - self.cur_idx
            self.cur_idx = 0
            self.num_stored = self.capacity

        _how_many = len(obs) - _batch_start
        copy_from_to(self.cur_idx, _batch_start, _how_many)
        self.cur_idx = (self.cur_idx + _how_many) % self.capacity
        self.num_stored = min(self.num_stored + _how_many, self.capacity)

        ends = list(np.where(self.done[:self.num_stored])[0])
        if self.num_stored == self.capacity:
            starts = [ends[-1]] + ends[:-1]
        else: 
            starts = [0] + ends[:-1]
        self.trajectory_indices = [[a,b] for a, b in zip(starts, ends)]

    def sample(self, batch_size: int) -> TransitionBatch:
        """Samples a batch of transitions from the replay buffer.

        Args:
            batch_size (int): the number of samples required.

        Returns:
            (tuple): the sampled values of observations, actions, next observations, rewards
            and done indicators, as numpy arrays, respectively. The i-th transition corresponds
            to (obs[i], act[i], next_obs[i], rewards[i], dones[i]).
        """
        indices = self._rng.choice(self.num_stored, size=batch_size)
        return self._batch_from_indices(indices)

    def sample_trajectory(self) -> Optional[TransitionBatch]:
        """Samples a full trajectory and returns it as a batch.

        Returns:
            (tuple): A tuple with observations, actions, next observations, rewards
            and done indicators, as numpy arrays, respectively; these will correspond
            to a full trajectory. The i-th transition corresponds
            to (obs[i], act[i], next_obs[i], rewards[i], dones[i])."""
        if self.trajectory_indices is None or len(self.trajectory_indices) == 0:
            return None
        idx = self._rng.choice(len(self.trajectory_indices))
        indices = np.arange(
            self.trajectory_indices[idx][0], self.trajectory_indices[idx][1]
        )
        return self._batch_from_indices(indices)

    def _batch_from_indices(self, indices: Sized) -> TransitionBatch:
        obs = self.obs[indices]
        next_obs = self.next_obs[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        done = self.done[indices]
        desired_goal = self.desired_goal[indices]

        return TransitionBatch(obs, action, next_obs, reward, done, desired_goal)

    def __len__(self):
        return self.num_stored

    def save(self, save_dir: Union[pathlib.Path, str]):
        """Saves the data in the replay buffer to a given directory.

        Args:
            save_dir (str): the directory to save the data to. File name will be
                replay_buffer.npz.
        """
        path = pathlib.Path(save_dir) / "replay_buffer.npz"
        np.savez(path, **self.get_state())

    def load(self, load_dir: Union[pathlib.Path, str]):
        """Loads transition data from a given directory.

        Args:
            load_dir (str): the directory where the buffer is stored.
        """
        if load_dir:
            path = pathlib.Path(load_dir) / "replay_buffer.npz"
            if path.exists():
                self.set_state(np.load(path))

    def get_all(self, shuffle: bool = False) -> TransitionBatch:
        """Returns all data stored in the replay buffer.

        Args:
            shuffle (int): set to ``True`` if the data returned should be in random order.
            Defaults to ``False``.
        """
        if shuffle:
            permutation = self._rng.permutation(self.num_stored)
            return self._batch_from_indices(permutation)
        else:
            return TransitionBatch(
                self.obs[: self.num_stored],
                self.action[: self.num_stored],
                self.next_obs[: self.num_stored],
                self.reward[: self.num_stored],
                self.done[: self.num_stored],
                self.desired_goal[: self.num_stored],
            )
    
    def get_trajectory_indices(self) -> Optional[List[Tuple[int, int]]]:
        """Returns trajectory indices.

        Returns:
            Optional[List[Tuple[int, int]]]: list containing a tuple for each trajectory.
                The first element of the tuple is the index at which the trajectory begins,
                while the second element is the index following the last element of the
                trajectory.
        """
        return self.trajectory_indices

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    def get_state(self):
        state_attr = ['obs', 'next_obs', 'action', 'reward', 'done',
                      'desired_goal', 'trajectory_indices', 'cur_idx',
                      '_start_last_trajectory', 'capacity', 'num_stored']
        return {k: getattr(self, k) for k in state_attr}

    def set_state(self, state):
        self.obs = state["obs"]
        self.next_obs = state["next_obs"]
        self.action = state["action"]
        self.reward = state["reward"]
        self.done = state["done"]
        self.desired_goal = state["desired_goal"]
        # TODO: the following ifs are built to support MOPO SAC buffers.
        # The saving format should be unified.
        self.num_stored = state['num_stored']
        self.capacity = state['capacity']
        self.cur_idx = state['cur_idx']
        if not isinstance(self.num_stored, int):
            self.num_stored = self.num_stored.item()
        if not isinstance(self.capacity, int):
            self.capacity = self.capacity.item()
        if not isinstance(self.cur_idx, int):
            self.cur_idx = self.cur_idx.item()
        self._start_last_trajectory = state['_start_last_trajectory']
        if "trajectory_indices" in state and len(state["trajectory_indices"]):
            self.trajectory_indices = state["trajectory_indices"]
            if not isinstance(self.trajectory_indices, list):
                self.trajectory_indices = self.trajectory_indices.tolist()
