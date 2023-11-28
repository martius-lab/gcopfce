from gym.envs.registration import register
from typing import Any, Dict,Tuple
from abc import ABC, abstractmethod
import numpy as np
import gym
import d4rl
import gymnasium
import mujoco
from gymnasium_robotics.envs.franka_kitchen.ik_controller import IKController
from gymnasium_robotics.utils.mujoco_utils import robot_get_obs


PINPAD = \
        "#####\\"+\
        "#OOO#\\"+\
        "#OOO#\\"+\
        "#OOO#\\"+\
        "#####"


register(
    id='maze2d-pinpad-v0',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=300,
    kwargs={
        'maze_spec':PINPAD,
        'reward_type':'sparse',
        'reset_target': False,
    }
)


class GCEnv(ABC):
    """Abstract class for goal-conditioned environments."""

    @abstractmethod
    def set_goal(
        self, goal: np.ndarray
    ):
        """Sets goal; useful for precise evaluation."""

    @abstractmethod
    def set_pos(
        self, pos: np.ndarray
    ):
        """Sets agent position; useful for precise evaluation."""

    @abstractmethod
    def sample_goal(
        self
    ) -> np.ndarray:
        """Samples a goal from the environment."""

    @abstractmethod
    def get_state(
        self
    ) -> dict:
        """Gets state."""

    @abstractmethod
    def set_state(
        self,
        state
    ):
        """Sets state."""

class MazeEnv(GCEnv):
    """Wraps D4RL pointmaze in a goal-conditioned framework.

    Args:
        name (str): the name of the environment to create.
        compact_goal_space (bool): flag shrinking goal space to only include position
        fixed_start (bool): resets the agent's position to the same value for each episode
    """

    def __init__(
        self, name: str, compact_goal_space: bool, fixed_start: bool
    ):
        self.name = name
        self._env = gym.make(name)
        self.compact_goal_space = compact_goal_space
        self.fixed_start = fixed_start
        self.goal_size = 2 if self.compact_goal_space else 4
        self.goal_space = gym.spaces.Box(low=np.array([-np.inf]*self.goal_size, dtype=np.float32),
                                         high=np.array([np.inf]*self.goal_size, dtype=np.float32))

    def __getattr__(
        self, attr: str
    ) -> Any:
        """Gets attribute from wrapped environment.

        Args:
            attr (str): attribute name

        Returns:
            Any: requested attribute
        """
        return getattr(self._env, attr)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Returns observation and goal.

        Returns:
            Dict[str, np.ndarray]: normalized obs
        """
        # pad goal with zeros in case of non-compact goal space
        goal = np.zeros(self.goal_size)
        goal[:2] = self._env.get_target()
        return dict(observation=self._obs,
                    desired_goal=goal,
                    achieved_goal=self._obs[:self.goal_size])
    
    def set_goal(
        self, goal: np.ndarray
    ):
        """Sets goal; useful for precise evaluation.

        Args:
            goal (np.ndarray): environment goal
        """
        self._env.set_target(goal)
        return self._get_obs()

    def set_pos(
        self, pos: np.ndarray
    ):
        """Sets agent position; useful for precise evaluation.

        Args:
            goal (np.ndarray): agent position excuding velocities
        """
        if pos is not None:
            self._obs = self._env.reset_to_location(pos)
        return self._get_obs()

    def reset(self) -> np.ndarray:
        """Resets the environment.

        Returns:
            np.ndarray: current observation
        """
        self._obs = self._env.reset()
        # by default, initial position is sampled uniformly from all empty/goal squares
        if self.fixed_start:
            if '4room' in self.name:
                # set position opposite to goal
                self.set_pos(self._env.goal_locations[0][::-1])
            elif 'umaze' in self.name:
                self.set_pos((3, 1))
            elif 'tunnel' in self.name:
                self.set_pos((3, 3))
            else:
                # get first empty square
                self.set_pos(self.empty_and_goal_locations[0])
        return self._get_obs()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Advances simulation by a step.

        Args:
            action (int): action to be executed

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: tuple containing
                observation, reward, done signal and infos
        """
        self._obs, rew, done, info = self._env.step(action)
        info = {'distance': np.linalg.norm(self._obs[:2] - self._env.get_target())}
        return self._get_obs(), rew, done, info

    def sample_goal(
        self
    ) -> np.ndarray:
        """Samples a goal from the environment."""
        goal_idx = self.np_random.randint(len(self.empty_and_goal_locations))
        goal = np.zeros(self.goal_size)
        goal[:2] = self.empty_and_goal_locations[goal_idx]
        return goal

    def get_eval_cases(self):
        goals = [{
            'open':[(1,1),(1,5),(3,1),(3,5)],
            'medium':[(3,4),(6,6),(1,6),(6,1)],
            'large':[(1,10),(7,1),(5,4)],
            'umaze':[(3,3),(1,3),(1,2),(1,1)],
            '4room':[(7,1),(2,6),(6,2),(6,6)]
            }[k] for k in ['4room', 'umaze', 'medium', 'open', 'large'] if k in self.name][0]
        return [(None, g) for g in goals]

    def get_state(self):
        """Gets state."""
        return self.np_random.__getstate__()

    def set_state(self, state):
        """Sets state."""
        self.np_random.__setstate__(state)


class PinpadEnv(MazeEnv):
    """Wraps D4RL pointmaze in a goal-conditioned framework.

    Args:
        name (str): the name of the environment to create.
        compact_goal_space (bool): flag shrinking goal space to only include position
        n_goals (int, optional): number of buttons to press/remember
        random_reset (bool, optional): whether to initialize the history randomly
        trial_length (int, optional): timelimit
    """

    def __init__(
        self, name: str, compact_goal_space: bool, n_goals=2, random_reset=True, trial_length=300
    ):
        self.name = name
        self.random_reset = random_reset
        self.trial_length = trial_length
        self._env = gym.make(f'maze2d-{self.name}')
        self.compact_goal_space = compact_goal_space
        self.n_goals = n_goals
        self.goal_size = self.n_goals
        self.obs_size = 4 + self.n_goals
        self.goal_space = gym.spaces.Box(low=np.array([-np.inf]*self.goal_size, dtype=np.float32),
                                         high=np.array([np.inf]*self.goal_size, dtype=np.float32))
        self.observation_space = gym.spaces.Box(low=np.array([-np.inf]*self.obs_size, dtype=np.float32),
                                                high=np.array([np.inf]*self.obs_size, dtype=np.float32))

    def __getattr__(
        self, attr: str
    ) -> Any:
        """Gets attribute from wrapped environment.

        Args:
            attr (str): attribute name

        Returns:
            Any: requested attribute
        """
        return getattr(self._env, attr)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Returns observation and goal.

        Returns:
            Dict[str, np.ndarray]: normalized obs
        """
        history = np.zeros(self.n_goals)
        history[:] = self._history[-self.n_goals:]
        history += np.random.normal(size=history.shape) * 0.05
        return dict(observation=np.concatenate([self._obs, history]),
                    desired_goal=self._goal,
                    achieved_goal=history)
    
    def set_goal(
        self, goal: np.ndarray
    ):
        """Sets goal; useful for precise evaluation.

        Args:
            goal (np.ndarray): environment goal
        """
        self._env.set_target((2,2))
        self._goal = goal
        return self._get_obs()

    def set_pos(
        self, pos: np.ndarray
    ):
        """Sets agent position; useful for precise evaluation.

        Args:
            goal (np.ndarray): agent position excuding velocities
        """
        if pos is not None:
            self._obs = self._env.reset_to_location(pos[:2])[:4]
            if len(pos) > 2:
                self._history = [e for e in pos[2:]]
        return self._get_obs()

    def reset(self) -> np.ndarray:
        """Resets the environment.

        Returns:
            np.ndarray: current observation
        """
        if self.random_reset:
            self._history = [n for n in self.sample_goal()]
        else:
            self._history = [0. for _ in range(self.n_goals)]
        self._goal = self.sample_goal()
        self._obs = self._env.reset()[:4]
        # by default, initial position is sampled uniformly from all empty/goal squares
        self._update_flag()
        self.step_counter = 0
        return self._get_obs()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Advances simulation by a step.

        Args:
            action (int): action to be executed

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: tuple containing
                observation, reward, done signal and infos
        """
        self._obs, rew, done, info = self._env.step(action)
        self._update_flag()
        obs = self._get_obs()
        rew = (np.sum(np.abs(obs['achieved_goal'] - obs['desired_goal'])) < 0.5).astype(np.float32)
        self.step_counter += 1
        return obs, rew, self.step_counter >= self.trial_length, info

    def _update_flag(self):
        lb, ub = 1.3, 2.3            
        flag = 0.
        if self._obs[0] < lb and self._obs[1] < lb:
            flag = 1.
        elif self._obs[0] < lb and self._obs[1] > ub:
            flag = 2.
        elif self._obs[0] > ub and self._obs[1] > ub:
            flag = 3.
        elif self._obs[0] > ub and self._obs[1] < lb:
            flag = 4.
        if flag and flag != self._history[-1]:
            self._history.append(flag)

    def sample_goal(
        self
    ) -> np.ndarray:
        """Samples a goal from the environment."""
        goal = [np.random.randint(4) + 1]
        for _ in range(self.n_goals-1):
            allowed = [float(i+1) for i in range(4) if i+1!=goal[-1]]
            goal.append(np.random.choice(allowed))
        return np.array(goal)

    def get_eval_cases(self):
        center = 1.8
        if self.random_reset:
            state = np.array([center, center] + [1., 2., 3., 4.][-self.n_goals:])
        else:
            state = np.array([center, center, 0., 0., 0., 0.])
        if self.n_goals == 2:
            goals = [np.array([1., 2.]), np.array([1., 3.]), np.array([1., 4.]),
                     np.array([2., 1.]), np.array([2., 3.]), np.array([2., 4.]),
                     np.array([3., 1.]), np.array([3., 2.]), np.array([3., 4.]),
                     np.array([4., 1.]), np.array([4., 2.]), np.array([4., 3.])]
        elif self.n_goals == 3:
            goals = [np.array([1., 2., 1.]), np.array([1., 3., 1.]), np.array([1., 4., 1.]),
                     np.array([2., 1., 2.]), np.array([2., 3., 2.]), np.array([2., 4., 2.]),
                     np.array([3., 1., 3.]), np.array([3., 2., 3.]), np.array([3., 4., 3.]),
                     np.array([4., 1., 4.]), np.array([4., 2., 4.]), np.array([4., 3., 4.])]
        else:
            raise NotImplementedError()
        return [(state, g) for g in goals]

    def render(self, mode='human'):
        if mode != 'human':
            return self._env.render(mode)
        viewer = self._env.env._get_viewer('human')
        positions = [[1.95, 1.95], [1.95, 4.05], [4.05, 4.05], [4.05, 1.95]]
        colors = ["0.2 0.2 0.8", "0.8 0.2 0.2", "0.8 0.8 0.2", "0.2 0.8 0.2"]
        for pos, col in zip(positions, colors):
            viewer.add_marker(
                pos=np.array(pos + [0.0]),
                size=np.array([0.45, 0.45, 0.1]),
                rgba=np.fromstring(col + " 0.4", dtype=np.float32, sep=" "),
                type=6,
                label="",
            )
        return self._env.render()

    def get_state(self):
        """Gets state."""
        return self.np_random.__getstate__()

    def set_state(self, state):
        """Sets state."""
        self.np_random.__setstate__(state)


class KitchenEnv(GCEnv):
    """Wraps D4RL kitchen in a goal-conditioned framework.

    Args:
        name (str): the name of the environment to create.
        frame_skip (int): number of simulation steps for environment step
        ik_controller (bool): enables inverse kinematic control
        max_episode_steps (int): timelimit
        random_init (bool): whether to set a random initial position for the arm
        compact_goal_space (bool): flag shrinking goal space to only include position
    """

    def __init__(
        self, name: str, frame_skip: int, ik_controller: bool, max_episode_steps: int,
        random_init: bool, compact_goal_space: bool = True
    ):
        from gymnasium_robotics.envs.franka_kitchen.franka_env import FrankaRobot
        from gymnasium_robotics.envs.franka_kitchen.kitchen_env import KitchenEnv
        FrankaRobot.metadata["render_fps"] = int(10 * 50 / frame_skip)
        KitchenEnv.metadata["render_fps"] = int(10 * 50 / frame_skip)
        self._env = gymnasium.make(name, render_mode='rgb_array', width=240, height=240,
                                   ik_controller=ik_controller, frame_skip=frame_skip)
        self._env._max_episode_steps = max_episode_steps
        self.random_init = random_init
        self.goal_size, self.obs_size = 17, 30
        self.goal_space = gym.spaces.Box(low=np.array([-np.inf]*self.goal_size, dtype=np.float32),
                                         high=np.array([np.inf]*self.goal_size, dtype=np.float32))
        self.observation_space = gym.spaces.Box(low=np.array([-np.inf]*self.obs_size, dtype=np.float32),
                                                high=np.array([np.inf]*self.obs_size, dtype=np.float32))
        self.OBS_ELEMENT_INDICES = {
                'bottom burner': np.array([11, 12]),
                'top burner': np.array([15, 16]),
                'light switch': np.array([17, 18]),
                'slide cabinet': np.array([19]),
                'hinge cabinet': np.array([20, 21]),
                'microwave': np.array([22]),
                'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
            }
        self.OBS_ELEMENT_GOALS = {
                'bottom burner': np.array([-0.88, -0.01]),
                'top burner': np.array([-0.92, -0.01]),
                'light switch': np.array([-0.69, -0.05]),
                'slide cabinet': np.array([0.37]),
                'hinge cabinet': np.array([0., 1.45]),
                'microwave': np.array([-0.75]),
                'kettle': np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06]),
            }
        self.GOAL_ELEMENT_INDICES = {
                'bottom burner': np.array([0, 1]),
                'top burner': np.array([2, 3]),
                'light switch': np.array([4, 5]),
                'slide cabinet': np.array([6]),
                'hinge cabinet': np.array([7, 8]),
                'microwave': np.array([9]),
                'kettle': np.array([10,11,12,13,14,15,16]),
            }

        self.goal_idxs = np.concatenate([v for v in self.OBS_ELEMENT_INDICES.values()])
        self.obs_idxs = np.concatenate([np.arange(0,9), np.arange(18,39)])

    def __getattr__(
        self, attr: str
    ) -> Any:
        """Gets attribute from wrapped environment.

        Args:
            attr (str): attribute name

        Returns:
            Any: requested attribute
        """
        return getattr(self._env, attr)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Returns observation and goal.

        Returns:
            Dict[str, np.ndarray]: normalized obs
        """
        return (dict(observation=self._obs['observation'][self.obs_idxs],
                     achieved_goal=(self._obs['observation'][self.obs_idxs])[self.goal_idxs],
                     desired_goal=self._goal))

    def set_goal(
        self, goal: np.ndarray
    ):
        """Sets goal; useful for precise evaluation.

        Args:
            goal (np.ndarray): environment goal
        """
        self._goal = goal
        return self._get_obs()

    def set_pos(
        self, pos: np.ndarray
    ):
        """Sets agent position; useful for precise evaluation.

        Args:
            pos (np.ndarray): agent position excuding velocities
        """
        if pos is not None:
            self._old_goal = self._goal.copy()
            qvel = self.robot_env.init_qvel
            ctrl = pos[:8].copy()
            ctrl[-1] = 255.
            self.robot_env.data.ctrl[:] = ctrl
            self.robot_env.set_state(pos, qvel)
            robot_obs = self.robot_env._get_obs()
            self._obs = self.unwrapped._get_obs(robot_obs)
            self._goal = self._old_goal.copy()
            self.render()
            return self._get_obs()

    def sample_goal(
        self
    ) -> np.ndarray:
        """Samples a goal from the environment."""
        goals = self.get_eval_cases()
        return self.get_eval_cases()[self.np_random.integers(len(goals))][1]

    def reset(self) -> np.ndarray:
        """Resets the environment.

        Returns:
            np.ndarray: current observation
        """
        self._obs = self._env.reset(seed=int(self.np_random.integers(1e5)))[0]
        starting_pos = self.get_starting_pos()
        self._obs = self._env.reset(seed=int(self.np_random.integers(1e5)))[0]
        self._goal = self.sample_goal()
        return self.set_pos(starting_pos)

    def get_starting_pos(self):
        starting_pos = self.robot_env.init_qpos.copy()
        if not self.random_init:
            return starting_pos
        lb = np.array([-0.55, 0.204, 2.14])
        ub = np.array([0.05, 0.530, 2.52])
        target_eef_pose = self.np_random.uniform(low=lb, high=ub)
        target_orientation = np.array([0,0.383,0.923,0])

        controller = IKController(self.robot_env.model, self.robot_env.data)
        ctrl_action = np.zeros(8)
        ctrl_action[-1] = self.robot_env.actuation_center[-1]
        for _ in range(50):
            delta_qpos = controller.compute_qpos_delta(target_eef_pose, target_orientation)
            ctrl_action[:7] = self.robot_env.data.ctrl.copy()[:7] + delta_qpos[:7]
            self.robot_env.data.ctrl[:] = ctrl_action
            mujoco.mj_step(self.robot_env.model, self.robot_env.data,
                           nstep=self.robot_env.frame_skip)
        robot_qpos, _ = robot_get_obs(self.robot_env.model, self.robot_env.data,
                                      self.robot_env.model_names.joint_names)
        starting_pos[:9] = robot_qpos
        return starting_pos

    def seed(self, seed):
        self.np_random = np.random.default_rng(seed=seed)

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Advances simulation by a step.

        Args:
            action (int): action to be executed

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: tuple containing
                observation, reward, done signal and infos
        """
        self._obs, _, _, done, _ = self._env.step(action)
        obs = self._get_obs()
        rew = 1.0 if all([np.linalg.norm(obs['observation'][o_idxs] - self._goal[g_idxs]) < 0.3 \
          for o_idxs, g_idxs in zip(self.OBS_ELEMENT_INDICES.values(),
                                    self.GOAL_ELEMENT_INDICES.values())]) else 0.0
        return obs, rew, done, {}

    def get_eval_cases(self):
        goals = []
        for id, g in zip(self.OBS_ELEMENT_INDICES.values(), self.OBS_ELEMENT_GOALS.values()):
            goal = self.robot_env.init_qpos.copy()
            goal[id] = g
            goals.append(goal[self.goal_idxs])
        return [(self.robot_env.init_qpos.copy(), g) for g in goals]

    def get_state(self):
        """Gets state."""
        return self.np_random.__getstate__()

    def set_state(self, state):
        """Sets state."""
        self.np_random.__setstate__(state)

    def render(self, mode='rgb_array'):
        return self.robot_env.render()
