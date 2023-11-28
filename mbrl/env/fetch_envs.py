import gymnasium as gym
from typing import Any, Dict, Tuple
import numpy as np
from mbrl.env.d4rl_envs import GCEnv


class FetchEnv(GCEnv):
    """Wraps fetch environments in a goal-conditioned framework.

    Args:
        name (str): the name of the environment to create.
        compact_goal_space (bool): flag shrinking goal space to only include position
    """

    def __init__(
        self, name: str, compact_goal_space: bool,
    ):
        self.name = name
        self._env = gym.make(name, render_mode='rgb_array')
        self.goal_space = self.observation_space['observation' if not compact_goal_space else 'desired_goal']
        self.observation_space = self.observation_space['observation']
        self.compact_goal_space = compact_goal_space


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
        if self.compact_goal_space:
            return self._obs
        achieved_goal = self._obs['observation']
        desired_goal = self._obs['observation']
        if 'reach' in self.name:
            desired_goal[:3] = self._obs['desired_goal']
            desired_goal[5:] = 0
        else:
            desired_goal[3:6] = self._obs['desired_goal']
            desired_goal[6:9] = desired_goal[3:6] - desired_goal[:3]
            desired_goal[11:14] = 0.
            desired_goal[14:17] = - desired_goal[20:23]
            desired_goal[17:20] = 0.
        return {
            'observation': self._obs['observation'],
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal
        }
    
    def set_goal(
        self, goal: np.ndarray
    ):
        """Sets goal; useful for precise evaluation.

        Args:
            goal (np.ndarray): environment goal
        """
        self._env.env.env.env.goal = goal.copy()
        self._obs = self._env.env.env.env._get_obs()
        return self._get_obs()

    def set_pos(
        self, pos: np.ndarray
    ):
        """Sets agent position; useful for precise evaluation.

        Args:
            goal (np.ndarray): agent position excuding velocities
        """
        if pos is not None:
            object_qpos = self._env.env.env.env._utils.get_joint_qpos(self.model, self.data, "object0:joint")
            assert object_qpos.shape == (7,)
            object_qpos[:2] = pos
            self._env.env.env.env._utils.set_joint_qpos(self.model, self.data, "object0:joint", object_qpos)
        if 'reach' in self.name.lower():
            self._env.env.env.env.sim.forward()
        else:
            self._env.env.env.env._mujoco.mj_forward(self.model, self.data)
        self._obs = self._env.env.env.env._get_obs()
        return self._get_obs()

    def reset(self) -> np.ndarray:
        """Resets the environment.

        Returns:
            np.ndarray: current observation
        """
        self._env.reset()

        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
            -self.target_range, self.target_range, size=3
        )
        if self.has_object:
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)

        state = None
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                    -self.obj_range, self.obj_range, size=2
                )
            state = object_xpos
        self.set_pos(state)
        self.set_goal(goal)
        self._obs = self._env.env.env.env._get_obs()
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
        self._obs, rew, terminated, truncated, info = self._env.step(action)
        info = {'distance': np.linalg.norm(self._obs['achieved_goal'] - self._obs['desired_goal'])}
        # set rewards to 0-1 interval
        return self._get_obs(), rew + 1., terminated or truncated, info

    def sample_goal(
        self
    ) -> np.ndarray:
        """Samples a goal from the environment."""
        return self._env.env.env.env._sample_goal()

    def get_eval_cases(self):
        """ We only evaluate on 24 fixed configurations."""
        ec_rand = np.random.default_rng(0)
        n_cases = 24
        cases = []
        for _ in range(n_cases):
            goal = self.initial_gripper_xpos[:3] + ec_rand.uniform(
                -self.target_range, self.target_range, size=3
            )
            if self.has_object:
                goal += self.target_offset
                goal[2] = self.height_offset
                if self.target_in_the_air and ec_rand.uniform() < 0.5:
                    goal[2] += ec_rand.uniform(0, 0.45)

            state = None
            if self.has_object:
                object_xpos = self.initial_gripper_xpos[:2]
                while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                    object_xpos = self.initial_gripper_xpos[:2] + ec_rand.uniform(
                        -self.obj_range, self.obj_range, size=2
                    )
                state = object_xpos

            cases.append((state.copy() if state is not None else state, goal.copy()))
        return cases

    def seed(self, seed):
        self.np_random = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))

    def get_state(self):
        """Gets state."""
        return self.np_random.__getstate__()

    def set_state(self, state):
        """Sets state."""
        self.np_random.__setstate__(state)

    def render(self, mode):
        return self._env.render()
