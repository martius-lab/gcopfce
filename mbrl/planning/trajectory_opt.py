import time
from typing import Callable, List, Optional, Sequence, cast
import hydra
import numpy as np
import omegaconf
import torch
import torch.distributions
import mbrl.models
import mbrl.types
import mbrl.util.math

from .core import Agent, complete_agent_cfg


class Optimizer:
    def __init__(self):
        pass

    def optimize(
        self,
        obj_fun: Callable[[torch.Tensor], torch.Tensor],
        x0: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Runs optimization.

        Args:
            obj_fun (callable(tensor) -> tensor): objective function to maximize.
            x0 (tensor, optional): initial solution, if necessary.

        Returns:
            (torch.Tensor): the best solution found.
        """
        pass


class ICEMOptimizer(Optimizer):
    """Implements the Improved Cross-Entropy Method (iCEM) optimization algorithm.

    iCEM improves the sample efficiency over standard CEM and was introduced by
    [2] for real-time planning.

    Args:
        num_iterations (int): the number of iterations (generations) to perform.
        elite_ratio (float): the proportion of the population that will be kept as
            elite (rounds up).
        population_size (int): the size of the population.
        population_decay_factor (float): fixed factor for exponential decrease in population size
        colored_noise_exponent (float): colored-noise scaling exponent for generating correlated
            action sequences.
        lower_bound (sequence of floats): the lower bound for the optimization variables.
        upper_bound (sequence of floats): the upper bound for the optimization variables.
        keep_elite_frac (float): the fraction of elites to keep (or shift) during CEM iterations
        alpha (float): momentum term.
        device (torch.device): device where computations will be performed.
        return_mean_elites (bool): if ``True`` returns the mean of the elites of the last
            iteration. Otherwise, it returns the max solution found over all iterations.
        population_size_module (int, optional): if specified, the population is rounded to be
            a multiple of this number. Defaults to ``None``.

    [2] C. Pinneri, S. Sawant, S. Blaes, J. Achterhold, J. Stueckler, M. Rolinek and
    G, Martius, Georg. "Sample-efficient Cross-Entropy Method for Real-time Planning".
    Conference on Robot Learning, 2020.
    """

    def __init__(
        self,
        num_iterations: int,
        elite_ratio: float,
        population_size: int,
        population_decay_factor: float,
        colored_noise_exponent: float,
        lower_bound: Sequence[Sequence[float]],
        upper_bound: Sequence[Sequence[float]],
        keep_elite_frac: float,
        alpha: float,
        device: torch.device,
        return_mean_elites: bool = False,
        population_size_module: Optional[int] = None,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.elite_ratio = elite_ratio
        self.population_size = population_size
        self.population_decay_factor = population_decay_factor
        self.elite_num = np.ceil(self.population_size * self.elite_ratio).astype(
            np.int32
        )
        self.colored_noise_exponent = colored_noise_exponent
        self.lower_bound = torch.tensor(lower_bound, device=device, dtype=torch.float32)
        self.upper_bound = torch.tensor(upper_bound, device=device, dtype=torch.float32)
        self.initial_var = ((self.upper_bound - self.lower_bound) ** 2) / 16
        self.keep_elite_frac = keep_elite_frac
        self.keep_elite_size = np.ceil(keep_elite_frac * self.elite_num).astype(
            np.int32
        )
        self.elite = None
        self.alpha = alpha
        self.return_mean_elites = return_mean_elites
        self.population_size_module = population_size_module
        self.device = device

        if self.population_size_module:
            self.keep_elite_size = self._round_up_to_module(
                self.keep_elite_size, self.population_size_module
            )
        self._rng = torch.Generator(device=self.device)

    @staticmethod
    def _round_up_to_module(value: int, module: int) -> int:
        if value % module == 0:
            return value
        return value + (module - value % module)

    def optimize(
        self,
        obj_fun: Callable[[torch.Tensor], torch.Tensor],
        x0: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[torch.Tensor, torch.Tensor, int], None]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Runs the optimization using iCEM.

        Args:
            obj_fun (callable(tensor) -> tensor): objective function to maximize.
            x0 (tensor, optional): initial mean for the population. Must
                be consistent with lower/upper bounds.
            callback (callable(tensor, tensor, int) -> any, optional): if given, this
                function will be called after every iteration, passing it as input the full
                population tensor, its corresponding objective function values, and
                the index of the current iteration. This can be used for logging and plotting
                purposes.
        Returns:
            (torch.Tensor): the best solution found.
        """
        mu = x0.clone()
        var = self.initial_var.clone()

        best_solution = torch.empty_like(mu)
        best_value = -np.inf

        for i in range(self.num_iterations):
            decay_population_size = np.ceil(
                np.max(
                    (
                        self.population_size * self.population_decay_factor**-i,
                        2 * self.elite_num,
                    )
                )
            ).astype(np.int32)

            if self.population_size_module:
                decay_population_size = self._round_up_to_module(
                    decay_population_size, self.population_size_module
                )
            # the last dimension is used for temporal correlations
            population = mbrl.util.math.powerlaw_psd_gaussian(
                self.colored_noise_exponent,
                size=(decay_population_size, x0.shape[1], x0.shape[0]),
                device=self.device, generator=self._rng
            ).transpose(1, 2)
            population = torch.minimum(
                population * torch.sqrt(var) + mu, self.upper_bound
            )
            population = torch.maximum(population, self.lower_bound)
            if self.elite is not None:
                kept_elites = torch.index_select(
                    self.elite,
                    dim=0,
                    index=torch.randperm(self.elite_num, device=self.device, generator=self._rng)[
                        : self.keep_elite_size
                    ],
                )
                if i == 0:
                    end_action = (
                        torch.normal(
                            mu[-1, :].repeat(kept_elites.shape[0], 1),
                            torch.sqrt(var[-1, :]).repeat(kept_elites.shape[0], 1),
                        )
                        .unsqueeze(1)
                        .to(self.device)
                    )
                    kept_elites_shifted = torch.cat(
                        (kept_elites[:, 1:, :], end_action), dim=1
                    )
                    population = torch.cat((population, kept_elites_shifted), dim=0)
                elif i == self.num_iterations - 1:
                    population = torch.cat((population, mu.unsqueeze(dim=0)), dim=0)
                else:
                    population = torch.cat((population, kept_elites), dim=0)

            values, trajectories, rewards = obj_fun(population)

            # filter out NaN values
            values[values.isnan()] = -1e-10
            best_values, elite_idx = values.topk(self.elite_num)
            self.elite = population[elite_idx]

            new_mu = torch.mean(self.elite, dim=0)
            new_var = torch.var(self.elite, unbiased=False, dim=0)
            mu = self.alpha * mu + (1 - self.alpha) * new_mu
            var = self.alpha * var + (1 - self.alpha) * new_var

            if best_values[0] > best_value:
                best_value = best_values[0]
                best_solution = population[elite_idx[0]].clone()

        if callback is not None:
            callback(population, values, trajectories, rewards)

        return mu if self.return_mean_elites else best_solution

    def seed(self, seed):
        self._rng.manual_seed(seed)

class TrajectoryOptimizer:
    """Class for using generic optimizers on trajectory optimization problems.

    This is a convenience class that sets up optimization problem for trajectories, given only
    action bounds and the length of the horizon. Using this class, the concern of handling
    appropriate tensor shapes for the optimization problem is hidden from the users, which only
    need to provide a function that is capable of evaluating trajectories of actions. It also
    takes care of shifting previous solution for the next optimization call, if the user desires.

    The optimization variables for the problem will have shape ``H x A``, where ``H`` and ``A``
    represent planning horizon and action dimension, respectively. The initial solution for the
    optimizer will be computed as (action_ub - action_lb) / 2, for each time step.

    Args:
        optimizer_cfg (omegaconf.DictConfig): the configuration of the optimizer to use.
        action_lb (np.ndarray): the lower bound for actions.
        action_ub (np.ndarray): the upper bound for actions.
        planning_horizon (int): the length of the trajectories that will be optimized.
        replan_freq (int): the frequency of re-planning. This is used for shifting the previous
        solution for the next time step, when ``keep_last_solution == True``. Defaults to 1.
        keep_last_solution (bool): if ``True``, the last solution found by a call to
            :meth:`optimize` is kept as the initial solution for the next step. This solution is
            shifted ``replan_freq`` time steps, and the new entries are filled using the initial
            solution. Defaults to ``True``.
    """

    def __init__(
        self,
        optimizer_cfg: omegaconf.DictConfig,
        action_lb: np.ndarray,
        action_ub: np.ndarray,
        planning_horizon: int,
        replan_freq: int = 1,
        keep_last_solution: bool = True,
    ):
        optimizer_cfg.lower_bound = np.tile(action_lb, (planning_horizon, 1)).tolist()
        optimizer_cfg.upper_bound = np.tile(action_ub, (planning_horizon, 1)).tolist()
        self.optimizer: Optimizer = hydra.utils.instantiate(optimizer_cfg)
        self.initial_solution = (
            ((torch.tensor(action_lb) + torch.tensor(action_ub)) / 2)
            .float()
            .to(optimizer_cfg.device)
        )
        self.initial_solution = self.initial_solution.repeat((planning_horizon, 1))
        self.previous_solution = self.initial_solution.clone()
        self.replan_freq = replan_freq
        self.keep_last_solution = keep_last_solution
        self.horizon = planning_horizon

    def optimize(
        self,
        trajectory_eval_fn: Callable[[torch.Tensor], torch.Tensor],
        callback: Optional[Callable] = None,
    ) -> np.ndarray:
        """Runs the trajectory optimization.

        Args:
            trajectory_eval_fn (callable(tensor) -> tensor): A function that receives a batch
                of action sequences and returns a batch of objective function values (e.g.,
                accumulated reward for each sequence). The shape of the action sequence tensor
                will be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size,
                planning horizon, and action dimension, respectively. Additionally, returns a list
                of rewards and observations.
            callback (callable, optional): a callback function
                to pass to the optimizer.

        Returns:
            (tuple of np.ndarray and float): the best action sequence.
        """
        best_solution = self.optimizer.optimize(
            trajectory_eval_fn,
            x0=self.previous_solution,
            callback=callback,
        )
        if self.keep_last_solution:
            self.previous_solution = best_solution.roll(-self.replan_freq, dims=0)
            # Note that initial_solution[i] is the same for all values of [i],
            # so just pick i = 0
            self.previous_solution[-self.replan_freq :] = self.initial_solution[0]
        return best_solution.cpu().numpy()

    def reset(self):
        """Resets the previous solution cache to the initial solution."""
        self.previous_solution = self.initial_solution.clone()


class TrajectoryOptimizerAgent(Agent):
    """Agent that performs trajectory optimization on a given objective function for each action.

    This class uses an internal :class:`TrajectoryOptimizer` object to generate
    sequence of actions, given a user-defined trajectory optimization function.

    Args:
        optimizer_cfg (omegaconf.DictConfig): the configuration of the base optimizer to pass to
            the trajectory optimizer.
        action_lb (sequence of floats): the lower bound of the action space.
        action_ub (sequence of floats): the upper bound of the action space.
        planning_horizon (int): the length of action sequences to evaluate. Defaults to 1.
        replan_freq (int): the frequency of re-planning. The agent will keep a cache of the
            generated sequences an use it for ``replan_freq`` number of :meth:`act` calls.
            Defaults to 1.
        verbose (bool): if ``True``, prints the planning time on the console.
        keep_last_solution (bool): if ``True``, the last solution found by a call to
            :meth:`optimize` is kept as the initial solution for the next step. This solution is
            shifted ``replan_freq`` time steps, and the new entries are filled using the initial
            solution. Defaults to ``True``.

    Note:
        After constructing an agent of this type, the user must call
        :meth:`set_trajectory_eval_fn`. This is not passed to the constructor so that the agent can
        be automatically instantiated with Hydra (which in turn makes it easy to replace this
        agent with an agent of another type via config-only changes).
    """

    def __init__(
        self,
        optimizer_cfg: omegaconf.DictConfig,
        action_lb: Sequence[float],
        action_ub: Sequence[float],
        planning_horizon: int = 1,
        replan_freq: int = 1,
        verbose: bool = False,
        keep_last_solution: bool = True,
    ):
        self.optimizer = TrajectoryOptimizer(
            optimizer_cfg,
            np.array(action_lb),
            np.array(action_ub),
            planning_horizon=planning_horizon,
            replan_freq=replan_freq,
            keep_last_solution=keep_last_solution,
        )
        self.optimizer_args = {
            "optimizer_cfg": optimizer_cfg,
            "action_lb": np.array(action_lb),
            "action_ub": np.array(action_ub),
        }
        self.trajectory_eval_fn: mbrl.types.TrajectoryEvalFnType = None
        self.actions_to_use: List[np.ndarray] = []
        self.replan_freq = replan_freq
        self.planning_horizon = planning_horizon
        self.verbose = verbose

    def set_trajectory_eval_fn(
        self, trajectory_eval_fn: mbrl.types.TrajectoryEvalFnType
    ):
        """Sets the trajectory evaluation function.

        Args:
            trajectory_eval_fn (callable): a trajectory evaluation function, as described in
                :class:`TrajectoryOptimizer`.
        """
        self.trajectory_eval_fn = trajectory_eval_fn

    def reset(self, planning_horizon: Optional[int] = None):
        """Resets the underlying trajectory optimizer."""
        # always reset, as iCEM needs to reset the elite set
        planning_horizon = planning_horizon or self.planning_horizon
        self.optimizer = TrajectoryOptimizer(
            cast(omegaconf.DictConfig, self.optimizer_args["optimizer_cfg"]),
            cast(np.ndarray, self.optimizer_args["action_lb"]),
            cast(np.ndarray, self.optimizer_args["action_ub"]),
            planning_horizon=planning_horizon,
            replan_freq=self.replan_freq,
        )

    def act(
        self, obs: np.ndarray, optimizer_callback: Optional[Callable] = None, goal: Optional[np.ndarray] = None,
        initialization: Optional[np.ndarray] = None, open_loop_sequence: Optional[np.ndarray] = None, **_kwargs
    ) -> np.ndarray:
        """Issues an action given an observation.

        This method optimizes a full sequence of length ``self.planning_horizon`` and returns
        the first action in the sequence. If ``self.replan_freq > 1``, future calls will use
        subsequent actions in the sequence, for ``self.replan_freq`` number of steps.
        After that, the method will plan again, and repeat this process.

        Args:
            obs (np.ndarray): the observation for which the action is needed.
            optimizer_callback (callable, optional): a callback function
                to pass to the optimizer.
            goal (np.ndarray, optional): a goal, in case of goal-conditioned planning.

        Returns:
            (np.ndarray): the action.
        """
        if self.trajectory_eval_fn is None:
            raise RuntimeError(
                "Please call `set_trajectory_eval_fn()` before using TrajectoryOptimizerAgent"
            )
        plan_time = 0.0

        if open_loop_sequence is not None:
            self.actions_to_use.extend(open_loop_sequence)

        if not self.actions_to_use:  # re-plan is necessary

            def trajectory_eval_fn(action_sequences):
                return self.trajectory_eval_fn(obs, action_sequences, goal)

            if initialization is not None:
                self.optimizer.previous_solution = initialization

            start_time = time.time()
            plan = self.optimizer.optimize(
                trajectory_eval_fn, callback=optimizer_callback
            )
            plan_time = time.time() - start_time

            self.actions_to_use.extend([a for a in plan[: self.replan_freq]])
        action = self.actions_to_use.pop(0)

        if self.verbose:
            print(f"Planning time: {plan_time:.3f}")
        return action

    def plan(self, obs: np.ndarray, optimizer_callback: Optional[Callable] = None, 
             goal: Optional[np.ndarray] = None, **_kwargs) -> np.ndarray:
        """Issues a sequence of actions given an observation.

        Returns s sequence of length self.planning_horizon.

        Args:
            obs (np.ndarray): the observation for which the sequence is needed.

        Returns:
            (np.ndarray): a sequence of actions.
        """
        if self.trajectory_eval_fn is None:
            raise RuntimeError(
                "Please call `set_trajectory_eval_fn()` before using TrajectoryOptimizerAgent"
            )

        def trajectory_eval_fn(action_sequences):
            return self.trajectory_eval_fn(obs, action_sequences, goal)

        plan = self.optimizer.optimize(trajectory_eval_fn, callback=optimizer_callback)
        return plan

    def seed(self, seed):
        self.optimizer.optimizer.seed(seed)

    def get_state(self):
        return dict(rng_state=self.optimizer.optimizer._rng.get_state())

    def set_state(self, state):
        self.optimizer.optimizer._rng.set_state(state['rng_state'])


def create_trajectory_optim_agent_for_model(
    model_env: mbrl.models.ModelEnv,
    agent_cfg: omegaconf.DictConfig,
    num_particles: int = 1,
    cost_discount: float = 1.0,
    cost_aggregation: str = 'sum',
    seed = None,
) -> TrajectoryOptimizerAgent:
    """Utility function for creating a trajectory optimizer agent for a model environment.

    This is a convenience function for creating a :class:`TrajectoryOptimizerAgent`,
    using :meth:`mbrl.models.ModelEnv.evaluate_action_sequences` as its objective function.


    Args:
        model_env (mbrl.models.ModelEnv): the model environment.
        agent_cfg (omegaconf.DictConfig): the agent's configuration.
        num_particles (int): the number of particles for taking averages of action sequences'
            total rewards.

    Returns:
        (:class:`TrajectoryOptimizerAgent`): the agent.

    """
    complete_agent_cfg(model_env, agent_cfg)
    agent = hydra.utils.instantiate(agent_cfg)

    def trajectory_eval_fn(initial_state, action_sequences, goal):
        return model_env.evaluate_action_sequences(
            action_sequences, initial_state=initial_state, goal=goal, num_particles=num_particles,
            discount=cost_discount, aggregation=cost_aggregation
        )

    agent.set_trajectory_eval_fn(trajectory_eval_fn)
    # TODO: disentangle this seed with the others
    agent.seed(seed)
    return agent
