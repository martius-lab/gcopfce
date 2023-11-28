import pathlib
from copy import deepcopy
import itertools
from functools import partial
from typing import Tuple, Union, Optional, Callable, cast
import networkx as nx
import numpy as np
import torch
import mbrl
from mbrl.models.util import MLPActorCritic
from mbrl.types import ModelInput, TransitionBatch
from mbrl.util.replay_buffer import ReplayBuffer, MOPOIterator, GoalRelabelingIterator
from mbrl.models.util import to_tensor
from mbrl.modules.core import Module
from mbrl.models import util as model_util
import matplotlib.pyplot as plt


class DistanceModule(Module):
    """Module computing a reward signal as distance between state pairs.
    Relies on the method proposed in MBOLD.
    """

    name = "distance"
    EVAL_LOG_FORMAT = [("inv_rank", "I", "float"),("hits", "H", "float")]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        attrs = ['gamma', 'units', 'lr', 'polyak', 'deterministic_actor',
                 'alpha', 'p_norm', 'target_noise',
                 'target_noise_clip', 'policy_delay', 'relabeling',
                 'p_geometric', 'p_rand_goals', 'n_ensemble',
                 'enable_mbpo', 'mbpo_num_updates_before_rollout', 'squash_policy',
                 'mbpo_accumulate', 'mbpo_num_rollouts', 'mbpo_rollout_horizon',
                 'mbpo_buffer_capacity', 'mbpo_lambda', 'mbpo_batch_size',
                 'enable_crr', 'crr_beta', 'crr_n_action_samples',
                 'crr_advantage_type', 'crr_weight_type', 'crr_max_weight',
                 'obs_dim', 'act_dim', 'goal_dim', 'act_limit']
        [setattr(self, k , kwargs[k]) for k in attrs]
        if self.enable_crr and self.deterministic_actor:
            print('CRR needs a stochastic actor. Overriding.')
            self.deterministic_actor = False
        self.sigmoid = True
        self.hidden_sizes = [self.units] * 2
        self.update_count = 0
        self.mbpo_buffer = None

    def init(self):
        """Initializes the module."""
        self.ac = torch.nn.ModuleList([MLPActorCritic(self.obs_dim, self.act_dim, self.goal_dim,
                                  self.act_limit, self.sigmoid,
                                  self.hidden_sizes, p_norm=self.p_norm,
                                  squash_policy = self.squash_policy).to(self.device) for _ in range(self.n_ensemble)])
        self.ac_targ = torch.nn.ModuleList([deepcopy(ac) for ac in self.ac])
        for ac in self.ac_targ:
            for p in ac.parameters():
                p.requires_grad = False
        self.pi_optimizer = [torch.optim.Adam(ac.pi.parameters(), lr=self.lr) for ac in self.ac]
        self.q_params = [itertools.chain(ac.q1.parameters(), ac.q2.parameters()) for ac in self.ac]
        self.q_optimizer = [torch.optim.Adam(q, lr=self.lr) for q in self.q_params]
        super().init()

    def _compute_loss_q(
        self,
        batch: TransitionBatch,
        id: int
        ) -> torch.Tensor:
        """Computes action-value loss.

        Args:
            model_in (TransitionBatch): training batch
            id (int): id of ensemble member to train

        Returns:
            torch.Tensor: action-value loss
        """

        o, a, o2, r, d, g = batch

        q1 = self.ac[id].q1(o,a,g)
        q2 = self.ac[id].q2(o,a,g)

        with torch.no_grad():
            a2, logp_a2 = self.ac[id].pi(o2, g, deterministic=self.deterministic_actor)

            if self.target_noise * self.target_noise_clip > 0:
                epsilon = torch.randn_like(a2) * self.target_noise
                epsilon = torch.clamp(epsilon, -self.target_noise_clip, self.target_noise_clip)
                a2 = a2 + epsilon
                a2 = torch.clamp(a2, -self.act_limit, self.act_limit)

            q1_pi_targ = self.ac_targ[id].q1(o2, a2, g)
            q2_pi_targ = self.ac_targ[id].q2(o2, a2, g)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + (1-d) * self.gamma * (q_pi_targ - self.alpha * logp_a2)

        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        q_info = dict(Q1Vals=q1, Q2Vals=q2, QLoss=loss_q)
        return loss_q, q_info 

    def _compute_loss_pi(
        self,
        batch: ModelInput,
        id: int
        ) -> torch.Tensor:
        """Computes policy loss.

        Args:
            model_in (ModelInput): training batch
            id (int): id of ensemble member to train

        Returns:
            torch.Tensor: policy loss
        """

        o, a, _, _, _, g = batch

        if self.enable_crr:
            log_probs = self.ac[id].pi.get_log_probs(o, a, g)

            with torch.no_grad():
                obs = torch.tile(o, (self.crr_n_action_samples, 1))
                goal = torch.tile(g, (self.crr_n_action_samples, 1))
                pi, _ = self.ac[id].pi(obs, goal, deterministic=False)
                
                q1_pi = self.ac[id].q1(obs, pi, goal)
                q2_pi = self.ac[id].q2(obs, pi, goal)
                values = torch.min(q1_pi, q2_pi)
                values = values.reshape(self.crr_n_action_samples, -1, 1)
                if self.crr_advantage_type == "mean":
                    values = values.mean(dim=0)
                elif self.crr_advantage_type == "max":
                    values = values.max(dim=0).values
                else:
                    raise ValueError(f"Invalid advantage type: {self.crr_advantage_type}.")
                q1_pi = self.ac[id].q1(o, a, g)
                q2_pi = self.ac[id].q2(o, a, g)
                advantages = torch.min(q1_pi, q2_pi) - values

            if self.crr_weight_type == "binary":
                weight = (advantages > 0.0).float()
            elif self.crr_weight_type == "exp":
                weight = (advantages / self.crr_beta).exp().clamp(0.0, self.crr_max_weight)
            else:
                raise ValueError(f"Invalid weight type: {self.weight_type}.")
            loss_pi = -(log_probs * weight).mean()
            pi_info = dict()
        else:
            pi, logp_pi = self.ac[id].pi(o, g, deterministic=self.deterministic_actor)
            q1_pi = self.ac[id].q1(o, pi, g)
            q2_pi = self.ac[id].q2(o, pi, g)
            q_pi = torch.min(q1_pi, q2_pi)
            loss_pi = (self.alpha * logp_pi - q_pi).mean()
            pi_info = dict(LogPi=logp_pi)

        return loss_pi, pi_info

    def update(
        self,
        batch: ModelInput
    ) -> float:
        """ Trains the distance module.

        Args:
            model_in (tensor or batch of transitions): the inputs to the model.

        Returns:
            float: loss for logging
        """

        if self.enable_mbpo:
            if not (self.update_count % self.mbpo_num_updates_before_rollout):
                self.add_rollouts_to_buffer()
            batch = self.sample_batch_from_buffer()

        batch = self.get_module_input(batch)
        logs = []
        for id in range(self.n_ensemble):

            self.q_optimizer[id].zero_grad()
            loss_q, log_q = self._compute_loss_q(batch, id)
            loss_q.backward()
            self.q_optimizer[id].step()

            # possibly update pi and target networks
            log_pi = {}
            if not self.update_count % self.policy_delay:
                for p in self.q_params[id]:
                    p.requires_grad = False
                self.pi_optimizer[id].zero_grad()
                loss_pi, log_pi = self._compute_loss_pi(batch, id)
                loss_pi.backward()
                self.pi_optimizer[id].step()

                for p in self.q_params[id]:
                    p.requires_grad = True
                with torch.no_grad(): 
                    for p, p_targ in zip(self.ac[id].parameters(), self.ac_targ[id].parameters()):
                        p_targ.data.mul_(self.polyak)
                        p_targ.data.add_((1 - self.polyak) * p.data)

            logs.append({**log_q, **log_pi})
            self.update_count += 1
        return np.mean([l['QLoss'].cpu().detach().numpy() for l in logs])

    def reward_fn(
        self,
        obs: torch.Tensor, 
        act: torch.Tensor,
        next_obs: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        elite_idxs: Optional[list] = None
    ) -> torch.Tensor:
        """Reward computed through distances.

        Args:
            obs (torch.Tensor): observation preceding the action
            act (torch.Tensor): action chosen by the agent
            next_obs (torch.Tensor): observation following the action
            goal (torch.Tensor, optional): goal for goal-conditioned planning

        Returns:
            torch.Tensor: reward scalar
        """
        with torch.no_grad():

            if len(goal.shape) < len(next_obs.shape):
                goal = torch.stack([goal] * next_obs.shape[0], 0)
            next_obs, goal = self.normalize_obs(next_obs.to(self.device)), self.normalize_goal(goal.to(self.device))
            distances = []

            elite_idxs = [i for i in range(self.n_ensemble)] if elite_idxs is None else elite_idxs
            for elite_idx in elite_idxs:
                pi, _ = self.ac[elite_idx].pi(next_obs, goal, deterministic=True)
                q1_pi = self.ac[elite_idx].q1(next_obs, pi, goal)
                q2_pi = self.ac[elite_idx].q2(next_obs, pi, goal)
                q_pi = torch.min(q1_pi, q2_pi).reshape(-1, 1)
                distance = - q_pi
                distances.append(distance)

            distance = torch.stack(distances).mean(0)
        return - distance

    def get_iter(
        self,
    ) -> Callable:
        return partial(GoalRelabelingIterator,
                       relabeling=self.relabeling,
                       p_geometric=self.p_geometric,
                       p_rand_goals=self.p_rand_goals,
                       )
    
    def log(        
        self,
        trial_n: int,
        offline_epochs: Optional[int] = None,
    ):
        with torch.no_grad():
            batch = self.replay_buffer.get_all()
            seen_obs, goal = batch.obs, batch.desired_goals
            seen_obs = torch.from_numpy(seen_obs).float().to(self.device)
            goal = self.replay_buffer.goal_proj_fn(seen_obs)[0]
            goal = torch.stack([goal]*seen_obs.shape[0], 0)
            rewards = self.reward_fn(None, None, seen_obs, goal)
            xs = self.replay_buffer.goal_proj_fn(seen_obs)[:, 0].cpu().detach().numpy()
            ys = self.replay_buffer.goal_proj_fn(seen_obs)[:, 1].cpu().detach().numpy()
            gx = goal.cpu().detach().numpy()[0, 0]
            gy = goal.cpu().detach().numpy()[0, 1]

            fig = plt.figure()
            ax = fig.add_subplot(111)
            im = ax.scatter(xs, ys, c=rewards.cpu().detach().numpy())
            fig.colorbar(im)
            ax.scatter(gx, gy, s=200, marker='x', color='r')
            plt.savefig(pathlib.Path(self.work_dir) / (f'{self.name}_{trial_n}.png'))
            plt.close()

            reward_idxs = torch.argsort(rewards.flatten(), descending=True)
            goal = self.replay_buffer.goal_proj_fn(seen_obs)[0]
            goal = torch.stack([goal]*seen_obs.shape[0], 0)
            relabel_idxs = self.relabeling_fn(None, None, seen_obs, goal).flatten()
            inv_rank = 1/(torch.arange(rewards.shape[0]).to(self.device)[relabel_idxs[reward_idxs].bool()]+1)
            hits = relabel_idxs[reward_idxs[:100]].sum().item()
            if self.logger is not None:
                self.logger.log_data(self.name, {'inv_rank': inv_rank.sum().item(), 'hits': hits})

    def load(self, load_dir: Union[str, pathlib.Path]):
        """Loads the model from the given path."""
        for i in range(self.n_ensemble):
            if load_dir and (pathlib.Path(load_dir) / (self.name+f'_pi_opt_{i}.pth')).exists():
                self.pi_optimizer[i].load_state_dict(torch.load(pathlib.Path(load_dir) / (self.name+f'_pi_opt_{i}.pth')))
            if load_dir and (pathlib.Path(load_dir) / (self.name+f'_q_opt_{i}.pth')).exists():
                state = torch.load(pathlib.Path(load_dir) / (self.name+f'_q_opt_{i}.pth'))
                if state['param_groups'][0]['params'] != self.q_optimizer[i].state_dict()['param_groups'][0]['params']:
                    state['param_groups'][0]['params'] = self.q_optimizer[i].state_dict()['param_groups'][0]['params']
                    state['state'] = {}
                    print('Resuming from legacy checkpoint. Optimizer is not restored.')
                self.q_optimizer[i].load_state_dict(state)
        if (pathlib.Path(load_dir) / (self.name+f'_mbpo_buffer.pth')).exists():
            self.mbpo_buffer = ReplayBuffer(self.mbpo_buffer_capacity,
                        self.model_env.observation_space.shape,
                        self.model_env.action_space.shape,
                        self.model_env.goal_space.shape,
                        max_trajectory_length=self.mbpo_rollout_horizon,
                        rng=self.replay_buffer.rng,
                        goal_proj_fn=self.replay_buffer.goal_proj_fn)
            self.mbpo_buffer.set_state(torch.load(pathlib.Path(load_dir) / (self.name+f'_mbpo_buffer.pth')))
        super().load(load_dir)

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves the model to the given directory. Skip model attribute."""
        for i in range(self.n_ensemble):
            torch.save(self.pi_optimizer[i].state_dict(), pathlib.Path(save_dir) / (self.name+f'_pi_opt_{i}.pth'))
            torch.save(self.q_optimizer[i].state_dict(), pathlib.Path(save_dir) / (self.name+f'_q_opt_{i}.pth'))
        # TODO: ensure determinism
        if self.mbpo_buffer:
            torch.save(self.mbpo_buffer.get_state(), pathlib.Path(save_dir) / (self.name+f'_mbpo_buffer.pth'))
        super().save(save_dir)

    @torch.no_grad()
    def act(self, obs: np.ndarray, goal: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues an action given an observation.

        Args:
            obs (np.ndarray): the observation for which the action is needed.

        Returns:
            (np.ndarray): the action.
        """
        obs = self.normalize_obs(model_util.to_tensor(obs).to(self.device).float().unsqueeze(0))
        goal = self.normalize_goal(model_util.to_tensor(goal).to(self.device).float().unsqueeze(0))
        pi, _ = self.ac[0].pi(obs, goal, deterministic=True)
        if not self.enable_crr:
            return pi.squeeze().cpu().detach().numpy()
        
        obs = torch.tile(obs, (self.crr_n_action_samples, 1))
        goal = torch.tile(goal, (self.crr_n_action_samples, 1))
        pi, _ = self.ac[0].pi(obs, goal, deterministic=False)
        # no target, as in d3rlpy
        q1_pi = self.ac[0].q1(obs, pi, goal)
        q2_pi = self.ac[0].q2(obs, pi, goal)
        values = torch.min(q1_pi, q2_pi)
        values = values.reshape(self.crr_n_action_samples, -1)
        probs = torch.softmax(values, 0)
        idxs = torch.multinomial(probs.T, 1, replacement=True)
        return pi.reshape(self.crr_n_action_samples, -1)[idxs].squeeze().cpu().detach().numpy()

    def add_rollouts_to_buffer(self):
        """Populates the buffer with model-based rollouts."""
        if not (self.mbpo_buffer and self.mbpo_accumulate):
            self.mbpo_buffer = ReplayBuffer(self.mbpo_buffer_capacity,
                                           self.model_env.observation_space.shape,
                                           self.model_env.action_space.shape,
                                           self.model_env.goal_space.shape,
                                           max_trajectory_length=self.mbpo_rollout_horizon,
                                           rng=self.replay_buffer.rng,
                                           goal_proj_fn=self.replay_buffer.goal_proj_fn)
        iterator = self.get_iter()(
                transitions=self.replay_buffer.get_all(shuffle=False),
                trajectory_indices=self.replay_buffer.get_trajectory_indices(),
                batch_size=self.mbpo_num_rollouts,
                relabeling_fn=self.relabeling_fn,
                goal_proj_fn=self.replay_buffer.goal_proj_fn,
                rng=self.replay_buffer.rng)
        indices = torch.randint(len(self.replay_buffer), size=(self.mbpo_num_rollouts,),
                                generator=self.generator, device=self.device).detach().cpu().numpy()
        batch = iterator[indices]
        obs, *_, desired_goals = cast(TransitionBatch, batch).astuple()

        # TODO: consider using particles, removing degenerate trajectories from buffer
        batches = []
        with torch.no_grad():
            model_state = self.model_env.reset(obs, return_as_np=False)
            obs = to_tensor(obs).to(self.device).float()
            desired_goals = to_tensor(desired_goals).to(self.device).float()
            for _ in range(self.mbpo_rollout_horizon):
                # TODO: use a high-level inferface instead
                action, _ = self.ac[0].pi(obs, goals=desired_goals)
                model_in, norm_obs, _ = self.model._get_model_input(model_state["obs"], action, with_items=True)
                means, logvars = self.model.model._default_forward(model_in, only_elite=True)
                stds = torch.sqrt(logvars.exp())
                norm_next_obs = torch.normal(means, stds, generator=self.generator)
                # sample bootstrap uniformly
                idxs  = torch.randint(norm_next_obs.shape[0], (norm_next_obs.shape[1],),
                                      generator=self.generator, device=self.device)
                norm_next_obs = norm_next_obs[idxs, torch.arange(norm_next_obs.shape[1])]
                if self.model.target_is_delta:
                    tmp_ = norm_next_obs + norm_obs
                    for dim in self.model.no_delta_list:
                        tmp_[:, dim] = norm_next_obs[:, dim]
                    norm_next_obs = tmp_
                next_obs = self.model_env.dynamics_model._get_model_output(norm_next_obs).float()
                model_state["obs"] = next_obs
                rewards = - self.mbpo_lambda * torch.linalg.norm(stds.max(0)[0], dim=-1, keepdims=True)
                dones = torch.zeros_like(rewards).bool()
                batches.append({k: v.detach().cpu().numpy() for k, v in zip(['obs', 'dg', 'next_obs', 'act', 'rew', 'done'],
                  [obs, desired_goals, next_obs, action, rewards, dones])})
                obs = next_obs

            batches = {k: np.stack([b[k] for b in batches], 1) for k in batches[0]}
            batches['done'][:, -1] = True
            batches = {k: v.reshape(-1, *v.shape[2:]) for k, v in batches.items()}
            joint_batch = (
                    {'observation': batches['obs'], 'desired_goal': batches['dg']},
                    batches['act'],
                    {'observation': batches['next_obs'], 'desired_goal': batches['dg']},
                    batches['rew'].reshape(-1),
                    batches['done'].reshape(-1),
            )
            self.mbpo_buffer.add_batch(*joint_batch)

        self.mbpo_buffer_iterator = MOPOIterator(
                relabeling=self.relabeling,
                p_geometric=self.p_geometric,
                p_rand_goals=self.p_rand_goals,
                transitions=self.mbpo_buffer.get_all(shuffle=False),
                trajectory_indices=self.mbpo_buffer.get_trajectory_indices(),
                batch_size=self.mbpo_batch_size,
                relabeling_fn=self.relabeling_fn,
                goal_proj_fn=self.mbpo_buffer.goal_proj_fn,
                rng=self.mbpo_buffer.rng,
            )
        self.curr_iterator = iter(self.mbpo_buffer_iterator)

    def sample_batch_from_buffer(self):
        try:
            batch = next(self.curr_iterator)
        except StopIteration:
            self.curr_iterator = iter(self.mbpo_buffer_iterator)
            batch = next(self.curr_iterator)
        return batch


class GraphModule(Module):
    """Module implementing graph-based value correction.
    """

    name = "graph"
    EVAL_LOG_FORMAT = []

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        attrs = ['threshold', 'min_threshold', 'update_threshold', 'buffer_sample_size',
                 'density_sampling', 'ensure_start_to_goal', 'connect_start', 'connect_goal',
                 'value_estimator', 'sorb']
        [setattr(self, k, kwargs[k]) for k in attrs]
        super().__init__(*args, **kwargs)

    def observe(
        self,
        **kwargs
    ):
        """Saves wrapped module."""
        self.v_fn = kwargs['additional_modules'][self.value_estimator].reward_fn
        self.gamma = kwargs['additional_modules'][self.value_estimator].gamma
        super().observe(**kwargs)

    def _recompute_graph(
        self,
        start,
        goal
    ):
        """Builds graph from replay buffer
        """
        value_estimator = self.v_fn

        buffer_sample = self.replay_buffer.get_all()
        self.obs = buffer_sample.obs
        self.goals = self.replay_buffer.goal_proj_fn(self.obs)
        self.obs, self.goals = (to_tensor(x).to(self.device).float() for x in [self.obs, self.goals])
        subsample_idxs = torch.randperm(len(buffer_sample), generator=self.generator, device=self.device)[:1000].detach().cpu().numpy()
        dataset = self.goals[subsample_idxs]
        n_batches = 1000
        batch_size = dataset.shape[0] // n_batches
        tiled_obs = torch.tile(self.obs, (batch_size, 1))
        distances = []
        for i in range(n_batches):
            tiled_goals = dataset[(i)*batch_size:(i+1)*batch_size].repeat_interleave(self.obs.shape[0], dim=0)
            distances.append(-value_estimator(None, None, tiled_obs, tiled_goals).squeeze().detach().cpu().numpy())
        # clipping prevents NaN in probabilities for very low Q values
        density = np.exp(-(np.log(-np.stack(distances, 0)) / np.log(0.99)).clip(None, 500) / 20.0).mean(0)
        p = 1/density
        density_p = p/p.sum()

        p = np.ones_like(density_p)
        uniform_p = p/p.sum()
        if self.density_sampling:
            uniform_p *= density_p
        uniform_p = uniform_p/uniform_p.sum()
        idxs = np.random.choice(np.arange(len(self.goals)), size=self.buffer_sample_size, replace=False, p=uniform_p)
        self.obs = self.obs[idxs]
        self.goals = self.goals[idxs]

        # --- graph construction takes roughly 5 seconds
        distances = []
        with torch.no_grad():
            for o in self.obs:
                distances.append(-value_estimator(None, None, torch.stack([o] * self.goals.shape[0], 0), self.goals))
            distances = torch.stack(distances, 0).cpu().detach().squeeze().numpy()
            distances = np.log(-distances) / np.log(self.gamma)
        self.graph = nx.from_numpy_array(distances, parallel_edges=False, create_using=nx.DiGraph)
        edge_weights = nx.get_edge_attributes(self.graph,'weight')

        threshold = self.threshold
        from heapq import heappop, heappush
        from itertools import count
        def maximum_edge_length(
            G, source
        ):
            G_succ = G._adj
            push = heappush
            pop = heappop
            dist = {}
            seen = {source: 0}
            c = count()
            fringe = [(0, next(c), source)]
            while fringe:
                (d, _, v) = pop(fringe)
                if v in dist:
                    continue  # already searched this node.
                dist[v] = d
                for u, e in G_succ[v].items():
                    cost = e.get('weight')
                    if cost is None:
                        continue
                    # vu_dist = dist[v] + cost
                    vu_dist = max(dist[v], cost)
                    if u in dist:
                        u_dist = dist[u]
                        if vu_dist < u_dist:
                            raise ValueError("Contradictory paths found:", "negative weights?")
                    elif u not in seen or vu_dist < seen[u]:
                        seen[u] = vu_dist
                        push(fringe, (vu_dist, next(c), u))
            return dist

        if self.ensure_start_to_goal:
            curr_graph = deepcopy(self.graph)
            start_id = self.buffer_sample_size
            goal_id = self.buffer_sample_size + 1
            curr_graph.add_nodes_from([start_id, goal_id])
            distances_from_start = -self.v_fn(None, None, torch.stack([start] * self.goals.shape[0], 0), self.goals).detach().cpu().numpy()
            distances_from_start = np.log(-distances_from_start) / np.log(self.gamma)
            curr_graph.add_weighted_edges_from([(start_id, i, d) for i, d in enumerate(distances_from_start)])
            distances_to_goal = -value_estimator(None, None, self.obs, goal).detach().cpu().numpy()
            distances_to_goal = np.log(-distances_to_goal) / np.log(self.gamma)
            curr_graph.add_weighted_edges_from([(i, goal_id, d) for i, d in enumerate(distances_to_goal)])
            lengths = maximum_edge_length(curr_graph.reverse(), goal_id)
            lengths = [v.item() if isinstance(v,np.ndarray) else v for v in lengths.values()]

            if self.ensure_start_to_goal:
                threshold = lengths[start_id]
            
        if self.update_threshold:
            self.threshold = threshold

        self.graph.remove_edges_from((e for e, w in edge_weights.items() if w > threshold))
        self.graph.remove_edges_from((e for e, w in edge_weights.items() if w < self.min_threshold))

        curr_graph = deepcopy(self.graph)
        goal_id = self.buffer_sample_size
        curr_graph.add_nodes_from([goal_id])
        distances_to_goal = -self.v_fn(None, None, self.obs, goal).detach().cpu().numpy()
        distances_to_goal = np.log(-distances_to_goal) / np.log(self.gamma)
        curr_graph.add_weighted_edges_from([(i, goal_id, d) for i, d in enumerate(distances_to_goal)])
        edge_weights = nx.get_edge_attributes(curr_graph,'weight')
        threshold = max(self.threshold, np.min(distances_to_goal)) if self.connect_goal else self.threshold
        curr_graph.remove_edges_from((e for e, w in edge_weights.items() if w > threshold))
        distances_on_graph = nx.shortest_path_length(curr_graph, target=goal_id, weight='weight')
        del distances_on_graph[self.goals.shape[0]]  # remove goal to goal distance
        self.vertex_to_goal = torch.ones(self.goals.shape[0], device=self.device) * 10000
        if len(distances_on_graph):
            idxs, values = zip(*[(k, v) for k, v in distances_on_graph.items()])
            self.vertex_to_goal[list(idxs)] = torch.tensor(np.concatenate(values), device=self.device)
            # TODO: discard subgoals which are not connected to goal!

    def _find_subgoal(
        self, 
        obs,
    ):
        """Finds subgoal from current obs.
        """

        start, goal = (to_tensor(x).to(self.device).float() for x in [obs['observation'], obs['desired_goal']])
        
        if self.graph is None:
            self._recompute_graph(start, goal)

        curr_graph = deepcopy(self.graph)
        start_id = self.buffer_sample_size
        goal_id = self.buffer_sample_size + 1
        curr_graph.add_nodes_from([start_id, goal_id])

        distances_from_start = -self.v_fn(None, None, torch.stack([start] * self.goals.shape[0], 0), self.goals).detach().cpu().numpy()
        distances_from_start = np.log(-distances_from_start) / np.log(self.gamma)

        if self.sorb:
            dist_to_goal = -self.v_fn(None, None, start.unsqueeze(0), goal).detach().cpu().numpy()
            dist_to_goal = (np.log(-dist_to_goal) / np.log(self.gamma)).item()
            idx = (distances_from_start.flatten() + self.vertex_to_goal.detach().cpu().numpy()).argmin()
            if (self.vertex_to_goal[idx] >= 10000) or dist_to_goal < min(distances_from_start[idx], self.threshold):
                self.subgoal = deepcopy(goal)
            else:
                self.subgoal = deepcopy(self.goals[idx])
        else:
            idxs = np.argsort(distances_from_start.flatten())[:100].reshape(-1)
            self.near_goals = self.goals[idxs]
            self.near_vertex_to_goal = self.vertex_to_goal[idxs]
            self.near_goals = torch.cat([self.near_goals, goal.unsqueeze(0)], 0)
            self.near_vertex_to_goal = torch.cat([self.near_vertex_to_goal, torch.zeros((1,), device=self.device)], 0)
            self.subgoal = deepcopy(goal)

    def on_reset(
        self,
        obs: np.ndarray,
    ):
        """Processes environment resets.

        Args:
            obs (np.ndarray): starting observation.
            replay_buffer (ReplayBuffer): the current replay buffer.
        """
        self.graph = None
        self.near_goals = None
        self._find_subgoal(obs)
        return {}

    def on_step(
        self,
        experience: Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool],
    ):
        """Processes the experience after each environment step.

        Args:
            experience (Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]): 
                tuple containing observation, action, next_observation, reward, done signal.
            replay_buffer (ReplayBuffer): the current replay buffer.
        """
        _, _, next_obs, _, _= experience
        self._find_subgoal(next_obs)
        return {}

    def reward_fn(
        self,
        obs: torch.Tensor, 
        act: torch.Tensor,
        next_obs: torch.Tensor,
        goal: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Reward computed as distance to subgoal selected in on_step.

        Args:
            obs (torch.Tensor): observation preceding the action
            act (torch.Tensor): action chosen by the agent
            next_obs (torch.Tensor): observation following the action
            goal (torch.Tensor, optional): goal for goal-conditioned planning

        Returns:
            torch.Tensor: reward scalar
        """

        if self.near_goals is not None:
            tiled_obs = torch.tile(next_obs, (self.near_goals.shape[0], 1))
            # TODO: pre-interleave goals
            tiled_goals = self.near_goals.repeat_interleave(next_obs.shape[0], dim=0)
            distances = -self.v_fn(None, None, tiled_obs, tiled_goals)
            distances = distances.reshape(self.near_goals.shape[0], -1)
            distances = torch.log(-distances) / np.log(self.gamma)
            # for each obs, find the threshold such that it is connected to the goal
            nvtg = self.near_vertex_to_goal.reshape(-1, 1)
            # find minimum distance that connects to goal
            thresh = torch.min(distances + torch.where(nvtg>=10000, 2*10000, 0), 0).values
            thresh = torch.clip(thresh, self.threshold, None if self.connect_start else self.threshold)
            distances = torch.where(distances <= thresh, distances, 10000)
            distances = distances + self.near_vertex_to_goal.reshape(-1, 1)
            distances = torch.min(distances, 0).values.reshape(-1, 1)
            disconnected = (distances >= 10000)
            distances = (torch.exp(distances*np.log(self.gamma)))
            # if not attached, just use distance to goal
            distances = torch.where(disconnected, self.v_fn(None, None, next_obs, self.subgoal), distances)
            return distances
        return self.v_fn(None, None, next_obs, self.subgoal)
