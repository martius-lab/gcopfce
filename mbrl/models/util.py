from typing import List, Sequence
import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import mbrl.types
import mbrl.util.math


def truncated_normal_init(m: nn.Module):
    """Initializes the weights of the given module using a truncated normal distribution."""

    if isinstance(m, nn.Linear):
        input_dim = m.weight.data.shape[0]
        stddev = 1 / (2 * np.sqrt(input_dim))
        mbrl.util.math.truncated_normal_(m.weight.data, std=stddev)
        m.bias.data.fill_(0.0)
    if isinstance(m, EnsembleLinearLayer):
        num_members, input_dim, _ = m.weight.data.shape
        stddev = 1 / (2 * np.sqrt(input_dim))
        for i in range(num_members):
            mbrl.util.math.truncated_normal_(m.weight.data[i], std=stddev)
        m.bias.data.fill_(0.0)


def rnd_weight_init(m: nn.Module):
    """Custom weight init for Conv2D and Linear layers for RND."""

    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class EnsembleLinearLayer(nn.Module):
    """Efficient linear layer for ensemble models."""

    def __init__(
        self, num_members: int, in_size: int, out_size: int, bias: bool = True
    ):
        super().__init__()
        self.num_members = num_members
        self.in_size = in_size
        self.out_size = out_size
        # TODO: check if this can be done with a generator
        self.weight = nn.Parameter(
            torch.rand(self.num_members, self.in_size, self.out_size)
        )
        if bias:
            # TODO: check if this can be done with a generator
            self.bias = nn.Parameter(torch.rand(self.num_members, 1, self.out_size))
            self.use_bias = True
        else:
            self.use_bias = False

        self.elite_models: List[int] = None
        self.use_only_elite = False

    def forward(self, x):
        if self.use_only_elite:
            xw = x.matmul(self.weight[self.elite_models, ...])
            if self.use_bias:
                return xw + self.bias[self.elite_models, ...]
            else:
                return xw
        else:
            xw = x.matmul(self.weight)
            if self.use_bias:
                return xw + self.bias
            else:
                return xw

    def extra_repr(self) -> str:
        return (
            f"num_members={self.num_members}, in_size={self.in_size}, "
            f"out_size={self.out_size}, bias={self.use_bias}"
        )

    def set_elite(self, elite_models: Sequence[int]):
        self.elite_models = list(elite_models)

    def toggle_use_only_elite(self):
        self.use_only_elite = not self.use_only_elite


def to_tensor(x: mbrl.types.TensorType):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    raise ValueError("Input must be torch.Tensor or np.ndarray.")


def mlp(sizes, activation, output_activation=nn.Identity):
    """ MLP builder from SpinningUp. """
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class SquashedGaussianMLPActor(nn.Module):
    """ SAC-like actor from SpinningUp. """
    def __init__(self, obs_dim, act_dim, goal_dim, hidden_sizes, activation, act_limit, p_norm, squash_policy):
        super().__init__()
        self.net = mlp([obs_dim + goal_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit
        self.p_norm = p_norm
        self.squash_policy = squash_policy
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, obs, goals, deterministic=False):
        net_out = self.net(torch.cat([obs, goals], -1))
        if self.p_norm:
            net_out = torch.nn.functional.normalize(net_out)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        pi_action = mu if deterministic else pi_distribution.rsample()

        if deterministic:
            logp_pi = torch.zeros_like(pi_action, device=pi_action.device).sum(axis=-1)
        else:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            if self.squash_policy:
                logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        if self.squash_policy:
            pi_action = torch.tanh(pi_action)
            pi_action = self.act_limit * pi_action
        else:
            pi_action = pi_action.clip(-self.act_limit, self.act_limit)
        return pi_action, logp_pi

    def get_log_probs(self, obs, act, goals):
        net_out = self.net(torch.cat([obs, goals], -1))
        if self.p_norm:
            net_out = torch.nn.functional.normalize(net_out)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        pi_action = act / self.act_limit
        pi_action = torch.atanh(pi_action)
        pi_action = pi_action.clip(-3, 3)
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1).clip(-30)
        logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        return logp_pi


class MLPQFunction(nn.Module):
    """ SAC-like Q-function from SpinningUp. """
    def __init__(self, obs_dim, act_dim, goal_dim, hidden_sizes, activation, sigmoid, p_norm):
        super().__init__()
        output_activation = nn.Identity if not sigmoid else nn.Sigmoid
        self.p_norm = p_norm
        if self.p_norm:
            self.q = mlp([obs_dim + act_dim + goal_dim] + list(hidden_sizes), activation)
            self.last_layer = nn.Linear(hidden_sizes[-1], 1)
            self.output_activation = output_activation()
        else:
            self.q = mlp([obs_dim + act_dim + goal_dim] + list(hidden_sizes) + [1], activation, output_activation)

    def forward(self, obs, act, goals):
        q = self.q(torch.cat([obs, act, goals], dim=-1))
        if self.p_norm:
            q = torch.nn.functional.normalize(q)
            q = self.output_activation(self.last_layer(q))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    """ SAC-like actor-critic interface with double Q-networks, from SpinningUp. """
    def __init__(self, obs_dim, act_dim, goal_dim, act_limit, sigmoid, hidden_sizes=(256,256),
                 activation=nn.ReLU, p_norm=False, squash_policy=True):
        super().__init__()
        obs_dim = obs_dim
        act_dim = act_dim
        act_limit = act_limit
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, goal_dim, hidden_sizes, activation, act_limit, p_norm, squash_policy)
        self.q1 = MLPQFunction(obs_dim, act_dim, goal_dim, hidden_sizes, activation, sigmoid, p_norm)
        self.q2 = MLPQFunction(obs_dim, act_dim, goal_dim, hidden_sizes, activation, sigmoid, p_norm)
