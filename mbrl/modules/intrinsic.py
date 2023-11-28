import pathlib
from typing import Optional, Callable
import torch
import torch.nn as nn
from mbrl.models.util import rnd_weight_init
from mbrl.types import ModelInput
from mbrl.util.replay_buffer import AdditionalIterator
from mbrl.modules.core import Module


class RNDModule(Module):
    """Module computing an RND-based intrinsic motivation signal.
    """

    name = "rnd"
    EVAL_LOG_FORMAT = []

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.obs_dim, self.rnd_hidden_dim, self.rnd_rep_dim, self.lr = \
            (kwargs.get(k) for k in ['obs_dim', 'rnd_hidden_dim', 'rnd_rep_dim', 'rnd_lr'])

    def init(self):
        """Initializes the module."""
        self.predictor = nn.Sequential(nn.Linear(self.obs_dim, self.rnd_hidden_dim), nn.ReLU(),
                                       nn.Linear(self.rnd_hidden_dim, self.rnd_hidden_dim), nn.ReLU(),
                                       nn.Linear(self.rnd_hidden_dim, self.rnd_rep_dim)).to(self.device)
        self.target = nn.Sequential(nn.Linear(self.obs_dim, self.rnd_hidden_dim), nn.ReLU(),
                                    nn.Linear(self.rnd_hidden_dim, self.rnd_hidden_dim), nn.ReLU(),
                                    nn.Linear(self.rnd_hidden_dim, self.rnd_rep_dim)).to(self.device)
        for param in self.target.parameters():
            param.requires_grad = False
        self.apply(rnd_weight_init)
        self.rnd_opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        super().init()

    def update(
        self,
        batch: ModelInput
    ) -> float:
        """ Trains the RND module.

        Args:
            model_in (tensor or batch of transitions): the inputs to the model.

        Returns:
            float: loss for logging
        """
        self.train()
        self.rnd_opt.zero_grad()
        obs = self.get_module_input(batch)[0]
        prediction, target = self.predictor(obs), self.target(obs)
        prediction_error = torch.square(target.detach() - prediction).mean(dim=-1, keepdim=True)
        loss = prediction_error.mean()
        loss.backward()
        self.rnd_opt.step()
        return loss.cpu().detach().numpy().item()

    def reward_fn(
        self,
        obs: torch.Tensor, 
        act: torch.Tensor,
        next_obs: torch.Tensor,
        goal: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Reward computed through RND.

        Args:
            obs (torch.Tensor): observation preceding the action
            act (torch.Tensor): action chosen by the agent
            next_obs (torch.Tensor): observation following the action
            goal (torch.Tensor, optional): goal for goal-conditioned planning

        Returns:
            torch.Tensor: reward scalar
        """
        next_obs = self.normalize_obs(next_obs)
        with torch.no_grad():
            prediction, target = self.predictor(next_obs), self.target(next_obs)
            prediction_error = torch.square(target.detach() - prediction).mean(dim=-1, keepdim=True)
            reward = prediction_error
        return reward

    def get_iter(
        self,
    ) -> Callable:
        return AdditionalIterator

    def get_state(self):
        return {**super().get_state(), 'optimizer': self.rnd_opt.state_dict()}

    def set_state(self, state):
        self.rnd_opt.load_state_dict(state['optimizer'])
        super().set_state(state)


class DisagreementModule(Module):
    """Module computing a disagreement-based intrinsic motivation signal.
    """

    name = "disagreement"
    EVAL_LOG_FORMAT = []

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        attrs = ['relevant_dims', 'obs_dim']
        [setattr(self, k, kwargs[k]) for k in attrs]
        if not self.relevant_dims:
            self.relevant_dims = [n for n in range(self.obs_dim)]

    def reward_fn(
        self,
        obs: torch.Tensor, 
        act: torch.Tensor,
        next_obs: torch.Tensor,
        goal: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Reward computed through RND.

        Args:
            obs (torch.Tensor): observation preceding the action
            act (torch.Tensor): action chosen by the agent
            next_obs (torch.Tensor): observation following the action
            goal (torch.Tensor, optional): goal for goal-conditioned planning

        Returns:
            torch.Tensor: reward scalar
        """

        with torch.no_grad():
            model_in = self.model._get_model_input(obs, act)
            means, logvars = self.model.model._default_forward(model_in, only_elite=False)
            variances = logvars.exp()
            reward = (means.var(0) + variances.var(0))[..., self.relevant_dims].mean(-1, keepdim=True)
        return reward

    def get_iter(
        self,
    ) -> Callable:
        return AdditionalIterator
