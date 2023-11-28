import torch


def maze(obs: torch.Tensor, compact_goal_space: bool) -> torch.Tensor:
    assert len(obs.shape) == 2
    return obs[:, :2] if compact_goal_space else obs

def kitchen(obs: torch.Tensor, compact_goal_space: bool) -> torch.Tensor:
    return obs[:, [11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]]

def fetch(obs: torch.Tensor, compact_goal_space: bool) -> torch.Tensor:
    assert len(obs.shape) == 2
    return obs[:, 3:6] if compact_goal_space else obs

def pinpad(obs: torch.Tensor, compact_goal_space: bool) -> torch.Tensor:
    assert len(obs.shape) == 2
    return obs[:, 4:]
