import torch


def maze(obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    if len(goal.shape) < len(next_obs.shape):
        goal = goal.unsqueeze(0)
    return (torch.linalg.norm(next_obs[:, :2] - goal[:, :2], dim=-1, keepdim=True) <= 0.5).float()

def kitchen(obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    if len(goal.shape) < len(next_obs.shape):
        goal = goal.unsqueeze(0)
    obs_idxs = [[11,12], [15,16], [17,18], [19], [20,21], [22], [23,24,25,26,27,28,29]]
    goal_idxs = [[0,1], [2,3], [4,5], [6], [7,8], [9], [10,11,12,13,14,15,16]]
    rewards = [(torch.linalg.norm(next_obs[..., o_id] - goal[..., g_id], dim=-1) < 0.3) for o_id, g_id in zip(obs_idxs, goal_idxs)]
    return torch.stack(rewards, -1).all(-1, keepdim=True).float()

def fetch(obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    if len(goal.shape) < len(next_obs.shape):
        goal = goal.unsqueeze(0)
    return (torch.linalg.norm(next_obs[:, 3:6] - goal, dim=-1, keepdim=True) <= 0.05).float()

def zero(obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(next_obs, dim=-1, keepdims=True) * 0.

def pinpad(obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    if len(goal.shape) < len(next_obs.shape):
        goal = goal.unsqueeze(0)
    return (torch.linalg.norm(next_obs[:, 4:] - goal, dim=-1, keepdim=True) <= 0.5).float()
