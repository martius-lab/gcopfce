import hydra
import numpy as np
import omegaconf
import torch

import mbrl.algorithms.pets as pets
from mbrl.util.env import make_env


@hydra.main(version_base='1.1', config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    env, term_fn, reward_fn, additional_modules = make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    return pets.train(env, term_fn, reward_fn, additional_modules, cfg)

if __name__ == "__main__":
    run()
