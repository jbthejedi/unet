import torch
import os
import torchvision.transforms as T
import matplotlib.pyplot as plt

from tqdm import tqdm
from pprint import pprint
from torch.utils.data import DataLoader, Subset, random_split
from omegaconf import OmegaConf 


def load_config(env="local"):
    base_config = OmegaConf.load("config/base.yaml")

    env_path = f"config/{env}.yaml"
    if os.path.exists(env_path):
        env_config = OmegaConf.load(env_path)
        # Merges env_config into base_config (env overrides base)
        config = OmegaConf.merge(base_config, env_config)
    else:
        config = base_config

    return config


def main():
    env = os.environ.get("ENV", "local")
    cfg = load_config(env)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    pprint(cfg_dict)

    main(cfg)

 
if __name__ == '__main__':
    main()
