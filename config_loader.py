import os
import yaml

def load_config(env="local"):
    with open("config/base.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    env_path = f"config/{env}.yaml"
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            env_config = yaml.safe_load(f)
        base_config.update(env_config)

    return base_config

