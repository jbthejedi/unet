import wandb
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from tqdm import tqdm
from dataclasses import dataclass
from torchvision.datasets import OxfordIIITPet


seed = 1337
torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class Config:
    name : str = "unet-oxford-pet"
    run : str = "unet-run"
    lr : float = 1e-3
    image_size : int = 128

def train_and_validate_model(config : Config):
    wandb.init(
        mode='disabled'
    )
    transform = T.Compose([
        T.Resize((config.image_size, config.image_size)),
        T.ToTensor(),
    ])
    target_transform = T.Compose([
        T.Resize((config.image_size, config.image_size)),
        T.PILToTensor(),
        T.Lambda(lambda x: x.long().squeeze(0) - 1)
    ])

    dataset = OxfordIIITPet(
        root='.',
        download=True,
        target_types='segmentation',
        transform=transform,
        target_transform=target_transform
    )

def main():
    config = Config()
    train_and_validate_model(config)
    
if __name__ == "__main__":
    main()






