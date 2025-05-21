import wandb
from pprint import pprint
import os
import yaml
# import hydra
import random
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

# import albumentations as A
# from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
from omegaconf import OmegaConf
from dataclasses import dataclass
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, Subset, random_split
from omegaconf import DictConfig


def visualize_dataset_grid(dataset, num_samples=6, cols=3):
    rows = (num_samples + cols - 1) // cols
    fig, axs = plt.subplots(rows * 2, cols, figsize=(cols * 4, rows * 2.5))

    for i in range(num_samples):
        image, mask = dataset[i]
        r = (i // cols) * 2
        c = i % cols

        axs[r, c].imshow(TF.to_pil_image(image))
        axs[r, c].set_title(f"Image {i}")
        axs[r, c].axis("off")

        axs[r + 1, c].imshow(mask, cmap="gray", vmin=0, vmax=2)
        axs[r + 1, c].set_title(f"Mask {i}")
        axs[r + 1, c].axis("off")

    # Hide any unused subplots
    total_plots = rows * cols
    for j in range(num_samples, total_plots):
        axs[(j // cols) * 2, j % cols].axis("off")
        axs[(j // cols) * 2 + 1, j % cols].axis("off")

    plt.tight_layout()
    plt.show()

# Define a color palette for your three classes (pets, background, boundary-ignored)
PALETTE = np.array([
    [255, 0,   0],    # class 0 (pet) → red
    [0,   255, 0],    # class 1 (background) → green
    [0,   0,   255],  # class 2 (boundary/ignored) → blue (we’ll drop this in overlay)
], dtype=np.uint8)

def mask_to_color(mask):
    """
    mask:  H×W  numpy array of ints in [0, num_classes)
    returns: H×W×3 uint8 RGB array
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(PALETTE.shape[0]):
        color_mask[mask == c] = PALETTE[c]
    return color_mask

def visualize_predictions(model, dataloader, device, num_batches=1):
    model.eval()
    with torch.no_grad():
        for i, (xb, yb) in enumerate(dataloader):
            xb = xb.to(device)
            yb = yb.squeeze(1).cpu().numpy()    # [B, H, W]
            logits = model(xb)                  # [B, C, H, W]
            preds = torch.argmax(logits, dim=1).cpu().numpy()  # [B, H, W]

            batch_size = xb.size(0)
            for j in range(batch_size):
                img = TF.to_pil_image(xb[j].cpu())
                gt_mask = yb[j]
                pred_mask = preds[j]

                # Convert to color
                gt_color = mask_to_color(gt_mask)
                pred_color = mask_to_color(pred_mask)

                # Overlay prediction on image
                img_np = np.array(img)
                overlay = img_np.copy()
                alpha = 0.5
                # only overlay pet class (class 0) for clarity
                pet_region = (pred_mask == 0)
                overlay[pet_region] = (
                    img_np[pet_region] * (1 - alpha) +
                    PALETTE[0] * alpha
                ).astype(np.uint8)

                # Plot side-by-side
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(img);            axes[0].set_title("Image");     axes[0].axis("off")
                axes[1].imshow(gt_color);       axes[1].set_title("GT Mask");  axes[1].axis("off")
                axes[2].imshow(pred_color);     axes[2].set_title("Pred Mask");axes[2].axis("off")
                axes[3].imshow(overlay);        axes[3].set_title("Overlay");  axes[3].axis("off")
                plt.tight_layout()
                plt.show()

            if i + 1 >= num_batches:
                break


def load_and_test_model(config, val_dl):
    # Setup W&B API
    # Replace with your actual project, run, and artifact names
    api = wandb.Api()
    artifact_name = 'jbarry-team/unet-oxford-pet/unet-server_best_model:latest'
    try:
        artifact = api.artifact(artifact_name, type='model')
        artifact_dir = artifact.download()

        # Load model
        model = UNet(in_channels=3, out_channels=3)
        model.load_state_dict(torch.load(f"{artifact_dir}/best_model.pth", map_location="cpu"))
        model.eval()
        print("Model loaded successfully.")
    except wandb.CommError as e:
        model = UNet(in_channels=3, out_channels=3).to(config.device)
        print(f"Artifact not found: {artifact_name}")
        print(f"Error: {e}")
    visualize_predictions(model, val_dl, config.device, num_batches=1)


def main(config):
    torch.manual_seed(config.seed)
    print(f"Config seed {config.seed}")

    print("Config device", config.device)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    dataset = configure_dataset(config)
    # visualize_dataset_grid(dataset, num_samples=8)
    train_dl, val_dl = get_train_val_dl(dataset, config)

    if config.test_model:
        load_and_test_model(config, val_dl)
    elif config.train_model:
        train_and_validate_model(train_dl, val_dl, config)


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

if __name__ == "__main__":
    env = os.environ.get("ENV", "local")
    config = load_config(env)

    print(f"Running in {env} environment")
    pprint(OmegaConf.to_container(config, resolve=True))

    main(config)

