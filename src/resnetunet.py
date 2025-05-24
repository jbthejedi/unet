import wandb
from PIL import Image
from pprint import pprint
import os
import yaml
from torchinfo import summary
import torchvision.models as models
import random
import torch

import torchvision.models as models
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from tqdm import tqdm
from omegaconf import OmegaConf
from dataclasses import dataclass
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, Subset, random_split, Dataset
from omegaconf import DictConfig


class DoubleConv(nn.Module):
    def __init__(
            self, in_channels,
            out_channels,
            p_dropout=None
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if p_dropout:
            layers.append(nn.Dropout(p=p_dropout))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


# ================================================================
# Layer (type)         Output Shape         Param #    ...
# ================================================================
# conv1                [1,  64, 112, 112]   9.5K
# bn1                  [1,  64, 112, 112]   128
# maxpool              [1,  64,  56,  56]   0
# layer1               [1,  64,  56,  56]   ...
# layer2               [1, 128,  28,  28]   ...
# layer3               [1, 256,  14,  14]   ...
# layer4               [1, 512,   7,   7]   ...
class UpBlock(nn.Module):

    """Upsample → DoubleConv, with optional skip-concat"""
    def __init__(self, in_ch, out_ch, p_dropout=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2) # Double H,W dims
        self.conv = DoubleConv(in_ch, out_ch, p_dropout=p_dropout)

        self.up_special = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2) # Double H,W dims
        self.conv_special = DoubleConv(in_ch * 2, in_ch, p_dropout=p_dropout)

    def forward(self, x, skip: torch.Tensor = None):
        if x.size(1) == skip.size(1):
            x = self.up_special(x) # -> (8, 64, 112, 112)
            cat = torch.cat([x, skip], dim=1) # -> (8, 128, 112, 112)
            x = self.conv_special(cat) # -> (8, 64, 112, 112)
            return x
        x = self.up(x)

        diffY = skip.size(2) - x.size(2) # Height
        diffX = skip.size(3) - x.size(3) # Width
        if diffY != 0 or diffX != 0:
            print("Difference in height or width")
            print(f"x.shape {x.shape}")
            print(f"skip.shape {skip.shape}")
            pad_left = diffX//2
            pad_right = diffX - diffX//2
            pad_top = diffY//2
            pad_bottom = diffY - diffY//2
            x = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class ResNetUNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # 1) Load ResNet-34 & grab layers
        base_model = models.resnet34(pretrained=True)
        self.layer0 = nn.Sequential(
            base_model.conv1,  # 64, /2
            base_model.bn1,
            base_model.relu,
        )
        self.pool0  = base_model.maxpool   # /2 again → 56×56 from 224
        
        self.layer1 = base_model.layer1    # 64, → 56×56
        self.layer2 = base_model.layer2    # 128 → 28×28
        self.layer3 = base_model.layer3    # 256 → 14×14
        self.layer4 = base_model.layer4    # 512 →  7×7

        # 2) Decoder: upsample + double-convs
        self.up4 = UpBlock(512, 256, p_dropout=.2)  # input is layer4 out
        self.up3 = UpBlock(256, 128, p_dropout=.1)
        self.up2 = UpBlock(128,  64, p_dropout=.1)
        self.up1 = UpBlock(64,   64, p_dropout=.1)
        # self.up0 = UpBlock(64,   32)
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        # 3) Final 1×1 conv → class logits
        self.classifier = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.layer0(x)    # → (N,64,112,112)
        x1 = self.pool0(x0)    # → (N,64, 56,56)
        x2 = self.layer1(x1)   # → (N,64, 56,56)
        x3 = self.layer2(x2)   # → (N,128,28,28)
        x4 = self.layer3(x3)   # → (N,256,14,14)

        x5 = self.layer4(x4)   # → (N,512, 7, 7)  ← bottleneck

        # Decoder
        d4 = self.up4(x5, x4)  # → (N,256,14,14)
        d3 = self.up3(d4, x3)  # → (N,128,28,28)
        d2 = self.up2(d3, x2)  # → (N, 64,56,56)
        d1 = self.up1(d2, x0)  # → (N, 64,112,112)
        d0 = self.up0(d1)      # → (N, 32,224,224)

        return self.classifier(d0)

def dice_score(logits, labels, fg_classes=[0], eps=1e-6):
    """Computes mean Dice score for a batch."""
    # print(f"logits.shape {logits.shape}")
    preds = torch.argmax(logits, dim=1)  # [B, H, W]
    # print(f"preds.shape {preds.shape}")
    dice = 0.0
    iou = 0.0
    for c in fg_classes:
        pred_c = (preds == c).float()
        label_c = (labels == c).float()
        intersection = (pred_c * label_c).sum()
        union = pred_c.sum() + label_c.sum()
        dice += (2. * intersection + eps) / (union + eps)
        # Added
        iou += (intersection + eps) / (union + eps)
    dice, iou = dice / len(fg_classes), iou / len(fg_classes)
    return dice, iou

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

def visualize_predictions(model, dataloader, device, num_batches=5):
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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# 1) build your Albumentations pipeline
def get_train_transforms(image_size):
    normalize = A.Normalize(mean=IMAGENET_MEAN, std =IMAGENET_STD)
    return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7,1.0), ratio=(0.75,1.33), p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5),
            A.ToGray(p=0.1),
            A.GaussianBlur(blur_limit=3, p=0.3),
            # optional elastic:
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            # normalize with ImageNet stats (in-place across all channels)
            A.Normalize(mean=(0.485,0.456,0.406), std =(0.229,0.224,0.225)),
            ToTensorV2(),
        ])

def get_val_transforms(image_size):
    normalize = A.Normalize(mean=IMAGENET_MEAN, std =IMAGENET_STD)
    return A.Compose([
        A.Resize(image_size, image_size),
        normalize,
        ToTensorV2(),
    ])

def get_test_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0,0,0), std=(1,1,1)),
        ToTensorV2(),
    ])


# Added
class PetSegDataset(Dataset):
    def __init__(self, base, transform, ignore_index):
        super().__init__()
        self.base = base
        self.transform = transform
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img_pil, mask_pil = self.base[i]
        img  = np.array(img_pil)
        mask = np.array(mask_pil)

        aug    = self.transform(image=img, mask=mask)
        img_t  = aug["image"]
        mask_t = aug["mask"].long()

        # Fix for negative labels:
        mask_t[mask_t == 0] = self.ignore_index  
        mask_t = mask_t - 1

        return img_t, mask_t


def make_split_indices(n, split=0.9, seed=1337):
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    cut = int(split * n)
    return idxs[:cut], idxs[cut:]


def configure_dataset(config):

    base = OxfordIIITPet(root=config.data_root,
                     split="trainval",
                     target_types="segmentation",
                     download=True)

    print(f"len trainval {len(base)}")
    if config.add_augmentation:
        train_ds = PetSegDataset(base, transform=get_train_transforms(config.image_size), ignore_index=2)
    else:
        print(f"No Augmentation!!")
        train_ds = PetSegDataset(base, transform=get_val_transforms(config.image_size), ignore_index=2)
    val_ds   = PetSegDataset(base, transform=get_val_transforms(config.image_size),   ignore_index=2)
    train_idxs, val_idxs = make_split_indices(len(base), split=config.p_train_len, seed=config.seed)
    train_ds = Subset(train_ds, train_idxs)
    val_ds   = Subset(val_ds,   val_idxs)
    print(f"Size of datasets: len(train) {len(train_ds)} len(val) {len(val_ds)}")

    if config.test_run:
        idxs = random.sample(range(len(train_ds)), config.sample_size)
        train_ds = Subset(train_ds, idxs)
        idxs = random.sample(range(len(val_ds)),   config.sample_size)
        val_ds   = Subset(val_ds,   idxs)
        print(f"Downsampling data: len(train) {len(train_ds)} len(val) {len(val_ds)}")

    base = OxfordIIITPet(root=config.data_root,
                     split="test",
                     target_types="segmentation",
                     download=True)
    print(f"len test dataset size {len(base)}")
    test_ds = PetSegDataset(base, transform=get_test_transforms(config.image_size), ignore_index=2)
    return train_ds, val_ds, test_ds
    

def get_train_val_dl(train_ds, val_ds, config):
    train_dl = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_dl, val_dl


def train_and_validate_model_standard(train_dl, val_dl, config):
    print("Train Standard")
    config_dict = OmegaConf.to_container(config, resolve=True)
    wandb.init(
        project=config.project,
        name=config.name,
        config=config_dict,
        mode=config.wandb_mode,
    )

    model = ResNetUNet(n_classes=3).to(config.device)

    # label indexes (pets, background, boundary)
    weights = torch.tensor([1.3, 1.0, 0.0], device=config.device)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)


    best_val_dice = 0.0
    for epoch in range(1, config.n_epochs+1):
        tqdm.write(f"Epoch {epoch}/{config.n_epochs+1}")

        model.train()
        with tqdm(train_dl, desc="Training") as pbar:
            train_loss = 0.0
            train_correct = 0.0
            total_dice = 0.0
            total_iou = 0.0
            train_total_pixels = 0
            total_samples = 0
            for xb, yb in pbar:
                xb, yb = xb.to(config.device), yb.to(config.device)
                logits = model(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, preds = torch.max(logits, 1)

                train_loss += loss.item() * yb.numel()
                train_correct += (preds == yb).sum().item()
                train_total_pixels += yb.numel()

                dice, iou = dice_score(logits, yb)
                total_dice += dice.item() * xb.size(0)
                total_iou += iou.item() * xb.size(0)

                total_samples += xb.size(0)

                pbar.set_postfix(loss=loss.item())

            train_epoch_loss = train_loss / train_total_pixels
            train_epoch_acc = train_correct / train_total_pixels
            train_epoch_dice = total_dice / total_samples
            train_epoch_iou = total_iou / total_samples
            tqdm.write(f"""
            Train Loss {train_epoch_loss:.4f} Train Acc {train_epoch_acc:.2f} 
            Train Dice {train_epoch_dice:.4f} Train IoU {train_epoch_iou:.4f}""")

        model.eval()
        with tqdm(val_dl, desc="Validation") as pbar:
            with torch.no_grad():
                val_loss = 0.0
                val_correct = 0.0
                val_total = 0
                total_dice = 0.0
                total_iou = 0.0
                total_samples = 0
                for xb, yb in pbar:
                    xb, yb = xb.to(config.device), yb.to(config.device)
                    logits = model(xb)
                    loss = criterion(logits, yb)

                    _, preds = torch.max(logits, 1)
                    val_correct += (preds == yb).sum().item()
                    val_loss += loss.item() * yb.numel()
                    val_total += yb.numel()

                    dice, iou = dice_score(logits, yb)
                    total_dice += dice.item() * xb.size(0)
                    total_iou += iou.item() * xb.size(0)

                    total_samples += xb.size(0)

                    pbar.set_postfix(loss=loss.item())

                val_epoch_loss = val_loss / val_total
                val_epoch_acc = val_correct / val_total
                val_epoch_dice = total_dice / total_samples
                val_epoch_iou = total_iou / total_samples
                tqdm.write(f"""
                    Val Loss {val_epoch_loss:.4f} Val Acc {val_epoch_acc:.2f} 
                    Val Dice {val_epoch_dice:.4f} Val IoU {val_epoch_iou:.4f}""")
                pbar.set_postfix(loss=loss.item())

                if config.device == 'cuda' and config.save_model and val_epoch_dice > best_val_dice:
                    tqdm.write("Writing best model...")
                    best_val_dice = val_epoch_dice
                    torch.save(model.state_dict(), "best_model.pth")
                    artifact = wandb.Artifact(name=f"{config.name}_best_model", type="model")
                    artifact.add_file("best_model.pth")
                    wandb.log_artifact(artifact)
                    tqdm.write("Model written.")
        wandb.log({
            "epoch": epoch,
            "train/loss": train_epoch_loss,
            "train/acc": train_epoch_acc,
            "train/dice": train_epoch_dice,
            "train/iou": train_epoch_iou,
            "val/loss": val_epoch_loss,
            "val/acc": val_epoch_acc,
            "val/dice": val_epoch_dice,
            "val/iou": val_epoch_iou,
            "lr": config.lr,
        })
    wandb.finish()
    if config.get("visualize_predictions"):
        visualize_predictions(model, val_dl, config.device, num_batches=1)
    

def train_and_validate_model_enhanced(train_dl, val_dl, config):
    print("Train Enhanced")
    config_dict = OmegaConf.to_container(config, resolve=True)
    wandb.init(
        project=config.project,
        name=config.name,
        config=config_dict,
        mode=config.wandb_mode,
    )

    model = ResNetUNet(n_classes=3).to(config.device)

    # Added
    #### Freeze entire encoder (layer0–layer4), leave decoder & head trainable ####
    for name, param in model.named_parameters():
        if name.startswith("layer") or name.startswith("pool0") or name.startswith("layer0"):
            param.requires_grad = False

    # label indexes (pets, background, boundary)
    weights = torch.tensor([1.3, 1.0, 0.0], device=config.device)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=2)

    # Added: Differential learning rates
    # 1) param groups
    enc = [*model.layer0.parameters(),
        *model.layer1.parameters(),
        *model.layer2.parameters(),
        *model.layer3.parameters(),
        *model.layer4.parameters()]
    # dec = [p for p in model.parameters() if p not in enc]
    dec = [
        p for name, p in model.named_parameters()
        if not name.startswith(("layer0","pool0","layer1","layer2","layer3","layer4"))
    ]
    optimizer = torch.optim.AdamW([
        {"params": enc, "lr": config.enc_lr}, # Encoder LRs
        {"params": dec, "lr": config.dec_lr}, # Decoder LRs
    ], weight_decay=1e-5)

    # Added
    # Cosine scheduler that runs in periods
    # T_0 = number of epochs in the first cycle
    # T_mult = factor by which the cycle length grows each restart
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.t0,
        T_mult=config.tmult,
        eta_min=config.eta_min,
    )

    best_val_dice = 0.0
    for epoch in range(1, config.n_epochs+1):
        tqdm.write(f"Epoch {epoch}/{config.n_epochs+1}")

        # Added
        # Warm-up first 2 epochs
        if epoch <= 2:
            for pg in optimizer.param_groups:
                pg["lr"] = pg["lr"] * (epoch / 2)
                print(f"Warm starting with lr {pg['lr']}")

        # Added
        # ——— Unfreeze logic ———
        if epoch == config.unfreeze_stage0:
            tqdm.write("Unfreezing layers 3 & 4")
            # unfreeze last two stages
            for p in model.layer3.parameters(): p.requires_grad = True
            for p in model.layer4.parameters(): p.requires_grad = True

        if epoch == config.unfreeze_stage1:
            # unfreeze everything
            tqdm.write("Unfreezing entire encoder")
            for p in enc: p.requires_grad = True

        model.train()
        with tqdm(train_dl, desc="Training") as pbar:
            train_loss = 0.0
            train_correct = 0.0
            total_dice = 0.0
            total_iou = 0.0
            train_total_pixels = 0
            total_samples = 0
            for xb, yb in pbar:
                xb, yb = xb.to(config.device), yb.to(config.device)
                logits = model(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, preds = torch.max(logits, 1)

                train_loss += loss.item() * yb.numel()
                train_correct += (preds == yb).sum().item()
                train_total_pixels += yb.numel()

                dice, iou = dice_score(logits, yb)
                total_dice += dice.item() * xb.size(0)
                total_iou += iou.item() * xb.size(0)

                total_samples += xb.size(0)

                pbar.set_postfix(loss=loss.item())

            train_epoch_loss = train_loss / train_total_pixels
            train_epoch_acc = train_correct / train_total_pixels
            train_epoch_dice = total_dice / total_samples
            train_epoch_iou = total_iou / total_samples
            tqdm.write(f"""
            Train Loss {train_epoch_loss:.4f} Train Acc {train_epoch_acc:.2f} 
            Train Dice {train_epoch_dice:.4f} Train IoU {train_epoch_iou:.4f}""")

        model.eval()
        with tqdm(val_dl, desc="Validation") as pbar:
            with torch.no_grad():
                val_loss = 0.0
                val_correct = 0.0
                val_total = 0
                total_dice = 0.0
                total_iou = 0.0
                total_samples = 0
                for xb, yb in pbar:
                    xb, yb = xb.to(config.device), yb.to(config.device)
                    logits = model(xb)
                    loss = criterion(logits, yb)

                    _, preds = torch.max(logits, 1)
                    val_correct += (preds == yb).sum().item()
                    val_loss += loss.item() * yb.numel()
                    val_total += yb.numel()

                    dice, iou = dice_score(logits, yb)
                    total_dice += dice.item() * xb.size(0)
                    total_iou += iou.item() * xb.size(0)

                    total_samples += xb.size(0)

                    pbar.set_postfix(loss=loss.item())

                val_epoch_loss = val_loss / val_total
                val_epoch_acc = val_correct / val_total
                val_epoch_dice = total_dice / total_samples
                val_epoch_iou = total_iou / total_samples
                tqdm.write(f"""
                    Val Loss {val_epoch_loss:.4f} Val Acc {val_epoch_acc:.2f} 
                    Val Dice {val_epoch_dice:.4f} Val IoU {val_epoch_iou:.4f}""")
                pbar.set_postfix(loss=loss.item())

                if config.device == 'cuda' and config.save_model and val_epoch_dice > best_val_dice:
                    tqdm.write("Writing best model...")
                    best_val_dice = val_epoch_dice
                    torch.save(model.state_dict(), "best_model.pth")
                    artifact = wandb.Artifact(name=f"{config.name}_best_model", type="model")
                    artifact.add_file("best_model.pth")
                    wandb.log_artifact(artifact)
                    tqdm.write("Model written.")
        wandb.log({
            "epoch": epoch,
            "train/loss": train_epoch_loss,
            "train/acc": train_epoch_acc,
            "train/dice": train_epoch_dice,
            "train/iou": train_epoch_iou,
            "val/loss": val_epoch_loss,
            "val/acc": val_epoch_acc,
            "val/dice": val_epoch_dice,
            "val/iou": val_epoch_iou,
            "lr": scheduler.get_last_lr()[0],
        })
        scheduler.step()
        tqdm.write(f"Lr {scheduler.get_last_lr()[0]:.2e}")
    wandb.finish()
    if config.get("visualize_predictions"):
        visualize_predictions(model, val_dl, config.device, num_batches=1)


def load_and_test_model(config):
    api = wandb.Api()
    artifact_name = config.artifact_name
    try:
        artifact = api.artifact(artifact_name, type='model')
        artifact_dir = artifact.download()

        # Load model
        model = ResNetUNet(n_classes=3).to(config.device)
        model.load_state_dict(torch.load(f"{artifact_dir}/best_model.pth", map_location="cpu"))
        model.eval()
        print("Model loaded successfully.")

        _, _, val_ds = configure_dataset(config)
        val_dl  = DataLoader(val_ds, batch_size=config.batch_size)
        # model = ResNetUNet(n_classes=3).to(config.device)
        visualize_predictions(model, val_dl, config.device, num_batches=10)
    except wandb.CommError as e:
        print(f"Artifact not found: {artifact_name}")
        print(f"Error: {e}")
        exit(0)

def main(config):
    torch.manual_seed(config.seed)
    print(f"Seed {config.seed}")

    print("Config device", config.device)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    if config.test_model:
        load_and_test_model(config)
    elif config.train_model:
        # dataset = configure_dataset(config)
        # visualize_dataset_grid(dataset, num_samples=8)
        train_ds, val_ds, _ = configure_dataset(config)
        train_dl, val_dl = get_train_val_dl(train_ds, val_ds, config)
        if config.train_enhanced:
            train_and_validate_model_enhanced(train_dl, val_dl, config)
        else:
            train_and_validate_model_standard(train_dl, val_dl, config)


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

    model = models.resnet34(pretrained=True)
    summary(model, input_size=(1, 3, 224, 224))

    main(config)
