import wandb
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn

from tqdm import tqdm
from dataclasses import dataclass
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, Subset, random_split


seed = 1337
torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    project : str = "unet-oxford-pet"
    name : str = "unet-run"
    lr : float = 1e-3
    image_size : int = 128
    batch_size : int = 16
    n_epochs : int = 10

    sample_size : int = 100
    test_run : bool = True
    wandb_active : str = 'disabled'

    # test_run : bool = False
    #wandb_active : str = 'online'


class DoubleConv(nn.Module):
    def __init__(
            self, in_channels,
            out_channels,
    ):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
    ):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64) # -> (64, 128, 128)
        self.pool1 = nn.MaxPool2d(2) # -> (64, 64, 64)
        self.down2 = DoubleConv(64, 128) # -> (128, 64, 64)
        self.pool2 = nn.MaxPool2d(2) # -> (128, 32, 32)
        self.down3 = DoubleConv(128, 256) # -> (256, 32, 32)
        self.pool3 = nn.MaxPool2d(2) # -> (256, 16, 16)
        self.down4 = DoubleConv(256, 512) # -> (512, 16, 16)
        self.pool4 = nn.MaxPool2d(2) # -> (512, 8, 8)

        self.middle = DoubleConv(512, 1024) # -> (1024, 8, 8)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2) # (512, 16, 16)
        self.conv4 = DoubleConv(1024, 512) # (512, 16, 16)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2) # (256, 32, 32)
        self.conv3 = DoubleConv(512, 256) # (256, 32, 32)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2) # (128, 64, 64)
        self.conv2 = DoubleConv(256, 128) # (128, 64, 64)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2) # (64, 128, 128)
        self.conv1 = DoubleConv(128, 64) # (64, 128, 128)

        self.out = nn.Conv2d(64, 3, 1) # (3, 128, 128)
    
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))

        m = self.middle(self.pool4(d4))

        u4 = self.conv4(torch.cat([self.up4(m), d4], dim=1))
        u3 = self.conv3(torch.cat([self.up3(u4), d3], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.conv1(torch.cat([self.up1(u2), d1], dim=1))

        return self.out(u1)

def dice_score(preds, labels, num_classes=3, eps=1e-6):
    """Computes mean Dice score for a batch."""
    print(f"preds.shape {preds.shape}")
    preds = torch.argmax(preds, dim=1)  # [B, H, W]
    dice = 0.0
    for c in range(num_classes):
        pred_c = (preds == c).float()
        label_c = (labels == c).float()
        intersection = (pred_c * label_c).sum()
        union = pred_c.sum() + label_c.sum()
        dice += (2. * intersection + eps) / (union + eps)
    return dice / num_classes

def train_and_validate_model(config : Config):
    wandb.init(
        project=config.project,
        name=config.name,
        config=vars(config),
        mode=config.wandb_active
    )
    transform = T.Compose([
        T.Resize((config.image_size, config.image_size)),
        # T.RandomHorizontalFlip(),
        # T.ColorJitter(),
        # T.RandomResizedCrop(config.image_size, scale=(0.8, 1.0)),
        T.ToTensor(),
    ])
    target_transform = T.Compose([
        T.Resize((config.image_size, config.image_size)),
        T.PILToTensor(),
        T.Lambda(lambda x: x.long().squeeze(0) - 1)
    ])

    dataset = OxfordIIITPet(
        root='/Users/justinbarry/projects/unet.bk/src/data/',
        download=False,
        target_types='segmentation',
        transform=transform,
        target_transform=target_transform
    )
    
    import random
    if config.test_run:
        indices = random.sample(range(len(dataset)), config.sample_size)
        dataset = Subset(dataset, indices)

    train_len = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_len, len(dataset) - train_len])
    train_dl = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = UNet(in_channels=3, out_channels=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)


    for epoch in range(1, config.n_epochs+1):
        tqdm.write(f"Epoch {epoch}/{config.n_epochs+1}")
        model.train()
        with tqdm(train_dl, desc="Training") as pbar:
            train_loss = 0
            train_correct = 0.0
            train_total = 0
            for xb, yb in pbar:
                xb = xb.to(device)
                yb = yb.squeeze(1).long().to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, preds = torch.max(logits, 1)

                train_loss += loss.item() * yb.numel()
                train_correct += (preds == yb).sum().item()
                train_total += yb.numel()
                pbar.set_postfix(loss=loss.item())

            train_epoch_loss = train_loss / train_total
            train_epoch_acc = train_correct / train_total
            tqdm.write(f"Train Loss {train_epoch_loss:.4f} Train Acc {train_epoch_acc:.2f}")

        model.eval()
        with tqdm(val_dl, desc="Validation") as pbar:
            with torch.no_grad():
                val_loss = 0
                val_correct = 0.0
                val_total = 0
                for xb, yb in pbar:
                    xb = xb.to(device)
                    yb = yb.squeeze(1).long().to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)

                    _, preds = torch.max(logits, 1)
                    val_correct += (preds == yb).sum().item()
                    val_loss += loss.item() * yb.numel()
                    val_total += yb.numel()
                    pbar.set_postfix(loss=loss.item())

                val_epoch_loss = val_loss / val_total
                val_epoch_acc = val_correct / val_total
                tqdm.write(f"Val Loss {val_epoch_loss:.4f} Val Acc {val_epoch_acc:.2f}")
                pbar.set_postfix(loss=loss.item())

        wandb.log({
            "epoch": epoch,
            "train/loss": train_epoch_loss,
            "train/acc": train_epoch_acc,
            "val/loss": val_epoch_loss,
            "val/acc": val_epoch_acc,
            "lr": optimizer.param_groups[0]['lr']
        })
    wandb.finish()


def main():
    config = Config()
    train_and_validate_model(config)
    
if __name__ == "__main__":
    main()






