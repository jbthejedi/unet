import wandb
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn

from tqdm import tqdm
from dataclasses import dataclass
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, random_split


seed = 1337
torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    name : str = "unet-oxford-pet"
    run : str = "unet-run"
    lr : float = 1e-3
    image_size : int = 128
    batch_size : int = 4
    n_epochs : int = 2


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

                train_loss = loss.item() * xb.size(0)
                _, preds = torch.max(logits, 1)
                train_correct += (preds == yb).sum().item()
                train_total += xb.size(0)
            train_epoch_loss += train_loss / train_total
            train_epoch_acc += train_correct / train_total
            pbar.set_postfix(f"Train Loss {train_epoch_loss} Train Acc {train_epoch_acc}")

        model.eval()
        with tqdm(val_dl, desc="Validation") as pbar:
            with torch.no_grad():
                for xb, yb in pbar:
                    xb = xb.to(device)
                    yb = yb.squeeze(1).long().to(device)
                    logits = model(xb)
                    total_loss += criterion(preds, yb).item()
            pbar.set_postfix(f"Val Loss {total_loss}")



def main():
    config = Config()
    train_and_validate_model(config)
    
if __name__ == "__main__":
    main()






