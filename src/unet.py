import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt



# ARCHITECTURE
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.middle = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3)) # [N, 512, H, W]

        m = self.middle(self.pool4(d4))

        u4 = self.conv4(torch.cat([self.up4(m), d4], dim=1)) # Upsample, Fuse, Refine
        u3 = self.conv3(torch.cat([self.up3(u4), d3], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.conv1(torch.cat([self.up1(u2), d1], dim=1))
        return self.out(u1)


# DATASET
image_size = 128
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])
target_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.PILToTensor(),
    transforms.Lambda(lambda x: x.long().squeeze(0) - 1),
])

dataset = OxfordIIITPet(
    root="./data", download=True, target_types="segmentation",
    transform=transform, target_transform=target_transform
)

train_len = int(0.8 * len(dataset))
train_ds, val_ds = random_split(dataset, [train_len, len(dataset) - train_len])
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=8)

### TRAIN
def train_model(model, train_dl, val_dl, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device)
            # yb = yb.squeeze(1).to(device)

            # Debugging snippet
            labels = [yb.squeeze().unique() for _, yb in train_dl]
            unique_labels = torch.unique(torch.cat(labels))
            print("Unque labels")
            print(unique_labels)  # Will show all label values present

            yb = yb.squeeze(1).long().to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                # yb = yb.squeeze(1).to(device)
                yb = yb.squeeze(1).long().to(device)
                preds = model(xb)
                total_loss += criterion(preds, yb).item()
        print(f"Epoch {epoch+1}, Val Loss: {total_loss/len(val_dl):.4f}")


def visualize(model, dataset, idx=0, device='cuda'):
    model.eval()
    x, y = dataset[idx]
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).argmax(dim=1).squeeze().cpu().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(x.squeeze().permute(1, 2, 0).cpu())
    ax[0].set_title("Input")
    ax[1].imshow(y.squeeze(), cmap='gray')
    ax[1].set_title("Ground Truth")
    ax[2].imshow(pred, cmap='gray')
    ax[2].set_title("Prediction")
    for a in ax: a.axis('off')
    plt.show()

# RUN
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet(in_channels=3, out_channels=3)
train_model(model, train_dl, val_dl, device)
visualize(model, val_ds, idx=10, device=device)
