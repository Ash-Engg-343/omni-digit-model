# main.py
# Shared-representation model: digit images and digit words map into one shared space.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
D = 64              # size of the shared vector
EPOCHS = 5
BATCH = 128
os.makedirs("outputs", exist_ok=True)

# ---------- Data ----------
tf = transforms.ToTensor()
train_ds = datasets.MNIST("data", train=True, download=True, transform=tf)
test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=256)

# ---------- Modules ----------
class ImageEncoder(nn.Module):
    """28x28 image -> vector of size D"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7x7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, D),
        )

    def forward(self, x):
        return self.net(x)

class ImageDecoder(nn.Module):
    """vector of size D -> 28x28 image"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(D, 64 * 7 * 7)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),  # 14x14
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Sigmoid() # 28x28
        )

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 64, 7, 7))

class OmniDigitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_enc = ImageEncoder()          # image path
        self.txt_emb = nn.Embedding(10, D)     # text path (digit word id -> vector)
        self.shared = nn.Sequential(           # shared core used by both modalities
            nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D)
        )
        self.img_dec = ImageDecoder()          # shared vector -> image output
        self.classifier = nn.Linear(D, 10)     # shared vector -> digit/word output

    def encode_image(self, x):
        return self.shared(self.img_enc(x))

    def encode_text(self, ids):
        return self.shared(self.txt_emb(ids))

# ---------- Training ----------
model = OmniDigitModel().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    for x, y in train_dl:
        x, y = x.to(DEVICE), y.to(DEVICE)

        s_img = model.encode_image(x)   # image -> shared
        s_txt = model.encode_text(y)    # digit word -> shared

        loss_cls = F.cross_entropy(model.classifier(s_img), y)        # image recognition
        loss_txt_cls = F.cross_entropy(model.classifier(s_txt), y)    # word recognition
        loss_rec_img = F.mse_loss(model.img_dec(s_img), x)            # image reconstruction
        loss_gen_txt = F.mse_loss(model.img_dec(s_txt), x)            # word -> image generation
        loss_align = (1 - F.cosine_similarity(s_img, s_txt)).mean()   # same digit, same place

        loss = loss_cls + loss_txt_cls + loss_rec_img + loss_gen_txt + loss_align

        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()

    print(f"epoch {epoch + 1}/{EPOCHS}  loss {total / len(train_dl):.4f}")

torch.save(model.state_dict(), "outputs/model.pt")

# ---------- Evaluation ----------
model.eval()
correct_img = 0
with torch.no_grad():
    for x, y in test_dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct_img += (model.classifier(model.encode_image(x)).argmax(1) == y).sum().item()
    img_acc = correct_img / len(test_ds)

    ids = torch.arange(10, device=DEVICE)
    text_acc = (model.classifier(model.encode_text(ids)).argmax(1) == ids).float().mean().item()

    # Generate an image from each word, then check whether the image path recognizes it
    gen = model.img_dec(model.encode_text(ids))
    cycle_acc = (model.classifier(model.encode_image(gen)).argmax(1) == ids).float().mean().item()
    save_image(gen, "outputs/generated_from_words.png", nrow=10)

print(f"\nImage recognition accuracy (test set): {img_acc:.3f}")
print(f"Word recognition accuracy:             {text_acc:.3f}")
print(f"Word -> image -> recognized accuracy:  {cycle_acc:.3f}")
print("Saved outputs/generated_from_words.png and outputs/model.pt")