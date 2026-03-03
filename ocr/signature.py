import os
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

dropout=0.2

class SignatureEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),


            nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),


            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 194, 3, padding=1),
            nn.BatchNorm2d(194),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),


            nn.Conv2d(194, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)  # REQUIRED for cosine
        return x
    
class SiameseNet(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.embedding_net = SignatureEmbeddingNet(embedding_dim)

    def forward(self, x1, x2):
        z1 = self.embedding_net(x1)
        z2 = self.embedding_net(x2)
        return z1, z2

class ResizePadSquare:
    def __init__(self, size, fill=255):
        self.size = size
        self.fill = fill  # white for grayscale PIL

    def __call__(self, img: Image.Image):
        w, h = img.size
        scale = self.size / max(w, h)
        nw, nh = int(w * scale), int(h * scale)

        img = TF.resize(img, (nh, nw))

        pad_w = self.size - nw
        pad_h = self.size - nh

        padding = (
            pad_w // 2,
            pad_h // 2,
            pad_w - pad_w // 2,
            pad_h - pad_h // 2,
        )

        img = TF.pad(img, padding, fill=self.fill)
        return img

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    ResizePadSquare(256, fill=255),   # white background
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class SCNN(nn.Module):
    def __init__(self, model_path=None):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SiameseNet().to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.eval()

    def preprocess_signature(self, img_path, blur_ksize=3, thresh_block=31, thresh_C=10):
        if isinstance(img_path, (str, os.PathLike)):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        elif isinstance(img_path, Image.Image):
            img = np.array(img_path)

        img_blur = cv2.GaussianBlur(img, (7, 7), 0)

        img_thresh = cv2.adaptiveThreshold(
            img_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            thresh_block,
            thresh_C
        )
        img_thresh = cv2.GaussianBlur(img_thresh, (blur_ksize, blur_ksize), 0)

        pil_img = Image.fromarray(img_thresh).convert("L")
        return pil_img

    def predict(self, img_path1, img_path2, threshold=0.8):
        # Load images

        img1 = Image.fromarray(img_path1).convert("L")
        img2 = Image.fromarray(img_path2).convert("L")

        img1 = self.preprocess_signature(img1)
        img2 = self.preprocess_signature(img2)

        x1 = transform(img1).unsqueeze(0).to(self.device)  # add batch dim
        x2 = transform(img2).unsqueeze(0).to(self.device)
        # Forward pass
        with torch.no_grad():
            z1, z2 = self.model(x1, x2)
            sim = F.cosine_similarity(z1, z2).item()

        # Prediction based on label convention: 0 = genuine, 1 = forgery
        pred = 1 if sim < threshold else 0
        print('similarity', sim, 'pred', pred)
        return pred

