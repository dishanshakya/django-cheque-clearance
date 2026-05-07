import os
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF


# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture  (mirrors model.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x,  dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        combined = torch.cat([max_pool, avg_pool], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention


class SiameseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1: 96×96 → 96×96 (no pooling)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            SpatialAttention(),
            nn.Dropout2d(0.1),
        )
        # Block 2: 96×96 → 48×48
        self.pool1 = nn.MaxPool2d(2)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            SpatialAttention(),
            nn.Dropout2d(0.15),
        )
        # Block 3: 48×48 → 24×24
        self.pool2 = nn.MaxPool2d(2)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            SpatialAttention(),
            nn.Dropout2d(0.25),
        )
        # Block 4: 24×24 → 12×12
        self.pool3 = nn.MaxPool2d(2)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            SpatialAttention(),
            nn.Dropout2d(0.30),
        )
        # Global pooling: any spatial size → 1×1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Embedding head
        self.embedding = nn.Sequential(
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
        )

    def forward_one(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x1, x2=None):
        emb1 = self.forward_one(x1)
        if x2 is not None:
            emb2 = self.forward_one(x2)
            return emb1, emb2
        return emb1


# ─────────────────────────────────────────────────────────────────────────────
# Transform  (96×96 to match model input, consistent with crop.py output)
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE = 96

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomInvert(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing  (mirrors crop_and_save_signature logic from crop.py)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_signature(img_input, target_size=92, final_size=96, pad_width=6):
    """
    Crops and binarises a signature image using Otsu thresholding,
    matching the crop_and_save_signature pipeline in crop.py.

    Parameters
    ----------
    img_input : str | os.PathLike | np.ndarray | PIL.Image.Image
    target_size : int   inner crop resize (default 92)
    final_size  : int   canvas size (default 96)
    pad_width   : int   padding added before Otsu detection (default 6)

    Returns
    -------
    PIL.Image.Image  (grayscale, final_size × final_size, white background)
    """
    # ── Load to grayscale numpy array ──────────────────────────────────────
    if isinstance(img_input, (str, os.PathLike)):
        arr = cv2.imread(str(img_input), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise FileNotFoundError(f"Cannot read image: {img_input}")
    elif isinstance(img_input, np.ndarray):
        arr = img_input if img_input.ndim == 2 else cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    elif isinstance(img_input, Image.Image):
        arr = np.array(img_input.convert("L"))
    else:
        raise TypeError(f"Unsupported input type: {type(img_input)}")

    # ── Pad with white before Otsu detection (reduces edge-clipping) ────────
    padded = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=255)

    _, binary = cv2.threshold(padded, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.where(binary > 4)
    if len(coords[0]) == 0:
        # Fallback: blank / unreadable image — return resized original
        pil = Image.fromarray(arr)
        return pil.resize((final_size, final_size), Image.Resampling.LANCZOS)

    # ── Bounding box in padded space, mapped back to original ──────────────
    y0 = coords[0].min() - pad_width
    y1 = coords[0].max() + 1 - pad_width
    x0 = coords[1].min() - pad_width
    x1 = coords[1].max() + 1 - pad_width

    # Add 10 % margin
    h, w = y1 - y0, x1 - x0
    pad_y, pad_x = int(h * 0.1), int(w * 0.1)
    y0 = max(0, y0 - pad_y)
    y1 = min(arr.shape[0], y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(arr.shape[1], x1 + pad_x)

    # ── Crop → light Gaussian blur → re-binarise with Otsu ─────────────────
    cropped = arr[y0:y1, x0:x1]
    blurred = cv2.GaussianBlur(cropped, (3, 3), 0)
    _, binary_crop = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Invert so signature is dark on white background
    sig_img = Image.fromarray(255 - binary_crop)

    # ── Resize to target_size and centre on a white final_size canvas ──────
    sig_img = sig_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (final_size, final_size), 255)
    paste_x = (final_size - target_size) // 2
    paste_y = (final_size - target_size) // 2
    canvas.paste(sig_img, (paste_x, paste_y))

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# SCNN  — main inference class
# ─────────────────────────────────────────────────────────────────────────────

class SCNN:
    """
    Siamese CNN wrapper for signature verification.

    Usage
    -----
    scnn = SCNN(model_path="best_siamese_....pt")
    pred = scnn.predict(img_array_1, img_array_2)
    # pred == 0 → Genuine,  pred == 1 → Forgery
    """

    # Default threshold from test.py
    DEFAULT_THRESHOLD = 0.7786

    def __init__(self, model_path: str | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = SiameseCNN().to(self.device)

        if model_path:
            ckpt = torch.load(model_path, map_location=self.device)
            # Checkpoint is a training snapshot dict — extract just the weights.
            # Falls back to treating the file as a bare state_dict if "model_state"
            # is not present (e.g. older saves).
            state = ckpt.get("model_state", ckpt)
            self.model.load_state_dict(state)

            # Also pick up the saved threshold if the checkpoint carries one
            # (can still be overridden per-call via the threshold= argument).
            if "threshold" in ckpt:
                self.DEFAULT_THRESHOLD = ckpt["threshold"]
                print(f"Loaded threshold from checkpoint: {self.DEFAULT_THRESHOLD:.4f}")

        self.model.eval()

    # ------------------------------------------------------------------
    def _to_tensor(self, img_input) -> torch.Tensor:
        """Preprocess → PIL → transform → (1, 1, H, W) tensor."""
        pil = preprocess_signature(img_input)
        return transform(pil).unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    def predict(self, img1_input, img2_input, threshold: float | None = None) -> int:
        """
        Compare two signature images.

        Parameters
        ----------
        img1_input, img2_input : str | Path | np.ndarray | PIL.Image.Image
            Reference and query signatures respectively.
        threshold : float, optional
            Euclidean-distance threshold.  Values *below* this → Genuine (0).
            Defaults to SCNN.DEFAULT_THRESHOLD (0.7786).

        Returns
        -------
        int
            0 = Genuine,  1 = Forgery
        """
        if threshold is None:
            threshold = self.DEFAULT_THRESHOLD

        x1 = self._to_tensor(img1_input)
        x2 = self._to_tensor(img2_input)

        with torch.no_grad():
            emb1, emb2 = self.model(x1, x2)
            distance = F.pairwise_distance(emb1, emb2).item()

        pred   = 0 if distance < threshold else 1
        label  = "Genuine" if pred == 0 else "Forgery"
        print(f"distance={distance:.4f}  threshold={threshold}  →  {label}")
        return pred