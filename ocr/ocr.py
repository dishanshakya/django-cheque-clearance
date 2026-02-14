from .crnn import CRNN, MICRNN, XCRNN
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

class MICROCR:
    MICR = "0123456789ABCD"

    micr2idx = {c: i for i, c in enumerate(MICR)}  # CTC blank = 0
    idx2micr = {i: c for c, i in micr2idx.items()}

    def __init__(self, model_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = MICRNN(num_class=len(MICROCR.MICR)+1).to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])

        self.model.eval()

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is None:
            raise FileNotFoundError(f"Image not found")

        h, w = img.shape
        img_height = 32
        new_w = int(w * (img_height / h))
        img_resized = cv2.resize(img, (new_w, img_height), interpolation=cv2.INTER_LINEAR)

        img_resized = cv2.adaptiveThreshold(img_resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        img_resized = cv2.GaussianBlur(img_resized, (3, 3), 0)


        # Scale to [0,1] float32
        img_resized = img_resized.astype(np.float32) / 255.0

        # Convert to tensor and add channel dimension
        img_tensor = torch.from_numpy(img_resized).unsqueeze(0)  # (1,H,W)
        img_tensor = img_tensor.unsqueeze(0)  # (1,1,H,W)

        return img_tensor

    def greedy_decoder(self, output, blank=0):
        blank = len(MICROCR.MICR)
        output = output.softmax(2)
        max_indices = output.argmax(2).permute(1, 0)
        decoded = []
        for indices in max_indices:
            s = ""
            prev = blank
            for i in indices:
                if i != prev and i != blank:
                    s += MICROCR.idx2micr.get(i.item(), '?')
                prev = i
            decoded.append(s)
        return decoded[0]

    def predict(self, img, bias=False):
        tensor = self.preprocess(img)
        with torch.no_grad():
            output = self.model(tensor.to(self.device))
            text = self.greedy_decoder(output)
            print(text)
            return text


class EnglishOCR:
    ENG = list(
        "abcdefghijklmnopqrstuvwxyz" +    # English lowercase
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ" +    # English uppercase
        "0123456789" +                    # English digits
        ".,!?;:'\"()[]{}<>/@#$%^&*+=" + # Common punctuation & symbols
        " "
    )
    digit_indices = [i for i, c in enumerate(ENG) if c.isdigit()]

    eng2idx = {c: i + 1 for i, c in enumerate(ENG)}  # CTC blank = 0
    idx2eng = {i: c for c, i in eng2idx.items()}

    def __init__(self, model_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = XCRNN(num_class=len(EnglishOCR.ENG)+1).to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])

        self.model.eval()

    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced = Image.fromarray(gray)
        fixed_height = 48

        def resize_keep_ratio(image, target_height=fixed_height):
            w, h = image.size
            new_w = int(w * target_height / h)
            return image.resize((new_w, target_height), Image.LANCZOS)

        transform = transforms.Compose([
            transforms.Lambda(resize_keep_ratio),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        img_tensor = transform(enhanced).unsqueeze(0)  # add batch dimension
        return img_tensor

    def predict(self, img, num_bias=0):
        tensor = self.preprocess(img)
        with torch.no_grad():
            output = self.model(tensor.to(self.device))
            if num_bias:
                output[..., EnglishOCR.digit_indices] += num_bias

            text = self.greedy_decoder(output)
            print(text)
            return text

    def greedy_decoder(self, output, blank=0):
        output = output.softmax(2)
        max_indices = output.argmax(2).permute(1, 0)
        decoded = []
        for indices in max_indices:
            s = ""
            prev = blank
            for i in indices:
                if i != prev and i != blank:
                    s += EnglishOCR.idx2eng.get(i.item(), '?')
                prev = i
            decoded.append(s)
        return decoded[0]

class NepaliOCR:
    ENG = list(
        "अआइईउऊऋएऐओऔ" +
        'कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह' +
        "ा"+
        'ि' +
        'ी' +
        'ु' +
        'ू' +
        'े' +
        'ै' +
        'ो' +
        'ौ' +
        'ं' +
        'ः' +
        'ँ' +
        'ृ' +
        '्' +
        "०१२३४५६७८९" +
        ".,!?;:'\"-()[]{}<>/@#$%^&*+=" +
        " "
    )


    digit_indices = [
        i for i, c in enumerate(ENG)
        if c in set("०१२३४५६७८९")
    ]

    eng2idx = {c: i + 1 for i, c in enumerate(ENG)}  # CTC blank = 0
    idx2eng = {i: c for c, i in eng2idx.items()}

    def __init__(self, model_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = XCRNN(num_class=len(NepaliOCR.ENG)+1).to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])

        self.model.eval()

    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced = Image.fromarray(gray)
        fixed_height = 64

        def resize_keep_ratio(image, target_height=fixed_height):
            w, h = image.size
            new_w = int(w * target_height / h)
            return image.resize((new_w, target_height), Image.LANCZOS)

        transform = transforms.Compose([
            transforms.Lambda(resize_keep_ratio),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        img_tensor = transform(enhanced).unsqueeze(0)  # add batch dimension
        return img_tensor

    def predict(self, img, num_bias=0):
        tensor = self.preprocess(img)
        with torch.no_grad():
            output = self.model(tensor.to(self.device))
            if num_bias:
                output[..., NepaliOCR.digit_indices] += num_bias
                text = self.greedy_decoder(output)
                text = text.replace('व', '१')
                text = text.replace('थ', '१')
            else:
                text = self.greedy_decoder(output)
            print(text)
            return text

    def greedy_decoder(self, output, blank=0):
        output = output.softmax(2)
        max_indices = output.argmax(2).permute(1, 0)
        decoded = []
        for indices in max_indices:
            s = ""
            prev = blank
            for i in indices:
                if i != prev and i != blank:
                    s += NepaliOCR.idx2eng.get(i.item(), '?')
                prev = i
            decoded.append(s)
        return decoded[0]



