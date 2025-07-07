import argparse
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from models.twt import TwT


def load_image(path: str, img_size: int = 32):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    image = Image.open(path).convert('RGB')
    return transform(image).unsqueeze(0)  # (1,3,H,W)


def main():
    parser = argparse.ArgumentParser(description='Inference using Trained TwT model')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--img-size', type=int, default=32)
    parser.add_argument('--num-classes', type=int, default=2)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = TwT(num_classes=args.num_classes)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    img_tensor = load_image(args.image, args.img_size).to(device)
    with torch.no_grad():
        logits = model.inference(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
    print('Predicted class:', pred)
    print('Probabilities:', probs.squeeze().cpu().numpy())


if __name__ == '__main__':
    main()
