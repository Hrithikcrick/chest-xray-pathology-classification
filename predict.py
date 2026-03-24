"""Run as 
python3 predict.py --data_path path-to-test-set --model_path path-to-checkpoint-directory --output submission.csv
Note:checkpoint directory has 3 model checkpoints.
We provide the path to checkpoint directory as we will ensemble the 3 models,not to an .pth file"""
import os
import gc
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from model import (
    build_effnet_b0,
    build_convnext_tiny,
    build_deit3_small,
    get_state_dict,
)



class TestDataset(Dataset):
    def __init__(self, df, image_dir, transform):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]["id"]
        img_path = os.path.join(self.image_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        return self.transform(image)


def make_eff_tf():
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])


def make_conv_tf():
    return transforms.Compose([
        transforms.Resize((320, 320), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])


def make_deit_tf():
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])


def load_label_cols(train_csv):
    train_df = pd.read_csv(train_csv)
    return list(train_df.columns[1:])


@torch.no_grad()
def predict_probs(model, loader, device):
    model = model.to(device)
    model.eval()

    probs_all = []

    for images in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        probs_all.append(probs)

    return np.concatenate(probs_all, axis=0)


def predict_effnet(loader, ckpt_path, num_classes, device):
    gc.collect()
    torch.cuda.empty_cache()

    model = build_effnet_b0(num_classes)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(get_state_dict(ckpt))

    probs = predict_probs(model, loader, device)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return probs


def predict_convnext(loader, ckpt_path, num_classes, device):
    gc.collect()
    torch.cuda.empty_cache()

    model = build_convnext_tiny(num_classes)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(get_state_dict(ckpt))

    probs = predict_probs(model, loader, device)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return probs


def predict_deit(loader, ckpt_path, num_classes, device):
    gc.collect()
    torch.cuda.empty_cache()

    model = build_deit3_small(num_classes)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(get_state_dict(ckpt))

    probs = predict_probs(model, loader, device)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                        help="Path to test images directory")

    parser.add_argument("--model_path", required=True,
                        help="Checkpoint directory")

    parser.add_argument("--output", default="submission.csv")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    img_dir = args.data_path
    root_dir = os.path.dirname(img_dir)

    test_csv = os.path.join(root_dir, "test.csv")
    train_csv = os.path.join(root_dir, "train.csv")

    eff_path = os.path.join(args.model_path, "best_model_singlelabel.pth")
    conv_path = os.path.join(args.model_path, "best_model_convnext_tiny_320.pth")
    deit_path = os.path.join(args.model_path, "best_deit3_small_singlelabel.pth")

    test_df = pd.read_csv(test_csv)
    label_cols = load_label_cols(train_csv)

    num_classes = len(label_cols)


    print("Num classes:", num_classes)
    print("Test size:", len(test_df))

    eff_loader = DataLoader(
        TestDataset(test_df, img_dir, make_eff_tf()),
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    conv_loader = DataLoader(
        TestDataset(test_df, img_dir, make_conv_tf()),
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    deit_loader = DataLoader(
        TestDataset(test_df, img_dir, make_deit_tf()),
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    print("Predicting EfficientNet...")
    eff_probs = predict_effnet(
        eff_loader,
        eff_path,
        num_classes,
        device
    )

    print("Predicting ConvNeXt...")
    conv_probs = predict_convnext(
        conv_loader,
        conv_path,
        num_classes,
        device
    )

    print("Predicting DeiT...")
    deit_probs = predict_deit(
        deit_loader,
        deit_path,
        num_classes,
        device
    )

    print("Shapes:")
    print("EfficientNet:", eff_probs.shape)
    print("ConvNeXt    :", conv_probs.shape)
    print("DeiT        :", deit_probs.shape)

    ensemble_probs = (
        0.25 * eff_probs +
        0.50 * conv_probs +
        0.25 * deit_probs
    )

    best_idx = np.argmax(ensemble_probs, axis=1)

    onehot = np.zeros((len(test_df), num_classes), dtype=np.int64)
    onehot[np.arange(len(test_df)), best_idx] = 1

    submission = pd.DataFrame(onehot, columns=label_cols)
    submission.insert(0, "id", test_df["id"].values)

    row_sums = submission.drop(columns=["id"]).sum(axis=1)
    bad_rows = int((row_sums != 1).sum())

    print("Bad rows:", bad_rows)
    if bad_rows != 0:
        raise ValueError("Submission is not one-hot for all rows.")

    submission.to_csv(args.output, index=False)
    print(f"Saved submission to: {args.output}")
    print(submission.head())


if __name__ == "__main__":
    main()
