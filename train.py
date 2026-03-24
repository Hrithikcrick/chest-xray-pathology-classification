"""Run as
python3 train.py --data_path path-to-train.csv --model_out_path path-to-checkpoint-directory --model [optional,by default all]
See the main function for the choices of optional model argument"""
import os
import time
import copy
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from model import build_effnet_b0, build_convnext_tiny, build_deit3_small


# CONFIG
SEED = 42
NUM_WORKERS = 2


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# DATASET
class TrainDataset(Dataset):

    def __init__(self, df, image_dir, transform):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_path = os.path.join(self.image_dir, row["id"])
        img = Image.open(img_path).convert("RGB")

        img = self.transform(img)

        label = int(row["label_idx"])

        return img, label


# TRANSFORMS
def make_train_tf(size):

    interp = InterpolationMode.BICUBIC if size >= 320 else InterpolationMode.BILINEAR

    return transforms.Compose([
        transforms.Resize((size, size), interpolation=interp),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        ),
    ])


def make_valid_tf(size):

    interp = InterpolationMode.BICUBIC if size >= 320 else InterpolationMode.BILINEAR

    return transforms.Compose([
        transforms.Resize((size, size), interpolation=interp),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        ),
    ])


# LOADERS
def make_loaders(train_df, valid_df, img_dir, size, batch):

    train_ds = TrainDataset(train_df, img_dir, make_train_tf(size))
    valid_ds = TrainDataset(valid_df, img_dir, make_valid_tf(size))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, valid_loader


# TRAIN
def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()

    running_loss = 0
    y_true = []
    y_pred = []

    for x,y in loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(logits,y)

        loss.backward()

        optimizer.step()

        preds = torch.argmax(logits,1)

        running_loss += loss.item()*x.size(0)

        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    loss = running_loss/len(loader.dataset)

    f1 = f1_score(y_true,y_pred,average="macro")

    return loss,f1

# VALIDATE
@torch.no_grad()
def validate(model,loader,criterion,device):

    model.eval()

    running_loss = 0

    y_true=[]
    y_pred=[]

    for x,y in loader:

        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = criterion(logits,y)

        preds = torch.argmax(logits,1)

        running_loss += loss.item()*x.size(0)

        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    loss = running_loss/len(loader.dataset)

    acc = accuracy_score(y_true,y_pred)

    f1 = f1_score(y_true,y_pred,average="macro")

    return loss,acc,f1


# FIT MODEL
def fit_model(model,train_loader,valid_loader,save_path,epochs,lr,device):

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)

    best_f1 = -1

    best_state=None

    for epoch in range(epochs):

        start=time.time()

        train_loss,train_f1 = train_one_epoch(
            model,train_loader,optimizer,criterion,device
        )

        val_loss,val_acc,val_f1 = validate(
            model,valid_loader,criterion,device
        )

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f}"
        )

        if val_f1>best_f1:

            best_f1 = val_f1

            best_state = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "model_state_dict":best_state,
                    "best_val_f1":best_f1
                },
                save_path
            )

            print("Saved:",save_path)

    model.load_state_dict(best_state)

    return model,best_f1


# MAIN
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_out_path", required=True)

    parser.add_argument("--model",default="all",
                        choices=["effnet","convnext","deit","all"])

    args = parser.parse_args()


    set_seed(SEED)

    train_csv = args.data_path

    root_dir = os.path.dirname(train_csv)

    IMG_DIR = os.path.join(root_dir,"images","images")

    CKPT_DIR = args.model_out_path

    os.makedirs(CKPT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:",device)

    train_df = pd.read_csv(train_csv)

    label_cols = list(train_df.columns[1:])

    num_classes = len(label_cols)

    train_df["label_idx"] = train_df[label_cols].values.argmax(axis=1)

    tr_df,va_df = train_test_split(
        train_df,
        test_size=0.15,
        stratify=train_df["label_idx"],
        random_state=SEED
    )

    # EfficientNet
    if args.model in ["effnet","all"]:

        print("Training EfficientNet")

        train_loader,valid_loader = make_loaders(
            tr_df,va_df,IMG_DIR,288,32
        )

        model = build_effnet_b0(num_classes)

        save_path = os.path.join(
            CKPT_DIR,"best_model_singlelabel.pth"
        )

        fit_model(
            model,
            train_loader,
            valid_loader,
            save_path,
            epochs=15,
            lr=1e-4,
            device=device
        )


    # ConvNeXt
    if args.model in ["convnext","all"]:

        print("Training ConvNeXt")

        train_loader,valid_loader = make_loaders(
            tr_df,va_df,IMG_DIR,320,32
        )

        model = build_convnext_tiny(num_classes)

        save_path = os.path.join(
            CKPT_DIR,"best_model_convnext_tiny_320.pth"
        )

        fit_model(
            model,
            train_loader,
            valid_loader,
            save_path,
            epochs=10,
            lr=1e-4,
            device=device
        )


    # DeiT Transformer
    if args.model in ["deit","all"]:

        print("Training DeiT")

        train_loader,valid_loader = make_loaders(
            tr_df,va_df,IMG_DIR,224,16
        )

        model = build_deit3_small(num_classes)

        save_path = os.path.join(
            CKPT_DIR,"best_deit3_small_singlelabel.pth"
        )

        fit_model(
            model,
            train_loader,
            valid_loader,
            save_path,
            epochs=25,
            lr=3e-5,
            device=device
        )


if __name__=="__main__":
    main()
