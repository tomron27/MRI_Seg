import os
import random
import pickle
import numpy as np
from datetime import datetime
from config import Config
from data_utils import probe_images_labels, pd_train_test_val_split, log_stats, write_stats, \
    save_best_model, visualize_batch
from dataloader import BRATS18Dataset
from albumentations import (Compose, CenterCrop, Resize, ShiftScaleRotate, HueSaturationValue)
from albumentations.pytorch.transforms import ToTensor
import torch
from torch.utils.data import DataLoader
from loss import DiceLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from get_models import get_model

# DEBUG
# import matplotlib
# matplotlib.use('TkAgg')
# from matplotlib import pyplot as plt


if __name__ == "__main__":

    # Load configuration
    params = Config()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(params.device_id)

    # Set seed
    random.seed(params.seed)
    np.random.seed(params.seed)

    # Create log dir
    log_dir = os.path.join(params.log_path, params.name, datetime.now().strftime("%Y%m%d_%H:%M:%S"))
    os.makedirs(log_dir)
    print("Log dir: '{}'".format(log_dir))

    # Save configuration
    pickle.dump(params, open(os.path.join(log_dir, "params.p"), "wb"))

    # Load metadata and split
    df = probe_images_labels(params.image_dir, params.label_dir)
    split_df = pd_train_test_val_split(df, random_state=params.seed, train_frac=params.train_frac)

    # Save metadata split
    split_df.to_csv(os.path.join(log_dir, "metadata.csv"))
    print("Saved metadata split to '{}'".format(os.path.join(log_dir, "metadata.csv")))
    train_metadata = split_df[split_df['fold'] == "train"]
    val_metadata = split_df[split_df['fold'] == "val"]
    test_metadata = split_df[split_df['fold'] == "test"]

    # Define transforms
    train_transforms = Compose(
        [
            Resize(*params.scale_size),
            CenterCrop(*params.crop_size),
            ShiftScaleRotate(),
            HueSaturationValue(),
            ToTensor()
         ]
    )

    test_transforms = Compose(
        [
            Resize(*params.scale_size),
            CenterCrop(*params.crop_size),
            ToTensor()
         ]
    )

    print("Train set:")
    train_dataset = BRATS18Dataset(train_metadata, transforms=train_transforms)

    print("Validation set:")
    val_dataset = BRATS18Dataset(val_metadata, transforms=test_transforms)

    print("Test set:")
    test_dataset = BRATS18Dataset(test_metadata, transforms=test_transforms)

    # Dataloaders
    train_loader = DataLoader(dataset=train_dataset, num_workers=params.num_workers,
                              batch_size=params.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, num_workers=params.num_workers,
                            batch_size=params.batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, num_workers=params.num_workers,
                             batch_size=params.batch_size, shuffle=False)

    # Model
    model = get_model(params)

    # CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of params: {}\nTotal number of trainable params: {}".format(total_params, trainable_params))

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    scheduler = StepLR(optimizer, step_size=params.optim_step, gamma=params.optim_factor)

    # Loss
    criterion = DiceLoss(smooth=params.loss_smooth)

    # Tensorbaord writers
    train_writer = SummaryWriter(os.path.join(log_dir, "train"))
    val_writer = SummaryWriter(os.path.join(log_dir, "validation"))

    # Training
    print("******* Training *******")
    best_score = 0.0
    for epoch in range(params.num_epochs):
        train_stats, val_stats = {}, {}
        for fold in ["train", "val"]:
            print("*** Epoch {} - {} ***".format(epoch + 1, fold))
            if fold == "train":
                for i, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
                    inputs, targets = inputs.to(device), targets.to(device)
                    # if i == 0:
                    #     visualize_batch(inputs, targets, epoch)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if scheduler is not None:
                        current_lr = scheduler.get_last_lr()[0]
                    else:
                        current_lr = params.lr

                    log_stats(train_stats, outputs, targets, loss, current_lr)
            elif fold == "val":
                with torch.no_grad():
                    for i, (inputs, targets) in tqdm(enumerate(val_loader), total=len(val_loader)):
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        log_stats(val_stats, outputs, targets, loss)

        # Progress optimizer scheduler
        if scheduler is not None:
            scheduler.step()

        # Write stats to tensorboard
        write_stats(train_stats, train_writer, epoch)
        write_stats(val_stats, val_writer, epoch)

        # Save best validation model
        best_score = save_best_model(model, val_stats, best_score, log_dir, params.name, score="dice_score")

    train_writer.close()
    val_writer.close()

    # Evaluation on test set
    print("******* Testing *******")
    model_file = os.path.join(log_dir, 'model', params.name + '__best')
    model.load_state_dict(torch.load(model_file))
    test_stats = {}
    with torch.no_grad():
        for i, (inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            log_stats(test_stats, outputs, targets, loss)

    avg_loss = sum(test_stats['loss']) / len(test_stats['loss'])
    avg_dice = sum(test_stats['dice_score']) / len(test_stats['dice_score'])

    print("Avg test loss: {}".format(avg_loss))
    print("Avg test dice score: {}".format(avg_dice))
