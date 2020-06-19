import os
import random
import pickle
import numpy as np
from datetime import datetime
from config import Config
from data_utils import probe_images_labels, pd_train_test_split, log_stats, write_stats, visualize_batch
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

from models.unet.unet import UNet

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

    # Load metadata
    df = probe_images_labels(params.image_dir, params.label_dir)
    train_metadata, test_metadata = pd_train_test_split(df, random_state=params.seed)

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

    print("Test set:")
    test_dataset = BRATS18Dataset(test_metadata, transforms=test_transforms)


    # Dataloaders
    train_loader = DataLoader(dataset=train_dataset, num_workers=params.num_workers,
                              batch_size=params.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, num_workers=params.num_workers,
                             batch_size=params.batch_size, shuffle=False)

    # Model
    model = UNet(in_channels=3, out_channels=params.num_classes, init_features=32,
                 pretrained=params.pretrained, weights=params.model_weights)

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

    # Tensorbaord
    writer = SummaryWriter(log_dir)

    for epoch in range(params.num_epochs):
        epoch_stats = {}
        for fold in ["train", "test"]:
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

                    log_stats(epoch_stats, fold, outputs, targets, loss, current_lr)
            elif fold == "test":
                with torch.no_grad():
                    for i, (inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader)):
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        log_stats(epoch_stats, fold, outputs, targets, loss)

        if scheduler is not None:
            scheduler.step()

        # Save model parameters
        if epoch % params.save_freq == 0:
            save_dir = os.path.join(log_dir, 'model')
            os.makedirs(save_dir, exist_ok=True)
            model_file = os.path.join(save_dir, params.name + '__epoch_{:04d}'.format(epoch))
            torch.save(model.state_dict(), model_file)

        # Write stats to tensorboard
        write_stats(epoch_stats, writer, epoch)

    writer.close()
