import os
from datetime import datetime
from src.config import Config
from src.data_utils import probe_images_labels, pd_train_test_split, log_stats, write_stats
from src.dataloader import BRATS18Dataset
from albumentations import (Compose, CenterCrop, Resize, ShiftScaleRotate, HueSaturationValue)
from albumentations.pytorch.transforms import ToTensor
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# DEBUG
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

# Load configuration
params = Config()

# Create log dir
log_dir = os.path.join(params.log_path, params.name, datetime.now().strftime("%Y%m%d_%H:%M:%S"))
os.makedirs(log_dir)
print("Log dir: '{}'".format(log_dir))

# Load metadata
df = probe_images_labels(params.image_dir, params.label_dir)
train_metadata, test_metadata = pd_train_test_split(df)

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
test_dataset = BRATS18Dataset(test_metadata)


# Dataloaders
train_loader = DataLoader(dataset=train_dataset, num_workers=params.num_workers,
                          batch_size=params.batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, num_workers=params.num_workers,
                         batch_size=params.batch_size, shuffle=False)

# Model
# TODO - add model
model = None

# CUDA
if torch.cuda.is_available():
    if hasattr(params, 'device_id'):
        torch.cuda.set_device(params.device_id)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of params: {}\nTotal number of trainable params: {}".format(total_params, trainable_params))

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
scheduler = StepLR(optimizer, step_size=params.optim_step, gamma=params.optim_factor)

# Loss
# TODO - Dice loss
criterion = None

# Tensorbaord
writer = SummaryWriter(log_dir)

for epoch in range(params.num_epochs):
    epoch_stats = {}
    for fold in ["train", "test"]:
        if fold == "train":
            print("*** Epoch {} Train fold".format(epoch + 1))
            for i, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backwards()
                optimizer.step()

                if scheduler is not None:
                    current_lr = scheduler.get_last_lr()[0]
                else:
                    current_lr = params.lr
                log_stats(epoch_stats, fold, outputs, targets, loss, current_lr)
        else:
            with torch.no_grad():
                for i, (inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader)):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    log_stats(epoch_stats, fold, outputs, targets, loss)

    # Save parameters
    if epoch % params.save_freq == 0:
        save_dir = os.path.join(log_dir, 'model')
        os.makedirs(save_dir, exist_ok=True)
        model_file = os.path.join(save_dir, params.name + '__epoch_{:04d}'.format(epoch))
        torch.save(model.state_dict(), model_file)

    # Write stats to tensorboard
    write_stats(epoch_stats, writer, epoch)

writer.close()
