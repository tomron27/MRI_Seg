import os
import random
import pickle
import numpy as np
import pandas as pd
from data_utils import probe_images_labels, pd_train_test_val_split, log_stats
from dataloader import BRATS18Dataset
from albumentations import (Compose, CenterCrop, Resize)
from albumentations.pytorch.transforms import ToTensor
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from get_models import get_model


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = str(2)

    # Paths and configurations
    seed = 42
    train_frac = 0.6
    scale_size = (256, 256)
    crop_size = (224, 224)
    num_workers = 4
    batch_size = 64
    log_dir = "/home/tomron27/projects/MRI_Seg/logs"
    image_dir = "/home/tomron27/datasets/BraTS18/proc/train/Database Images/Full - Train/"
    label_dir = "/home/tomron27/datasets/BraTS18/proc/train/Database Images/Full - Single/"

    model_paths = {"unet_no_pt":
                  ("unet", os.path.join(log_dir, "unet_no_pt/20200621_11:17:52/")),
              "unet_resnet34_no_pt":
                  ("unet_resnet34", os.path.join(log_dir, "unet_resnet34_no_pt/20200621_19:19:24/")),
              "unet_resnet34_pt":
                  ("unet_resnet34", os.path.join(log_dir, "unet_resnet34_pt/20200622_09:51:43/")),
              "unet_resnet34_no_pt_scse":
                  ("unet_resnet34_scse", os.path.join(log_dir, "unet_resnet34_no_pt_scse/20200622_15:59:28/")),
              "unet_resnet34_pt_scse":
                  ("unet_resnet34_scse", os.path.join(log_dir, "unet_resnet34_pt_scse/20200622_21:39:42/")),
              "unet_resnet50_no_pt":
                  ("unet_resnet50", os.path.join(log_dir, "unet_resnet50_no_pt/20200623_09:46:17/")),
              # "unet_resnet50_pt":
              #     ("unet_resnet50", os.path.join(log_dir, "unet_resnet50_pt/20200623_17:44:48/")),
              }

    # Set seed
    random.seed(seed)
    np.random.seed(seed)

    # Load metadata and split
    df = probe_images_labels(image_dir, label_dir)
    split_df = pd_train_test_val_split(df, random_state=seed, train_frac=train_frac)

    test_metadata = split_df[split_df['fold'] == "test"]

    test_transforms = Compose(
        [
            Resize(*scale_size),
            CenterCrop(*crop_size),
            ToTensor()
         ]
    )

    print("Test set:")
    test_dataset = BRATS18Dataset(test_metadata, transforms=test_transforms)
    test_loader = DataLoader(dataset=test_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = []
    for model_name, (model_arch, model_log_dir) in model_paths.items():
        print("*** Evaluating: '{}' ***".format(model_name))
        model_params = pickle.load(open(os.path.join(model_log_dir, "params.p"), "rb"))
        if not hasattr(model_params, "arch_name"):
            model_params.arch_name = "unet"
        model = get_model(model_params)
        model.load_state_dict(torch.load(os.path.join(model_log_dir, "model", model_name + "__best")))
        model = model.to(device)
        test_stats = {}
        with torch.no_grad():
            for i, (inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader)):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                log_stats(test_stats, outputs, targets)

        avg_test_dice = sum(test_stats['dice_score']) / len(test_stats['dice_score'])
        results.append((model_name, avg_test_dice))

    result_df = pd.DataFrame(results, columns=["model_name", "test_dice"])
    result_df.to_csv("brats18_test_results.csv", index=False)

