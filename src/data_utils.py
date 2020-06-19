import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def probe_images_labels(image_dir, label_dir):
    image_list = []
    for dirpath, dirnames, filenames in os.walk(image_dir):
        if not dirnames:
            for image_file in filenames:
                path = Path(os.path.join(dirpath, image_file))
                image_list.append(path)
    label_list = []
    for dirpath, dirnames, filenames in os.walk(label_dir):
        if not dirnames:
            for label_file in filenames:
                path = Path(os.path.join(dirpath, label_file))
                label_list.append(path)

    image_df = pd.DataFrame([(path, path.parts[-1], path.parts[-1].split(" ")[0])
                             for path in image_list], columns=['image_path', 'file', 'patient'])
    label_df = pd.DataFrame([(path, path.parts[-1], path.parts[-1].split(" ")[0])
                             for path in label_list], columns=['label_path', 'file', 'patient'])
    result = image_df.merge(label_df, on=['file', 'patient'])
    return result


def pd_train_test_split(df, by_patient=True, shuffle=True, random_state=42, test_frac=0.2):
    if by_patient:
        grouped = pd.DataFrame({'count': df.groupby(["patient"]).size()}).reset_index()
        grouped_split = int(test_frac * len(grouped))
        grouped_train, grouped_test = grouped[grouped_split:], grouped[:grouped_split]
        train = df.merge(grouped_train, on='patient')
        test = df.merge(grouped_test, on='patient')
        if shuffle:
            train = train.sample(frac=1, random_state=random_state).reset_index(drop=True)
            test = test.sample(frac=1, random_state=random_state).reset_index(drop=True)
        return train, test
    if shuffle:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split = int(test_frac*len(df))
    train, test = df[split:], df[:split]
    return train, test


def get_dice_score(y_true, y_pred, smooth=0.01):
    assert y_pred.size() == y_true.size()
    y_pred = y_pred[:, 0].contiguous().view(-1)
    y_true = y_true[:, 0].contiguous().view(-1)
    intersection = (y_pred * y_true).sum()
    dsc = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
    return dsc.item()


def log_stats(stats, fold, outputs, targets, loss, lr=None):
    loss = loss.item()
    if fold + '_loss' in stats:
        stats[fold + '_loss'].append(loss)
    else:
        stats[fold + '_loss'] = [loss]

    dice_score = get_dice_score(targets, outputs)

    if fold + '_dice_score' in stats:
        stats[fold + '_dice_score'].append(dice_score)
    else:
        stats[fold + '_dice_score'] = [dice_score]

    if lr is not None:
        stats['lr'] = lr


def write_stats(stats, writer, epoch):
    # Compute stuff
    avg_train_loss = sum(stats['train_loss']) / len(stats['train_loss'])
    avg_test_loss = sum(stats['test_loss']) / len(stats['test_loss'])
    avg_train_dice = sum(stats['train_dice_score']) / len(stats['train_dice_score'])
    avg_test_dice = sum(stats['test_dice_score']) / len(stats['test_dice_score'])
    
    # Loss
    writer.add_scalar('Loss/train_loss', avg_train_loss, epoch + 1)
    writer.add_scalar('Loss/test_loss', avg_test_loss, epoch + 1)

    # Dice
    writer.add_scalar('Dice_Score/train_dice_score', avg_train_dice, epoch + 1)
    writer.add_scalar('Dice_Score/test_dice_score', avg_test_dice, epoch + 1)

    # LR
    writer.add_scalar('Learning_Rate', stats['lr'], epoch + 1)


def visualize_batch(inputs, targets, epoch):
    images = (inputs.cpu().permute([0, 3, 2, 1]).numpy()*255).astype(np.uint8)
    labels = (targets.cpu().permute([0, 3, 2, 1]).numpy()).astype(np.uint8)
    dest_dir = "batch_visualize__epoch_{:03d}".format(epoch)
    os.makedirs(dest_dir, exist_ok=True)
    for i in range(images.shape[0]):
        image = images[i]
        label = labels[i][:, :, 0]
        for layer in range(image.shape[2]):
            plt.imshow(image[:, :, layer])
            plt.contour(label)
            plt.savefig(os.path.join(dest_dir, "test_image{:2d}_layer_{}.png".format(i, layer)))
            plt.clf()

