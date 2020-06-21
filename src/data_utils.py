import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
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


def pd_train_test_val_split(df, random_state=42, train_frac=0.6):
    test_valid_frac = 1-((1-train_frac)/2.0)
    grouped = pd.DataFrame({'count': df.groupby(["patient"]).size()}).reset_index()
    grouped_train, grouped_val, grouped_test = \
        np.split(grouped.sample(frac=1, random_state=random_state),
                 [int(train_frac * len(grouped)), int(test_valid_frac * len(grouped))])
    grouped_train['fold'] = "train"
    grouped_val['fold'] = "val"
    grouped_test['fold'] = "test"
    grouped_res = grouped_train.append(grouped_val).append(grouped_test)

    result = df.merge(grouped_res, on='patient')
    return result


def get_dice_score(y_true, y_pred, smooth=0.01):
    assert y_pred.size() == y_true.size()
    y_pred = y_pred[:, 0].contiguous().view(-1)
    y_true = y_true[:, 0].contiguous().view(-1)
    intersection = (y_pred * y_true).sum()
    dsc = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
    return dsc.item()


def log_stats(stats, outputs, targets, loss, lr=None):
    loss = loss.item()
    if 'loss' in stats:
        stats['loss'].append(loss)
    else:
        stats['loss'] = [loss]

    dice_score = get_dice_score(targets, outputs)

    if 'dice_score' in stats:
        stats['dice_score'].append(dice_score)
    else:
        stats['dice_score'] = [dice_score]

    if lr is not None:
        stats['lr'] = lr


def save_best_model(model, stats, best_score, log_dir, model_name, score="dice_score"):
    current_avg_score = sum(stats[score]) / len(stats[score])
    if current_avg_score > best_score:
        print("Model improved '{}' from {} to {}".format(score, best_score, current_avg_score))
        save_dir = os.path.join(log_dir, 'model')
        os.makedirs(save_dir, exist_ok=True)
        model_file = os.path.join(save_dir, model_name + '__best')
        print("Saving model to '{}'".format(model_file))
        torch.save(model.state_dict(), model_file)
    return current_avg_score


def write_stats(stats, writer, epoch):
    # Compute stuff
    avg_loss = sum(stats['loss']) / len(stats['loss'])
    avg_dice = sum(stats['dice_score']) / len(stats['dice_score'])
    
    # Loss
    writer.add_scalar('Loss', avg_loss, epoch + 1)

    # Dice
    writer.add_scalar('Dice_Score', avg_dice, epoch + 1)

    # LR
    if 'lr' in stats:
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

