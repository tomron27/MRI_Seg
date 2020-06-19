import os
from pathlib import Path
import pandas as pd


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

    image_df = pd.DataFrame([(path, path.parts[-1]) for path in image_list], columns=['image_path', 'file'])
    label_df = pd.DataFrame([(path, path.parts[-1]) for path in label_list], columns=['label_path', 'file'])
    result = image_df.merge(label_df, on='file')
    return result


def pd_train_test_split(df, shuffle=True, random_state=42, test_frac=0.2):
    if shuffle:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split = int(test_frac*len(df))
    train, test = df[split:], df[:split]
    return train, test


def log_stats(stats, fold, outputs, targets, loss, lr=None):
    loss = loss.item()
    if fold + '_loss' in stats:
        stats[fold + '_loss'].append(loss)
    else:
        stats[fold + '_loss'] = [loss]

    # TODO - measure dice here


def write_stats(stats, writer, epoch):
    # Compute stuff
    avg_train_loss = sum(stats['train_loss']/len(stats['train_loss']))
    avg_test_loss = sum(stats['test_loss'] / len(stats['test_loss']))
    
    # Loss
    writer.add_scalar('Loss/train_loss', avg_train_loss, epoch + 1)
    writer.add_scalar('Loss/test_loss', avg_test_loss, epoch + 1)

    # Dice
    # TODO - log dice score
