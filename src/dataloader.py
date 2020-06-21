import cv2
import numpy as np
import pandas as pd
import torch
import datetime


class BRATS18Dataset(torch.utils.data.Dataset):
    def __init__(self, metadata, transforms=None):
        super(BRATS18Dataset, self).__init__()

        self.metadata = metadata
        # data augmentation
        self.transforms = transforms

        print("Found {} slices from {} patients".format(len(self.metadata), len(pd.unique(self.metadata['patient']))))

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        image_path, label_path = self.metadata.iloc[index][['image_path', 'label_path']].values
        image, label = cv2.imread(str(image_path)), cv2.imread(str(label_path), 0)

        if self.transforms:
            try:
                # Apply albumentation transformations
                obj = self.transforms(image=image, mask=label)
                image, label = obj['image'], obj['mask']
                # Convert mask back to binary tensor
                label = (label > torch.Tensor([0.0])).float()
            except:
                raise ValueError("Exception on: {}, {}".format(image_path, label_path))

        return image, label

    def __len__(self):
        return len(self.metadata)
