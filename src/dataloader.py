import cv2
import numpy as np
import torch
import datetime
from src.data_utils import probe_images_labels, pd_train_test_split


class BRATS18Dataset(torch.utils.data.Dataset):
    def __init__(self, metadata, transforms=None):
        super(BRATS18Dataset, self).__init__()

        self.metadata = metadata
        # data augmentation
        self.transforms = transforms

        print("Found {} samples".format(len(self.metadata)))

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        image_path, label_path = self.metadata.iloc[index][['image_path', 'label_path']].values
        image, label = cv2.imread(str(image_path)), cv2.imread(str(label_path))

        if self.transforms:
            obj = self.transforms(image=image, mask=label)
            image, label = obj['image'], obj['mask']

        return image, label

    def __len__(self):
        return len(self.metadata)


if __name__ == "__main__":

    test_file = "/home/tomron27/datasets/BraTS18/proc/train/Database Images/HGG - Single/Brats18_TCIA08_469_1 Slice 95.png"
    im = cv2.imread(test_file)
    np_im = np.array(im)

    test_image_dir = "/home/tomron27/datasets/BraTS18/proc/train/Database Images/HGG - Train/"
    test_label_dir = "/home/tomron27/datasets/BraTS18/proc/train/Database Images/HGG - Single/"
    df = probe_images_labels(test_image_dir, test_label_dir)
    train, test = pd_train_test_split(df)

    train_dataset = BRATS18Dataset(train)
    image, label = train_dataset.__getitem__(0)
    x=0
