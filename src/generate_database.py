from PIL import Image
import nibabel as nib
import numpy as np
import os
import cv2
import imageio
import matplotlib.pyplot as plt

#################################################################################
################################# Train Dataset #################################
#################################################################################

source_data_dir = "/home/tomron27/datasets/BraTS18/source/train"
proc_data_dir = "/home/tomron27/datasets/BraTS18/proc/"

for dirpath, dirnames, filenames in os.walk(source_data_dir):
    if not dirnames and len(filenames) > 0:
        seg_img = None
        current_folder = os.path.basename(dirpath)
        for file in filenames:
            current_file = os.path.join(dirpath, file)
            print(current_file)
            if file.find('flair') != -1:
                scan_FLAIR = nib.load(current_file)
                FLAIR = scan_FLAIR.get_fdata()
                FLAIR = np.rot90(FLAIR, 3)

            if file.find('t1ce') != -1:
                scan_T1ce = nib.load(current_file)
                T1ce = scan_T1ce.get_fdata()
                T1ce = np.rot90(T1ce, 3)

            if file.find('t2') != -1:
                scan_T2 = nib.load(current_file)
                T2 = scan_T2.get_fdata()
                T2 = np.rot90(T2, 3)

            if file.find('seg') != -1:
                seg = nib.load(current_file)
                seg_img = seg.get_fdata()  # Convert NII to numpy.ndarray
                seg_img = np.rot90(seg_img, 3)  # Clockwise rotation of 90 degrees
                seg_img[seg_img == 4.0] = 3.0
                seg_values = np.unique(seg_img)

        # RGB_Labels = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0)}

        if seg_img:
            for s in range(seg_img.shape[2]):
                try:
                    slice = seg_img[:, :, s]
                    if not np.any(slice):
                        continue
                    else:
                        # We concatenate three contrasts to form a single "RGB" image
                        # First channel: T1ce; Second channel: T2; Third channel: FLAIR
                        Multi_slice = np.zeros((slice.shape[0], slice.shape[1], 3))
                        Multi_slice[:, :, 0] = T1ce[:, :, s]
                        Multi_slice[:, :, 1] = T2[:, :, s]
                        Multi_slice[:, :, 2] = FLAIR[:, :, s]

                        # Normalize images intensity
                        slice_RGB = cv2.normalize(Multi_slice, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
                        Label_grey = seg_img[:, :, s]

                        # Resize the images and labels to width and height of 512
                        dim = (512, 512)
                        slice_RGBr = cv2.resize(slice_RGB, dim, interpolation=cv2.INTER_NEAREST)
                        img = Image.fromarray(slice_RGBr)

                        img_dest_dir = os.path.join(dirpath.replace("source", "proc"), current_folder, "img")
                        os.makedirs(img_dest_dir, exist_ok=True)
                        img.save(os.path.join(img_dest_dir, str(current_folder) + ' slice ' + str(s) + '.png'))

                        label_multi_dest_dir = os.path.join(dirpath.replace("source", "proc"),
                                                            current_folder, "label_multi")
                        labelr = cv2.resize(Label_grey, dim, interpolation=cv2.INTER_NEAREST)
                        labelr = labelr.astype(np.uint8)
                        imageio.imwrite(os.path.join(label_multi_dest_dir,
                                                     str(current_folder) + ' slice ' + str(s) + '.png'), labelr)

                        label_single_dest_dir = os.path.join(dirpath.replace("source", "proc"),
                                                             current_folder, "label_single")
                        labelr[labelr > 0] = 1.0
                        imageio.imwrite(os.path.join(label_single_dest_dir,
                                                     str(current_folder) + ' slice ' + str(s) + '.png'), labelr)
                except:
                    continue
