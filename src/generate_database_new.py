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

# root_dir = "/home/tomron27/MRI_Project"
# base_dir = os.path.join(root_dir, "Project")
# os.chdir(base_dir)
data_dir = "/home/tomron27/datasets/BraTS18/source/train/"
dest_dir = data_dir.replace("source", "proc")

scan_dirs = ["HGG", "LGG"]

for scan_type in scan_dirs:
    dir = os.path.join(data_dir, scan_type)
    for i, folder in enumerate(os.listdir(dir)):
        try:
            if i % 10 == 9:
                print(i)
            if folder[0] == '.':
                continue
            else:
                current_folder = os.path.join(dir, folder)
            # Load scans
            # Each nii file contains information about the scan, and we need to extract the scan itself
            # In addition, each scan need a clockwise rotation of 90 degrees
            patient_scans = os.listdir(current_folder)
            for file in patient_scans:
                print(file)
                if file.find('flair') != -1:
                    scan_FLAIR = nib.load(current_folder + '/' + file)
                    FLAIR = scan_FLAIR.get_fdata()
                    FLAIR = np.rot90(FLAIR, 3)

                if file.find('t1ce') != -1:
                    scan_T1ce = nib.load(current_folder + '/' + file)
                    T1ce = scan_T1ce.get_fdata()
                    T1ce = np.rot90(T1ce, 3)

                if file.find('t2') != -1:
                    scan_T2 = nib.load(current_folder + '/' + file)
                    T2 = scan_T2.get_fdata()
                    T2 = np.rot90(T2, 3)

                if file.find('seg') != -1:
                    seg = nib.load(current_folder + '/' + file)
                    seg_img = seg.get_fdata()  # Convert NII to numpy.ndarray
                    seg_img = np.rot90(seg_img, 3)  # Clockwise rotation of 90 degrees
                    seg_img[seg_img == 4.0] = 3.0
                    seg_values = np.unique(seg_img)

            for s in range(seg_img.shape[2]):
                Slice = seg_img[:, :, s]
                if not np.any(Slice):
                    continue
                else:
                    # We concatenate three contrasts to form a single "RGB" image
                    # First channel: T1ce; Second channel: T2; Third channel: FLAIR
                    Multi_Slice = np.zeros((Slice.shape[0], Slice.shape[1], 3))
                    Multi_Slice[:, :, 0] = T1ce[:, :, s]
                    Multi_Slice[:, :, 1] = T2[:, :, s]
                    Multi_Slice[:, :, 2] = FLAIR[:, :, s]

                    # Normalize images intensity
                    Slice_RGB = cv2.normalize(Multi_Slice, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
                    Label_grey = seg_img[:, :, s]
                    # Resize the images and labels to width and height of 512
                    dim = (512, 512)
                    Slice_RGBr = cv2.resize(Slice_RGB, dim, interpolation=cv2.INTER_NEAREST)
                    img = Image.fromarray(Slice_RGBr)

                    img_dest_dir = os.path.join(dest_dir, 'Database Images', '{} - Train'.format(scan_type))
                    os.makedirs(img_dest_dir, exist_ok=True)
                    img.save(os.path.join(img_dest_dir, str(folder) + ' Slice ' + str(s) + '.png'))

                    img_label_multi_dir = os.path.join(dest_dir, 'Database Images', '{} - Multi'.format(scan_type))
                    os.makedirs(img_label_multi_dir, exist_ok=True)
                    labelr = cv2.resize(Label_grey, dim, interpolation=cv2.INTER_NEAREST)
                    labelr = labelr.astype(np.uint8) # convert dtype 'float64' to 'uint8'
                    imageio.imwrite(os.path.join(img_label_multi_dir, str(folder) + ' Slice ' + str(s) + '.png'), labelr)

                    img_label_single_dir = os.path.join(dest_dir, 'Database Images', '{} - Single'.format(scan_type))
                    os.makedirs(img_label_single_dir, exist_ok=True)
                    labelr[labelr > 0] = 1.0
                    imageio.imwrite(os.path.join(img_label_single_dir, str(folder) + ' Slice ' + str(s) + '.png'), labelr)
        except:
            continue
