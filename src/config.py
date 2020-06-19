class Config(object):
    def __init__(self):
        self.name = "unet__no_pt"
        self.log_path = "/home/tomron27/projects/MRI_Seg/logs/"
        self.image_dir = "/home/tomron27/datasets/BraTS18/proc/train/Database Images/HGG - Train/"
        self.label_dir = "/home/tomron27/datasets/BraTS18/proc/train/Database Images/HGG - Single/"
        self.scale_size = (256, 256)
        self.crop_size = (224, 224)
        self.num_epochs = 100
        self.num_classes = 1
        self.lr = 1e-4
        self.batch_size = 64
        self.num_workers = 1
        self.device_id = 2
        self.optim_step = 20
        self.optim_factor = 0.75
        self.save_freq = 20
        self.loss_smooth = 0.01
        self.pretrained = False
        self.model_weights = "/home/tomron27/projects/MRI_Seg/src/models/unet/unet.pt"
        self.seed = 42