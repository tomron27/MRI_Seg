class Config(object):
    def __init__(self):
        self.name = "unet_resnet50_no_pt_scse"
        self.arch_name = "unet_resnet50_scse"
        self.log_path = "/home/tomron27/projects/MRI_Seg/logs/"
        self.image_dir = "/home/tomron27/datasets/BraTS18/proc/train/Database Images/Full - Train/"
        self.label_dir = "/home/tomron27/datasets/BraTS18/proc/train/Database Images/Full - Single/"
        self.scale_size = (256, 256)
        self.crop_size = (224, 224)
        self.num_epochs = 60
        self.num_classes = 1
        self.lr = 1e-4
        self.batch_size = 64
        self.num_workers = 4
        self.device_id = 2
        self.optim_step = 20
        self.optim_factor = 0.75
        self.save_freq = 20
        self.loss_smooth = 0.01
        self.pretrained = False
        self.model_weights = "/home/tomron27/projects/MRI_Seg/src/models/unet/unet.pt"
        self.seed = 42
        self.train_frac = 0.6
