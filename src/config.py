
class Config(object):
    def __init__(self):
        self.name = "test_experiment_1"
        self.log_path = "/home/tomron27/projects/MRI_Seg/logs/"
        self.image_dir = "/home/tomron27/datasets/BraTS18/proc/train/Database Images/HGG - Train/"
        self.label_dir = "/home/tomron27/datasets/BraTS18/proc/train/Database Images/HGG - Single/"
        self.scale_size = (256, 256)
        self.crop_size = (224, 224)
        self.num_epochs = 100
        self.lr = 0.001
        self.batch_size = 16
        self.num_workers = 4
        self.device_id = 2
        self.optim_step = 100
        self.optim_factor = 0.5
        self.save_freq = 20
