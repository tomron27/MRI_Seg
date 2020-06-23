import segmentation_models_pytorch as smp
from models.unet.unet import UNet


def get_model(params):
    arch = params.arch_name
    if arch.lower() == "unet":
        model = UNet(in_channels=3, out_channels=params.num_classes, init_features=32,
                     pretrained=params.pretrained, weights=params.model_weights)
    elif arch.lower() == "unet_resnet34":
        weights = "imagenet" if params.pretrained else None
        model = smp.Unet(encoder_name="resnet34", encoder_depth=5, in_channels=3, classes=1, encoder_weights=weights,
                         activation="sigmoid")
    elif arch.lower() == "unet_resnet34_scse":
        weights = "imagenet" if params.pretrained else None
        model = smp.Unet(encoder_name="resnet34", encoder_depth=5, in_channels=3, classes=1, encoder_weights=weights,
                         activation="sigmoid", decoder_attention_type="scse")
    elif arch.lower() == "unet_resnet50":
        weights = "imagenet" if params.pretrained else None
        model = smp.Unet(encoder_name="resnet50", encoder_depth=5, in_channels=3, classes=1, encoder_weights=weights,
                         activation="sigmoid")
    elif arch.lower() == "unet_resnet50_scse":
        weights = "imagenet" if params.pretrained else None
        model = smp.Unet(encoder_name="resnet50", encoder_depth=5, in_channels=3, classes=1, encoder_weights=weights,
                         activation="sigmoid", decoder_attention_type="scse")
    else:
        raise NotImplementedError("Unknown architecture: '{}".format(arch))
    return model
