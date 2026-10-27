from segmentation.baselines.cenet import cenet
from segmentation.baselines.linknet import linknet
from segmentation.baselines.transunet import transunet
from segmentation.baselines.unet import resunet, unet
from segmentation.baselines.unet3p import unet3p
from segmentation.baselines.unetpp import unetpp
from segmentation.baselines.unetv2 import unet_v2

BASELINES = {
    "unet": unet,
    "resunet": resunet,
    "unetpp": unetpp,
    "unet3p": unet3p,
    "unet_v2": unet_v2,
    "linknet": linknet,
    "transunet": transunet,
    "cenet": cenet,
}
