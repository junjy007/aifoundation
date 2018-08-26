import os
# noinspection PyCompatibility
import urllib.request
from .imconvert import load_generator_from, tensor2im, image_transform

AVAILABLE_TARGET_STYLES = [
    "apple2orange", "orange2apple",
    "summer2winter_yosemite", "winter2summer_yosemite",
    "horse2zebra", "zebra2horse", "monet2photo",
    "style_monet", "style_cezanne", "style_ukiyoe",
    "style_vangogh", "sat2map", "map2sat",
    "cityscapes_photo2label", "cityscapes_label2photo",
    "facades_photo2label", "facades_label2photo", "iphone2dslr_flower"
]

TARGET_STYLE = AVAILABLE_TARGET_STYLES[10]


def get_trained_model_parameters(style, modeldir):
    model_path = "{}/{}.pth".format(modeldir, style)
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(
            "http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/"
            + TARGET_STYLE + ".pth",
            model_path)
    return model_path


# build the style model
class CGANMod:
    def __init__(self, global_args, target_style=TARGET_STYLE):
        """
        :param global_args: global setting of the service
        """
        self.device = global_args.device
        self.pwd_ = os.path.dirname(os.path.abspath(__file__))
        self.target_style = target_style
        # noinspection PyPep8
        fpth = lambda p: os.path.join(self.pwd_, p)
        self.model_path = get_trained_model_parameters(
            target_style,
            fpth("saved_style_models"))
        self.netG = load_generator_from(self.model_path)
        self.netG.to(global_args.device)
        self.netG.eval()
        return

    def process(self, im):
        """
        :param im: a PIL image
        :type im: np.ndarray
        :return:
        """
        original_r = float(im.size[0]) / im.size[1]
        im_t = image_transform(im).unsqueeze(0).to(self.device)
        res = self.netG(im_t)
        outim = tensor2im(res)
        return outim
