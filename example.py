from PIL import Image
import numpy as np
import visdom
from utils.svr import app
from utils.yolov3.yolo_mod import YoloMod
from utils.cganimstyler.cgan_mod import CGANMod
DEBUG = False
viz = visdom.Visdom()

# each module accepts PIL image and output PIL image
mods = dict(yolo=YoloMod(),
            cgan=CGANMod())

def imread(fname):
    im = np.asarray(Image.open(fname).convert('RGB'))
    if im.ndim == 3 and im.shape[2]>3:
        im = im[:,:,:3]
    return np.ascontiguousarray(im)

def yolo_proc(fname):
    im = imread(fname)
    outim = mods['yolo'].process(im)
    viz.image(outim.transpose([2, 0, 1]),
              opts=dict(title='YOLO', caption=fname))


def cgan_proc(fname):
    im = Image.open(fname).convert('RGB')
    outim = mods['cgan'].process(im)
    viz.image(outim.transpose([2, 0, 1]),
              opts=dict(title=mods['cgan'].target_style, caption=fname))

app.upload_callbacks = dict(detect=yolo_proc, style_vangogh=cgan_proc)
if not DEBUG:
    app.app.run()
else:
    fname = "/Users/junli/local/projects/ai101/static/img/dog.jpg"
    if DEBUG == 'yolo':
        im = imread(fname)
    elif DEBUG == 'cgan':
        im = Image.open(fname).convert('RGB')
    outim = mods[DEBUG].process(im)
    print(outim.shape)
    viz.image(outim.transpose([2, 0, 1]),
              opts=dict(title=DEBUG, caption=fname))
