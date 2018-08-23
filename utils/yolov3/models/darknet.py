import numpy as np
import torch
import torch.nn as nn

DEBUG_CONSTRUCTION = False
DEBUG_DARKNET_FORWARD = False
DEBUG_DARKNET_LOADRAW = False

def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks specifications. Each blocks describes a block in
    the neural network to be built. Block is represented as a dictionary in
    the list.
    """
    lines = []
    with open(cfgfile, 'r') as f:
        for l in f:
            l_ = l.rstrip().lstrip()
            if len(l_) == 0 or l_[0] == '#':
                continue
            lines.append(l_)

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:  # we are dealing with an assign statement
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks

def get_flag(spec, k):
    v = spec.get(k)
    if isinstance(v, str):
        v = int(v)
    elif v is None:
        v = 0
    else:
        raise ValueError("{}, got {}".format(k, v))
    return v

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

def create_upsample_block(block_spec, net_context):
    """
    build a upsample block
    """
    assert block_spec['type'] == 'upsample', \
        "Module Type Error, must be upsample"
    m = nn.Sequential()
    stride = int(block_spec["stride"])
    upsample = nn.Upsample(scale_factor=2, mode="nearest")
    m.add_module("upsample_{}".format(net_context['block_index']),
                 upsample)
    return m

def create_conv_mod_block(block_spec, net_context):
    """
    build a convolutional block
    :param block_spec: a dict of specification of the block
    :param net_context: the global or relevatn settings of the net_context
        NB: this is an I/O parameter
    :type net_context: dict
    :return: a nn.Sequential object containing the operations of this
      convolutional block
    """
    assert block_spec['type'] == 'convolutional', \
        "Module Type Error, must be conv"
    m = nn.Sequential()

    # batch-normalisation: if batch norm, then no bias
    does_batch_normalize = get_flag(block_spec, 'batch_normalize')
    bias = False if does_batch_normalize else True

    activation = block_spec["activation"]
    filters = int(block_spec["filters"])
    does_padding = int(block_spec["pad"])
    kernel_size = int(block_spec["size"])
    stride = int(block_spec["stride"])

    if does_padding == 1:
        padding = (kernel_size - 1) // 2
    else:
        padding = 0

    # The "essential layer of this block"
    conv = nn.Conv2d(in_channels=net_context['prev_filters'],
                     out_channels=filters,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     bias=bias
                     )
    index = net_context['block_index']
    m.add_module("conv_{0}".format(index), conv)

    # Add the Batch Norm Layer
    if does_batch_normalize:
        bn = nn.BatchNorm2d(filters)
        m.add_module("batch_norm_{0}".format(index), bn)

    # Check the activation.
    # It is either Linear or a Leaky ReLU for YOLO
    if activation == "leaky":
        activn = nn.LeakyReLU(0.1, inplace=True)
        m.add_module("leaky_{0}".format(index), activn)
        # if linear do nothing
    return m, filters

# def create_route_block(block_spec, net_context):
#     """
#     build a route layer. This is empty, we will deal with routing in
#     forward computation of Darknet.
#     """
#     assert block_spec['type'] == 'route', \
#         "Module Type Error, must be route"
#         # Start  of a route
#         start = int(x["layers"][0])
#         # end, if there exists one.
#         try:
#             end = int(x["layers"][1])
#         except:
#             end = 0
#         # Positive anotation
#         if start > 0:
#             start = start - index
#         if end > 0:
#             end = end - index
#         route = EmptyLayer()
#         module.add_module("route_{0}".format(index), route)
#         if end < 0:
#             filters = output_filters[index + start] \
#                     + output_filters[index + end]
#         else:
#             filters= output_filters[index + start]

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_yolo_block(block_spec, net_context):
    """
    build a Yolo detection layer
    """
    m = nn.Sequential()
    mask = [int(ms_) for ms_ in block_spec['mask'].split(',')]
    anchors = [int(as_) for as_ in block_spec['anchors'].split(',')]
    anchors = [(a1, a2) for a1, a2 in
               zip(anchors[::2], anchors[1::2])]
    anchors = [anchors[i] for i in mask]
    det = DetectionLayer(anchors)
    m.add_module("detector_{}".format(net_context['block_index']),
                 det)
    return m


def create_modules(blocks):
    """
    build network layers according to the specification in blocks.
    :param blocks: a list of building blocks, each element is a dict.
        specifying a network layer.
    :type blocks: [dict]
    """
    net_info = blocks[0]
    module_list = nn.ModuleList()
    output_filters = []  # output channels after each block
    ctx = dict(block_index=0, prev_filters=3)

    filters = None
    for index, x in enumerate(blocks[1:]):
        ctx['block_index'] = index
        if x['type'] == 'convolutional':
            mod_block, filters = create_conv_mod_block(x, ctx)

        elif x['type'] == 'upsample':
            mod_block = create_upsample_block(x, ctx)

        elif x["type"] == "route":
            mod_block = nn.Sequential()
            original_layers_ = x['layers']
            x['layers'] = [int(l_) for l_ in x['layers'].split(',')]
            x['layers'] = [(index + l_ if l_ < 0 else l_)
                           for l_ in x['layers']]
            if DEBUG_CONSTRUCTION:
                print("Route at layer {}:({}): prev layers {}".format(
                    index, original_layers_, x['layers']))
            filters = np.sum([output_filters[l_] for l_ in x['layers']])

            route = EmptyLayer()
            mod_block.add_module("route_{0}".format(index), route)
        elif x["type"] == "shortcut":
            mod_block = nn.Sequential()
            shortcut = EmptyLayer()
            mod_block.add_module("shortcut_{}".format(index), shortcut)
        elif x['type'] == 'yolo':
            mod_block = create_yolo_block(x, ctx)

        ctx['prev_filters'] = filters
        output_filters.append(filters)
        module_list.add_module("block_{}".format(index), mod_block)

    return net_info, module_list

def predict_transform(pred, input_dim, anchors, num_classes):
    """
    The output of YOLO is a convolutional feature map that
    contains the bounding box attributes along the depth of the
    feature map. The attributes bounding boxes predicted by a cell
    are stacked one by one along each other. So, if you have to
    access the second bounding of cell at (5,6), then you will
    have to index it by map[5,6, (5+C): 2*(5+C)]. This form is
    very inconvenient for output processing such as thresholding
    by a object confidence, adding grid offsets to centers,
    applying anchors etc.

    So predict_transform function takes an detection feature map
    and turns it into a 2-D tensor, where each row of the tensor
    corresponds to attributes of a bounding box, in the following order.

    bb-1 @ (0,0) (5+C) | bb-2 @ (0,0) (5+C) | ...

    :param pred: predition feature map from the last convolution layer
    :param input_dim: input dimension (it is a square)
    :param anchors: the anchor boxes
    :param num_classes:
    """

    batch_size = pred.size(0)
    pred_height = pred.size(2)
    stride = input_dim // pred_height
    grid_size = input_dim // stride  # must be pred_height?
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    # previous
    #    batch_size x bbox_attrs
    # x pred_height (grid_size) x pred_width (grid_size)
    pred = pred.view(batch_size,
                     bbox_attrs * num_anchors,
                     grid_size * grid_size)  #
    # now each sample
    # bb1@(0,0)[0], bb1@(0,1)[0], bb1@(0,2)[0], ...
    # bb1@(0,0)[1], bb1@(0,1)[1], bb1@(0,2)[1], ...
    # ...
    # bb2@(0,0)[0], bb2@(0,1)[0], bb2@(0,2)[0], ...
    # ...

    pred = pred.transpose(1, 2).contiguous()  # for each model
    # now each sample
    # bb1@(0,0)[0], bb1@(0,0)[1], ..., bb2@(0,0)[0], ... -> (0,0) bb's
    # bb1@(0,1)[0], bb1@(0,1)[1], ..., bb2@(0,1)[0], ... -> (0,1) bb's
    # ...

    pred = pred.view(batch_size, grid_size * grid_size * num_anchors,
                     bbox_attrs)
    # now each sample
    # bb1@(0,0)[0], bb1@(0,0)[1], ..., bb1@(0,0)[5+C-1],
    # bb2@(0,0)[0], bb2@(0,0)[1], ..., bb2@(0,0)[5+C-1], -> (0,0) bb's
    # ...
    # bb1@(0,1)[0], bb1@(0,1)[1], ..., bb2@(0,1)[0], ... -> (0,1) bb's
    # ...

    # seems each anchor correspond to a BB at a cell.
    anchors = [(a[0] / float(stride), a[1] / float(stride))
               for a in anchors]

    # Sigmoid the  centre_X, centre_Y. and object confidencce
    pred[:, :, 0] = torch.sigmoid(pred[:, :, 0])
    pred[:, :, 1] = torch.sigmoid(pred[:, :, 1])
    pred[:, :, 4] = torch.sigmoid(pred[:, :, 4])

    # Add the center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)
    x_y_offset = torch.cat(
        (x_offset, y_offset), 1).repeat(1, num_anchors)
    x_y_offset = x_y_offset.view(-1, 2).unsqueeze(0)
    pred[:, :, :2] += x_y_offset
    #print(pred[0, ::10, :2])
    #print(x_y_offset[0, ::10])

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    pred[:, :, 2:4] = torch.exp(pred[:, :, 2:4]) * anchors

    pred[:, :, 5:5 + num_classes] = \
        torch.sigmoid((pred[:, :, 5:5 + num_classes]))
    pred[:, :, :4] *= stride
    return pred


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) \
                 * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def write_results(pred, confid_threshold, nms_conf=0.4):
    """
    Non-maximum repression and other transformation.
    :param pred:
    :param confid_threshold:
    :param nms_conf:
    :return: corners, classes and scores
    """
    conf_mask = (pred[:, :, 4] > confid_threshold)
    batch_size = pred.size(0)
    batch_boxes = []
    batch_classes = []
    batch_scores = []
    for i in range(batch_size):
        p_ = pred[i]
        det_indexes = conf_mask[i].nonzero().squeeze()
        if det_indexes.numel() == 0: continue
        p_ = p_[det_indexes]
        box_corner = torch.zeros_like(p_)
        box_corner[:, 0] = (p_[:, 0] - p_[:, 2] / 2.0)
        box_corner[:, 1] = (p_[:, 1] - p_[:, 3] / 2.0)
        box_corner[:, 2] = (p_[:, 0] + p_[:, 2] / 2.0)
        box_corner[:, 3] = (p_[:, 1] + p_[:, 3] / 2.0)
        boxes = box_corner[:, :4]
        scores, clss = torch.max(p_[:, 5:], dim=1)

        for c_ in torch.unique(clss):
            c = c_.item()
            cindex = (clss == c).nonzero().squeeze()
            nc = cindex.numel()
            for i1_ in range(nc - 1):
                i1 = cindex[i1_]
                i2s = cindex[i1_ + 1:]
                if scores[i1] < confid_threshold:  # if alread supressed?
                    continue
                # supress those with much overlapping
                ious = bbox_iou(boxes[i1].unsqueeze(dim=0), boxes[i2s])
                supress_ind = i2s[(ious > nms_conf).nonzero().squeeze()]
                scores[supress_ind] = 0
                # print(i1, ious)
                # print('\t', supress_ind)
        im_valid_det_ind = (scores > confid_threshold).nonzero().squeeze()
        # 3 tensors per image in batch
        batch_boxes.append(boxes[im_valid_det_ind])
        batch_classes.append(clss[im_valid_det_ind])
        batch_scores.append(scores[im_valid_det_ind])
    return batch_boxes, batch_classes, batch_scores


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.mod_specs = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.mod_specs)
        self.mod_specs = self.mod_specs[1:]  # 1st element is netinfo

    def forward(self, x, save_layers=[], save_prefix=""):
        is_first_yolo_layer = True
        total_detections = None
        outputs = {}
        input_dim = x.shape[2]  # int(self.net_info['height']) # fixed-sized image

        for i, (spec, mod) in enumerate(zip(self.mod_specs,
                                            self.module_list)):
            mtype = spec['type']
            if mtype in ['convolutional', 'upsample']:
                x = mod(x)
                if DEBUG_DARKNET_FORWARD:
                    print("\tLayer-{}:{}, out-dim: {}" \
                          .format(i, mtype, x.shape))
            elif mtype == 'route':
                if len(spec['layers']) > 1:
                    x = torch.cat([outputs[l_] for l_ in spec['layers']],
                                  dim=1)
                else:
                    x = outputs[spec['layers'][0]]
                if DEBUG_DARKNET_FORWARD:
                    print("\tLayer-{}: route {}, "
                          "output dim: {}" \
                          .format(i, spec['layers'], x.shape))
            elif mtype == 'shortcut':
                from_ = int(spec['from'])
                x = outputs[i - 1] + outputs[i + from_]
                if DEBUG_DARKNET_FORWARD:
                    print("\tLayer-{}:{}, out-dim: {}" \
                          .format(i, mtype, x.shape))

            elif mtype == 'yolo':
                anchors = mod[0].anchors
                num_classes = int(spec['classes'])

                # Transform
                x = predict_transform(x, input_dim,
                                      anchors,
                                      num_classes)
                if is_first_yolo_layer:
                    total_detections = x
                else:
                    total_detections = torch.cat((total_detections, x),
                                                 dim=1)

                is_first_yolo_layer = False
                if DEBUG_DARKNET_FORWARD:
                    print("\tLayer-{}: Yolo produce {} "
                          "detection proposals" \
                          .format(i, x.shape[1]))

            # end of layer-type branches
            outputs[i] = x

            if i in save_layers:
                fname = "{}{:03d}-{}.pt".format(save_prefix, i, mtype)
                torch.save(x, fname)

        # end-of-for-layers
        return total_detections

    def load_net_binary(self, weight_file_name):
        """
        Raw float numbers of the weights. This is to load the original
        weight file. We can later save them as a pytorch nn.Module's state
        dict.

        :param dnet: Darknet object
        :param weight_file_name:
        """
        with open(weight_file_name, "rb") as fp:
            header = np.fromfile(fp, dtype=np.int32, count=5)

            v_major, v_minor, v_sub = header[:3]
            images_trained = header[3]
            w_ = np.fromfile(fp, dtype=np.float32)

        total_param_size = 0
        for i, (spec, mod) in enumerate(zip(self.mod_specs,
                                            self.module_list)):
            if DEBUG_DARKNET_LOADRAW:
                print("Loading block-{}:{} parameters:".format(i, spec['type']))

            if spec['type'] == 'convolutional':
                does_bn = get_flag(spec, 'batch_normalize')
                param_list = []
                param_names = []
                if does_bn:
                    bn = mod[1]
                    param_list += \
                        [bn.bias, bn.weight, bn.running_mean, bn.running_var]
                    param_names += \
                        ['bn.bias', 'bn.weight', 'bn.running_mean',
                         'bn.running_var']
                else:
                    param_list.append(mod[0].bias)
                    param_names.append('conv.bias')

                param_list.append(mod[0].weight)
                param_names.append('conv.weight')
                for p_, pn_ in zip(param_list, param_names):
                    s_ = p_.numel()
                    p_.data.copy_(
                        torch.from_numpy(
                            w_[total_param_size:total_param_size + s_]) \
                            .view_as(p_)
                    )
                    total_param_size += s_
                    if DEBUG_DARKNET_LOADRAW:
                        print("\t{}:{}:{}".format(s_, p_.shape, pn_))

            else:
                assert len(list(mod.parameters())) == 0
                # only conv layers has parameters

            if DEBUG_DARKNET_LOADRAW:
                print("\ttotal:{}".format(total_param_size))


