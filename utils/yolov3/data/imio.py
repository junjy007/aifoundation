import cv2


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


#     TODO: handle cuda
# def prep_image(img, input_dim):
#     """
#     Prepare image for inputting to the neural network.
#
#     """
#     img = (letterbox_image(img, (inp_dim, inp_dim)))
#     img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
#     img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
#     return img


def mark_box(out_im, box, c, score, class_names,
             class_colors=None, input_size=416):
    raw_h, raw_w = out_im.shape[:2]
    scaling_factor = min(float(input_size) / raw_h, float(input_size) / raw_w)
    box[:, [1, 3]] -= (input_size - scaling_factor * raw_h) / 2.0
    box[:, [0, 2]] -= (input_size - scaling_factor * raw_w) / 2.0
    box /= scaling_factor
    box[:, 1].clamp_(0.0, raw_h)
    box[:, 3].clamp_(0.0, raw_h)
    box[:, 0].clamp_(0.0, raw_w)
    box[:, 2].clamp_(0.0, raw_w)

    for b_, c_, s_ in zip(box, c, score):
        c1 = tuple(b_[:2].int())
        c2 = tuple(b_[2:4].int())
        if class_colors is None:
            color = (0, 255, 0)
        else:
            color = class_colors[c_.item() % len(class_colors)]
        label = "{0}".format(class_names[c_.item()])
        cv2.rectangle(out_im, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(out_im, c1, c2, color, -1)
        cv2.putText(out_im, label, (c1[0], c1[1] + t_size[1] + 4),
                    cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

    return out_im
