import tensorflow as tf
import numpy as np
import cv2

FLAGS = tf.app.flags.FLAGS


coco_color_map = [
    0x00, 0x55, 0xFF,
	0xFF, 0x11, 0x00,
	0xFF, 0x1A, 0x00,
	0xFF, 0x22, 0x00,
	0xFF, 0x2A, 0x00,
	0xFF, 0x33, 0x00,
	0xFF, 0x3C, 0x00,
	0xFF, 0x44, 0x00,
	0xFF, 0x4C, 0x00,		#//  29
	0xFF, 0x55, 0x00,		#//  30
	0xFF, 0x5E, 0x00,		#//  31
	0xFF, 0x66, 0x00,		#//  32
	0xFF, 0x6E, 0x00,		#//  33
	0xFF, 0x77, 0x00,		#//  34
	0xFF, 0x77, 0x00,		#//  34
	0xFF, 0x80, 0x00,		#//  35
	0xFF, 0x88, 0x00,		#//  36
	0xFF, 0x90, 0x00,		#//  37
	0xFF, 0x99, 0x00,		#//  38
	0xFF, 0xA2, 0x00,		#//  39
	0xFF, 0xAA, 0x00,		#//  40
	0xF6, 0xB2, 0x00,		#//  41
	0xEE, 0xBB, 0x00,		#//  42
	0xE6, 0xC4, 0x00,		#//  43
	0xDD, 0xCC, 0x00,		#//  44
	0xD4, 0xD4, 0x00,		#//  45
	0xCC, 0xDD, 0x00,		#//  46
	0xC4, 0xE6, 0x00,		#//  47
	0xBB, 0xEE, 0x00,		#//  48
	0xB2, 0xF6, 0x00,		#//  49
	0xAA, 0xFF, 0x00,		#//  50
	0xA2, 0xFF, 0x00,		#//  51
	0x99, 0xFF, 0x00,		#//  52
	0x90, 0xFF, 0x00,		#//  53
	0x88, 0xFF, 0x00,		#//  54
	0x80, 0xFF, 0x00,		#//  55
	0x77, 0xFF, 0x00,		#//  56
	0x6E, 0xFF, 0x00,		#//  57
	0x66, 0xFF, 0x00,		#//  58
	0x5E, 0xFF, 0x00,		#//  59
	0x55, 0xFF, 0x00,		#//  61
	0x44, 0xFF, 0x11,		#//  62
	0x3C, 0xFF, 0x1A,		#//  63
	0x33, 0xFF, 0x22,		#//  64
	0x2A, 0xFF, 0x2A,		#//  65
	0x22, 0xFF, 0x33,		#//  66
	0x1A, 0xFF, 0x3C,		#//  67
	0x11, 0xFF, 0x44,		#//  68
	0x08, 0xFF, 0x4C,		#//  69
	0x00, 0xFF, 0x55,		#//  70
	0x00, 0xFF, 0x5E,		#//  71
	0x00, 0xFF, 0x66,		#//  72
	0x00, 0xFF, 0x6E,		#//  73
	0x00, 0xFF, 0x77,		#//  74
	0x00, 0xFF, 0x80,		#//  75
	0x00, 0xFF, 0x88,		#//  76
	0x00, 0xFF, 0x90,		#//  77
	0x00, 0xFF, 0x99,		#//  78
	0x00, 0xFF, 0xA2,		#//  79
	0x00, 0xFF, 0xAA,		#//  80
	0x00, 0xF6, 0xB2,
	0x00, 0xEE, 0xBB,
	0x00, 0xE6, 0xC4,
	0x00, 0xDD, 0xCC,
	0x00, 0xD4, 0xD4,
	0x00, 0xCC, 0xDD,
	0x00, 0xC4, 0xE6,
	0x00, 0xBB, 0xEE,
	0x00, 0xB2, 0xF6,
	0x00, 0xAA, 0xFF,
	0x00, 0xA2, 0xFF,
	0x00, 0x99, 0xFF,
	0x00, 0x90, 0xFF,
	0x00, 0x88, 0xFF,
	0x00, 0x80, 0xFF,
	0x00, 0x77, 0xFF,
	0x00, 0x6E, 0xFF,
	0x00, 0x66, 0xFF,
	0x00, 0x5E, 0xFF,
    0xFF, 0x08, 0x00,
    0x00, 0xA2, 0xFF,
	0x00, 0x99, 0xFF,
	0x00, 0x90, 0xFF,
	0x00, 0x88, 0xFF,
	0x00, 0x80, 0xFF,
	0x00, 0x77, 0xFF,
	0x00, 0x6E, 0xFF,
	0x00, 0x66, 0xFF,
	0x00, 0x5E, 0xFF,
    0xFF, 0x08, 0x00,
    0x00, 0x66, 0xFF,
	0x00, 0x5E, 0xFF,
    0xFF, 0x08, 0x00,

]


coco_class_dict =  {0: u'__background__',
 1: u'person',
 2: u'bicycle',
 3: u'car',
 4: u'motorcycle',
 5: u'airplane',
 6: u'bus',
 7: u'train',
 8: u'truck',
 9: u'boat',
 10: u'traffic light',
 11: u'fire hydrant',
 12: u'stop sign',
 13: u'parking meter',
 14: u'bench',
 15: u'bird',
 16: u'cat',
 17: u'dog',
 18: u'horse',
 19: u'sheep',
 20: u'cow',
 21: u'elephant',
 22: u'bear',
 23: u'zebra',
 24: u'giraffe',
 25: u'backpack',
 26: u'umbrella',
 27: u'handbag',
 28: u'tie',
 29: u'suitcase',
 30: u'frisbee',
 31: u'skis',
 32: u'snowboard',
 33: u'sports ball',
 34: u'kite',
 35: u'baseball bat',
 36: u'baseball glove',
 37: u'skateboard',
 38: u'surfboard',
 39: u'tennis racket',
 40: u'bottle',
 41: u'wine glass',
 42: u'cup',
 43: u'fork',
 44: u'knife',
 45: u'spoon',
 46: u'bowl',
 47: u'banana',
 48: u'apple',
 49: u'sandwich',
 50: u'orange',
 51: u'broccoli',
 52: u'carrot',
 53: u'hot dog',
 54: u'pizza',
 55: u'donut',
 56: u'cake',
 57: u'chair',
 58: u'couch',
 59: u'potted plant',
 60: u'bed',
 61: u'dining table',
 62: u'toilet',
 63: u'tv',
 64: u'laptop',
 65: u'mouse',
 66: u'remote',
 67: u'keyboard',
 68: u'cell phone',
 69: u'microwave',
 70: u'oven',
 71: u'toaster',
 72: u'sink',
 73: u'refrigerator',
 74: u'book',
 75: u'clock',
 76: u'vase',
 77: u'scissors',
 78: u'teddy bear',
 79: u'hair drier',
 80: u'toothbrush'}



def getcolor(score):
    r = 1.0
    g = 1.0
    b = 1.0
    if score < 0.25:
        r = 0
        g = 4 * score
    elif score < 0.5:
        r = 0
        b = 1 + 4 * (0.25 - score)
    elif score < 0.75:
        r = 4*(score - 0.5)
        b = 0
    else:
        g = 1 + 4 * (0.75 - score)
        b = 0
    r *= 255
    g *= 255
    b *= 255
    return (int(r), int(g), int(b))


def checkpoly(ind_box, w, h):
    if np.min(ind_box[:, 0]) < 0:
        return 0
    elif np.max(ind_box[:, 0]) > w:
        return 0
    elif np.min(ind_box[:, 1]) < 0:
        return 0
    elif np.max(ind_box[:, 1]) > h:
        return 0
    return 1


def makepoly_with_score(batch, boxes, scores, h, w, img, thres=0.1):
    w = w.astype(np.int32)
    h = h.astype(np.int32)
    img = img[:, :, :, ::-1]
    img = img.astype(np.uint8)

    for i in range(batch):
        for box, score in zip(boxes[i], scores[i]):
            if score > thres:
                color = getcolor(score)
                valid_box = box.reshape([-1, 4, 2]).astype(np.int32)
                for ind_box in valid_box:
                    if checkpoly(ind_box, w, h):
                        img[i] = cv2.drawContours(img[i], [ind_box], 0, color, 1)

    img = img.astype(np.float32)
    img = img / 255
    return img


def makepoly_with_score_color(batch, boxes, scores, h, w, img, color, thres=0.0):
    w = int(w)
    h = int(h)
    img = img.astype(np.uint8)
    for i in range(batch):
        for box, score in zip(boxes[i], scores[i]):
            if score > thres:
                valid_box = box.reshape([-1, 4, 2]).astype(np.int32)
                for ind_box in valid_box:
                    ind_box[:, 0] = np.clip(ind_box[:, 0], 0, w)
                    ind_box[:, 1] = np.clip(ind_box[:, 1], 0, h)
                    img[i] = cv2.drawContours(img[i], [ind_box], 0, color, 3)
    img = img
    return img


def debugclassifier(quadboxes, gt_boxes, roi_idx, offset_ch, mask, classifier_ch):
    """
    :param quadboxes: (B, 300, 8)
    :param gt_boxes: (N',8) gathered
    :param roi_idx: (N', 2) idx
    :param offset_ch: (N' 8) gatherd
    :parma classifier_ch : (N')
    :return:
    """
    geo_map = np.zeros((quadboxes.shape[0], FLAGS.input_size, FLAGS.input_size, 3), dtype=np.uint8)
    for i,(idx,bin) in enumerate(zip(roi_idx, mask)):
        b, n = idx
        quad = quadboxes[b,n,:].reshape([4, 2]).astype(np.int32)
        gt_box = gt_boxes[i].reshape([4, 2]).astype(np.int32)
        offset = offset_ch[i].reshape([4, 2]).astype(np.int32)
        score = classifier_ch[i]
        color = getcolor(score)

        if checkpoly(quad, FLAGS.input_size, FLAGS.input_size) and score > 0.5:
            geo_map[b] = cv2.drawContours(geo_map[b], [offset], 0, (255, 255, 255), 1)
            geo_map[b] = cv2.circle(geo_map[b], (offset[0, 0], offset[0, 1]), 4, (153, 151, 89), -1)
            geo_map[b] = cv2.circle(geo_map[b], (offset[1, 0], offset[1, 1]), 4, (153, 89, 151), -1)
            geo_map[b] = cv2.circle(geo_map[b], (offset[2, 0], offset[2, 1]), 4, (91, 241, 241), -1)
            geo_map[b] = cv2.circle(geo_map[b], (offset[3, 0], offset[3, 1]), 4, (241, 91, 134), -1)

            geo_map[b] = cv2.drawContours(geo_map[b], [quad], 0, color, 1)
            geo_map[b] = cv2.circle(geo_map[b], (quad[0, 0], quad[0, 1]), 4, (255, 0, 0), -1)
            geo_map[b] = cv2.circle(geo_map[b], (quad[1, 0], quad[1, 1]), 4, (0, 255, 0), -1)
            geo_map[b] = cv2.circle(geo_map[b], (quad[2, 0], quad[2, 1]), 4, (0, 0, 255), -1)
            geo_map[b] = cv2.circle(geo_map[b], (quad[3, 0], quad[3, 1]), 4, (255, 255, 255), -1)

    geo_map = geo_map.astype(np.float32)
    geo_map = geo_map / 255
    return geo_map


def debugclass(quadboxes, gt_label, roi_idx, gt_boxes, mask, classifier_ch):
    """
    :param quadboxes: (B, 300, 8)
    :param gt_label: (B, 300)
    :param roi_idx: (N', 2) idx
    :param gt_boxes: (B, 300, 8) gatherd
    :parma classifier_ch : (N')
    :return:
    """

    geo_map = np.zeros((quadboxes.shape[0], FLAGS.input_size, FLAGS.input_size, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    lineType = 2

    for i, (idx, bin) in enumerate(zip(roi_idx, mask)):
        b, n = idx
        quad = quadboxes[b, n, :].reshape([4, 2]).astype(np.int32)
        class_info = classifier_ch[i]

        class_idx = np.argmax(class_info, axis= -1)
        class_score = np.amax(class_info, axis=-1)

        rgb = (coco_color_map[3*class_idx], coco_color_map[3*class_idx+1], coco_color_map[3*class_idx+2])

        if checkpoly(quad, FLAGS.input_size, FLAGS.input_size) and class_score > 0.5:
            geo_map[b] = cv2.drawContours(geo_map[b], [quad], 0, rgb, 1)
            #cv2.putText(geo_map[b], (coco_class_dict[class_idx + 1] + u" " + str(class_score)),
            #            (quad[0][0],quad[0][1]),
            #            font,
            #            fontScale,
            #            rgb,
            #            lineType)

    label_shape = gt_label.shape
    fontcolor = (255, 0, 0)

    for b in range(label_shape[0]):
        for i in range(label_shape[1]):
            if gt_label[b, i] != 0:
                quad = gt_boxes[b, i, :].reshape([4, 2]).astype(np.int32)
                geo_map[b] = cv2.drawContours(geo_map[b], [quad], 0, fontcolor, 1)
                #cv2.putText(geo_map[b], (coco_class_dict[gt_label[b,i]]),
                #            (quad[0][0], quad[0][1]),
                #            font,
                #            fontScale,
                #            fontColor,
                #            lineType)

    geo_map = geo_map.astype(np.float32)
    geo_map = geo_map / 255
    return geo_map


def viz_pos_neg_anchor(concat_anchor_box, concat_score_box, pos, mask, input_shape):
    """
    :param concat_anchor_box: (B, N', 8)
    :param concat_score_box: (B, N', 1)
    :param pos: (B, N', 1)
    :param mask: (B, N', 1)
    :return:
    """
    viz_anchor_box_pos_img = np.zeros(input_shape, dtype=np.uint8)
    viz_anchor_box_neg_img = np.zeros(input_shape, dtype=np.uint8)
    for b, (n_concat_anchor_box, n_concat_score_box, n_pos, n_mask) in enumerate(zip(concat_anchor_box, concat_score_box, pos, mask)):
        for each_concat_anchor_box, each_concat_score_box, each_pos, each_mask in zip(n_concat_anchor_box,
                                                                                      n_concat_score_box,
                                                                                      n_pos, n_mask):
            if each_mask[0] == 1:
                quad = each_concat_anchor_box.reshape([4, 2]).astype(np.int32)
                color = getcolor(each_concat_score_box[0])
                if each_pos[0] == 1:
                    viz_anchor_box_pos_img[b] = cv2.drawContours(viz_anchor_box_pos_img[b], [quad], 0, color, 1)
                else:
                    viz_anchor_box_neg_img[b] = cv2.drawContours(viz_anchor_box_neg_img[b], [quad], 0, color, 1)

    viz_anchor_box_pos_img = viz_anchor_box_pos_img.astype(np.float32)
    viz_anchor_box_pos_img = viz_anchor_box_pos_img / 255
    viz_anchor_box_neg_img = viz_anchor_box_neg_img.astype(np.float32)
    viz_anchor_box_neg_img = viz_anchor_box_neg_img / 255
    return viz_anchor_box_pos_img, viz_anchor_box_neg_img