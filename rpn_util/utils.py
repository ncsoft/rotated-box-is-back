import numpy as np
import tensorflow as tf


def bbox_transform_inv_tf(boxes, delta):
    w = tf.subtract(boxes[:, 2], boxes[:, 0]) + 1.0
    h = tf.subtract(boxes[:, 3], boxes[:, 1]) + 1.0
    center_x = tf.add(boxes[:, 0], w*0.5)
    center_y = tf.add(boxes[:, 1], h*0.5)

    dx = delta[:, 0]
    dy = delta[:, 1]
    dw = delta[:, 2]
    dh = delta[:, 3]

    pred_center_x = tf.add(center_x, tf.multiply(w, dx))
    pred_center_y = tf.add(center_y, tf.multiply(h, dy))
    pred_w = tf.multiply(tf.exp(dw), w)
    pred_h = tf.multiply(tf.exp(dh), h)

    pred_boxes0 = tf.subtract(pred_center_x, pred_w*0.5)
    pred_boxes1 = tf.subtract(pred_center_y, pred_h*0.5)
    pred_boxes2 = tf.add(pred_center_x, pred_w * 0.5)
    pred_boxes3 = tf.add(pred_center_y, pred_h * 0.5)

    return tf.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)


def bbox_transform_inv(boxes, delta):
    w = boxes[:, 2] - boxes[:, 0] + 1.0
    h = boxes[:, 3] - boxes[:, 1] + 1.0
    center_x = boxes[:, 0] + w*0.5
    center_y = boxes[:, 1] + h*0.5

    dx = delta[:, 0]
    dy = delta[:, 1]
    dw = delta[:, 2]
    dh = delta[:, 3]

    pred_center_x = center_x + w * dx
    pred_center_y = center_y + h * dy
    pred_w = np.exp(dw) * w
    pred_h = np.exp(dh) * h

    pred_boxes0 = pred_center_x - pred_w * 0.5
    pred_boxes1 = pred_center_y - pred_h * 0.5
    pred_boxes2 = pred_center_x + pred_w * 0.5
    pred_boxes3 = pred_center_y + pred_h * 0.5

    return np.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)


def clip_boxes_tf(boxes, im):
    b0 = tf.maximum(tf.minimum(boxes[:, 0], im[1] - 1), 0)
    b1 = tf.maximum(tf.minimum(boxes[:, 1], im[0] - 1), 0)
    b2 = tf.maximum(tf.minimum(boxes[:, 2], im[1] - 1), 0)
    b3 = tf.maximum(tf.minimum(boxes[:, 3], im[0] - 1), 0)
    return tf.stack([b0, b1, b2, b3], axis=1)


def clip_boxes(boxes, im):
    b0 = np.maximum(np.minimum(boxes[:, 0], im[1] - 1), 0)
    b1 = np.maximum(np.minimum(boxes[:, 1], im[0] - 1), 0)
    b2 = np.maximum(np.minimum(boxes[:, 2], im[1] - 1), 0)
    b3 = np.maximum(np.minimum(boxes[:, 3], im[0] - 1), 0)
    return np.stack([b0, b1, b2, b3], axis=1)


def bbox_transform_tf(rois, gt_rois):
    rois_width = rois[:, 2] - rois[:, 0]
    rois_height = rois[:, 3] - rois[:, 1]
    rois_ctr_x = rois[:, 0] + rois_width/2
    rois_ctr_y = rois[:, 1] + rois_height/2

    gt_width = gt_rois[:, 2] - gt_rois[:, 0]
    gt_height = gt_rois[:, 3] - gt_rois[:, 1]
    gt_ctr_x = gt_rois[:, 0] + gt_width/2
    gt_ctr_y = gt_rois[:, 1] + gt_height/2

    targets_dx = (gt_ctr_x - rois_ctr_x)/rois_width
    targets_dy = (gt_ctr_y - rois_ctr_y)/rois_height
    targets_dw = tf.log(gt_width/rois_width)
    targets_dh = tf.log(gt_height/rois_height)

    targets = tf.stack((targets_dx, targets_dy, targets_dw, targets_dh), axis=1)
    return targets


def bbox_transform(rois, gt_rois):
    rois_width = rois[:, 2] - rois[:, 0] + 1.0
    rois_height = rois[:, 3] - rois[:, 1] + 1.0
    rois_ctr_x = rois[:, 0] + rois_width/2
    rois_ctr_y = rois[:, 1] + rois_height/2

    gt_width = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_height = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + gt_width/2
    gt_ctr_y = gt_rois[:, 1] + gt_height/2

    targets_dx = (gt_ctr_x - rois_ctr_x)/rois_width
    targets_dy = (gt_ctr_y - rois_ctr_y)/rois_height
    targets_dw = np.log(gt_width/rois_width)
    targets_dh = np.log(gt_height/rois_height)

    targets = np.stack((targets_dx, targets_dy, targets_dw, targets_dh), axis=1)
    return targets


def bbox_overlaps(boxes, gt_boxes):
    gt_boxes = gt_boxes[:, 0:4]
    boxes = boxes.reshape((-1, 4))

    x11, y11, x12, y12 = np.hsplit(boxes, 4)
    x21, y21, x22, y22 = np.hsplit(gt_boxes, 4)

    xI1 = np.maximum(x11, x21.reshape((1, -1)))
    yI1 = np.maximum(y11, y21.reshape((1, -1)))

    xI2 = np.minimum(x12, x22.reshape((1, -1)))
    yI2 = np.minimum(y12, y22.reshape((1, -1)))

    intersection = np.maximum(0, (xI2 - xI1 + 1)) * np.maximum(0, (yI2 - yI1 + 1))
    box_area = (x12 - x11 + 1) * (y12 - y11 + 1)
    gt_box_area = (x22 - x21 + 1) * (y22 - y21 + 1)

    union = (box_area + gt_box_area.reshape((1, -1))) - intersection

    overlaps = intersection / union

    return np.maximum(overlaps, 0)
