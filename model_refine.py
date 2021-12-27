import tensorflow as tf
import numpy as np
from net import resnet_v1
from net import resnet
from roi_rotate import rotate_np
from roi_rotate import roi_rotate_quad as roi_rotate_quad
from utils import quad_to_param, getboxes, bbox_inverse_transform, bbox_transform_from_anchor_list, generate_anchor_box
from intermediate_processing import filter_score_idx, standard_nms_idx, Rboxiou, makeseggt, select_pos_neg_mask, select_pos_neg_reg_mask
from debug_utils import debugclassifier, debugclass, makepoly_with_score

import tensorflow.contrib.slim as slim

tf.app.flags.DEFINE_integer('text_scale', 512, '')
tf.app.flags.DEFINE_integer('crop_width',8, '')
tf.app.flags.DEFINE_integer('crop_height', 8, '')
tf.app.flags.DEFINE_integer('crop_seg_width', 32, '')
tf.app.flags.DEFINE_integer('crop_seg_height', 32, '')
tf.app.flags.DEFINE_integer('num_class', 1, '')
tf.app.flags.DEFINE_integer('num_nms', 150, '')
tf.app.flags.DEFINE_integer('num_refine_proposal', 512, '')
FLAGS = tf.app.flags.FLAGS


def upsample(inputs, output_shape):
    return tf.image.resize_images(inputs, size=[output_shape[1], output_shape[2]])


def unpool(inputs):
    shape = tf.shape(inputs)
    h = tf.cast(shape[1] * 2, tf.int32)
    w = tf.cast(shape[2] * 2, tf.int32)
    return tf.image.resize_bilinear(inputs, size=[h,  w])


def pool(inputs):
    shape = tf.shape(inputs)
    h = tf.cast(shape[1] / 2, tf.int32)
    w = tf.cast(shape[2] / 2, tf.int32)
    return tf.image.resize_bilinear(inputs, size=[h,  w])


def normalblock(h_before, ch_num, is_training, strides=(1, 1), no_activation = False):
    h_before = tf.layers.batch_normalization(h_before, training=is_training)
    h_before = tf.layers.conv2d(inputs=h_before, filters=ch_num, kernel_size=[1, 1], strides=[1, 1],
                                padding="SAME")
    if not no_activation:
        h_before = tf.nn.relu(h_before)

    h_before = tf.layers.batch_normalization(h_before, training=is_training)
    h_before = tf.layers.conv2d(inputs=h_before, filters=ch_num, kernel_size=[3, 3], strides=strides,
                                padding="SAME")
    if not no_activation:

        h_before = tf.nn.relu(h_before)

    return h_before


def feature_extraction(feature, is_training):
    with tf.variable_scope("mini_feature_extraction"):
        seed_ch = 64
        feature = tf.layers.batch_normalization(feature, training=is_training)
        feature = tf.layers.conv2d(inputs=feature, filters=seed_ch, kernel_size=[3, 3],
                                   strides=[1, 1], padding="same")
        feature = tf.nn.relu(feature)

        feature = tf.layers.max_pooling2d(feature, [2, 2], [2, 2], name="pool_1")

        feature = tf.layers.batch_normalization(feature, training=is_training)
        feature = tf.layers.conv2d(inputs=feature, filters=seed_ch * 2, kernel_size=[3, 3],
                                   strides=[1, 1], padding="same")
        feature = tf.nn.relu(feature)

        feature = tf.layers.max_pooling2d(feature, [2, 2], [2, 2], name="pool_2")

        feature = tf.layers.batch_normalization(feature, training=is_training)
        feature = tf.layers.conv2d(inputs=feature, filters=seed_ch * 4, kernel_size=[3, 3],
                                   strides=[1, 1], padding="same")
        feature = tf.nn.relu(feature)

        feature = tf.layers.max_pooling2d(feature, [2, 2], [2, 2], name="pool_3")
        feature = tf.layers.batch_normalization(feature, training=is_training)
        feature = tf.layers.conv2d(inputs=feature, filters=seed_ch * 8, kernel_size=[3, 3],
                                   strides=[1, 1], padding="same")
        feature = tf.nn.relu(feature)

    return feature


class Rpn(object):
    def __init__(self, param, base_size, is_training):
        self.is_training = is_training
        self.param = param
        self.base_size = base_size

    def build_network(self, inputs, im_shape):
        """

        :param input: feature map lists
        :return: ??
        """
        rpn_box_lists = []
        rpn_cls_lists = []
        rpn_anchor_lists = []

        for i, feature_map in enumerate(inputs):
            feature_map_shape = tf.shape(feature_map)
            each_rpn = tf.layers.batch_normalization(feature_map, training=self.is_training)
            each_rpn_score = tf.layers.conv2d(inputs=each_rpn, filters=self.param["NUM_ANCHOR"] * 2, kernel_size=[1, 1],
                                              strides=[1, 1], padding="same", name="rpn_score_layer_{}".format(i))
            each_rpn_box_geo = tf.layers.conv2d(inputs=each_rpn, filters=self.param["NUM_ANCHOR"] * 4, kernel_size=[1, 1],
                                                strides=[1, 1], padding="same" , kernel_initializer=tf.zeros_initializer, bias_initializer=tf.zeros_initializer,  name="rpn_box_layer_{}".format(i))
            each_rpn_box_theta = (tf.layers.conv2d(inputs=each_rpn, filters=self.param["NUM_ANCHOR"] * 1,
                                                   kernel_size=[1, 1], activation=tf.nn.sigmoid,
                                                   kernel_initializer=tf.zeros_initializer,
                                                   bias_initializer=tf.zeros_initializer,
                                                   strides=[1, 1], padding="same",
                                                   name="rpn_box_layer_theta_{}".format(i)) - 0.5) * np.pi / 2.0
            each_rpn_box = tf.concat([each_rpn_box_geo, each_rpn_box_theta], axis=-1)
            rpn_box_lists.append(each_rpn_box)
            rpn_cls_lists.append(each_rpn_score)
            rpn_anchor_lists.append(generate_anchor_box(im_shape[1:3], feature_map_shape[1:3], self.base_size))

        concat_anchor_box, concat_score_box = bbox_transform_from_anchor_list(rpn_box_lists,
                                                                              rpn_cls_lists,
                                                                              rpn_anchor_lists,
                                                                              im_shape)
        rpn_box_offset_list = []
        for rpn_box in rpn_box_lists:
            # (B, H(i)*W(i)*NUM_ANCHOR, 4)
            shape = tf.shape(rpn_box)
            rpn_box_offset_list.append(tf.reshape(rpn_box, [shape[0], -1, 5]))
        concat_anchor_offset = tf.concat(rpn_box_offset_list, axis=1)
        boxes, scores = self._proposal_layer(concat_anchor_box, concat_score_box, FLAGS.num_nms)
        return [boxes, scores], [concat_anchor_box, concat_score_box, concat_anchor_offset]

    def _proposal_layer(self, concat_anchor_box, concat_score_box, max_size):
        """
        :param concat_anchor_box: (B, N', 8*2)
        :param concat_score_box:  (B, N', 1*2)
        :return: (B, MAX_SIZE, 8), (B,MAX_SIZE, 1)
        """
        concat_anchor_box = concat_anchor_box[:, :, :8]
        concat_anchor_box_shape = tf.shape(concat_anchor_box)

        softmax_rpn_cls_reshape = tf.nn.softmax(concat_score_box, axis=-1)
        concat_score_box = softmax_rpn_cls_reshape[:, :, 1:]

        filtered_box, filtered_score, filtered_idx = filter_score_idx(concat_anchor_box, concat_score_box, 0.1)
        return tf.py_func(standard_nms_idx, [filtered_box, filtered_score,
                                             filtered_idx, concat_anchor_box_shape[0],
                                             max_size], [tf.float32, tf.float32])


class Textmask(object):
    def __init__(self, is_training, input_shape):
        self.is_training = is_training
        self.input_shape = input_shape

    def build_network(self, input_list, quadboxes_list):
        cropped_feature_map_list = []
        roi_idx_list = []
        cropped_num_acc = tf.constant(0, dtype=tf.int64)

        for each_input, each_quadboxes in zip(input_list, quadboxes_list):
            cropped, roi_idx, batch_cord, theta_cord = roi_rotate_quad(each_input, each_quadboxes, (8, 8),
                                                                       self.input_shape)
            cropped = tf.reshape(cropped, [-1, 8, 8 , each_input.shape[-1]])
            cropped_feature_map_list.append(cropped)
            cropped_num = tf.shape(each_quadboxes)[1]

            roi_idx_b, roi_idx_n = tf.split(roi_idx, 2, axis=-1)

            roi_idx_n += cropped_num_acc
            roi_idx = tf.concat([roi_idx_b, roi_idx_n], axis=-1)
            cropped_num_acc += tf.cast(cropped_num, dtype=tf.int64)

            roi_idx_list.append(roi_idx)
            # stage_cropped_num.append(tf.shape(cropped)[0])

        # (M, FLAGS.crop_height, FLAGS.crop_width, each_input.shape[-1)
        whole_crop = tf.concat(cropped_feature_map_list, axis=0)
        whole_quad_boxes = tf.concat(quadboxes_list, axis=1)
        whole_roi_idx = tf.concat(roi_idx_list, axis=0)

        whole_crop = normalblock(whole_crop, 32, is_training=self.is_training)
        whole_crop = unpool(whole_crop)
        whole_crop = normalblock(whole_crop, 16, is_training=self.is_training)
        whole_crop = unpool(whole_crop)
        whole_crop = normalblock(whole_crop, 8, is_training=self.is_training)
        whole_crop = tf.layers.batch_normalization(whole_crop, training=self.is_training)
        sep_map = tf.layers.conv2d(inputs=whole_crop, filters=1, kernel_size=[1, 1], strides=[1, 1],
                                   padding="SAME", activation=tf.nn.sigmoid)
        score_map = tf.layers.conv2d(inputs=whole_crop, filters=1, kernel_size=[1, 1], strides=[1, 1],
                                     padding="SAME", activation=tf.nn.sigmoid)

        return tf.concat([score_map, sep_map], axis=-1), whole_roi_idx, None

    def loss_layer(self, segmap, input_quad_boxes, gt_boxes, roi_idx, gt_contours):
        """Build CTC Loss layer for training"""

        overlaps = Rboxiou(boxes=input_quad_boxes, quaryboxes=gt_boxes)
        overlaps_idx = tf.argmax(overlaps, -1)
        max_overlaps = tf.reduce_max(overlaps, -1)

        shape = tf.shape(overlaps_idx)
        batch_idx = tf.range(shape[0])
        batch_idx = tf.reshape(batch_idx, (shape[0], 1))
        b = tf.tile(batch_idx, (1, shape[1]))

        b = tf.cast(b, tf.int32)
        overlaps_idx = tf.cast(overlaps_idx, tf.int32)
        indexa = tf.stack([b, overlaps_idx], -1)
        gt_rearranged = tf.gather_nd(gt_contours, indexa)

        # (B, MAX_LENGTH, 8)
        # -> batch_cord(N'), tf.gather_nd(gt_boxes, indexa) (N',8)
        # (input? quadboxes, gt_rearranged, w,h)
        seg_shape = self.input_shape

        gt_picked = tf.gather_nd(gt_rearranged, roi_idx)  # (N' , 8)
        quad_boxes = tf.gather_nd(input_quad_boxes, roi_idx)
        max_overlaps_gather = tf.gather_nd(max_overlaps, roi_idx)

        max_overlaps_gather_mask = tf.where(max_overlaps_gather > 0.5, tf.ones_like(max_overlaps_gather),
                                            tf.zeros_like(max_overlaps_gather))
        max_overlaps_gather_iou_5 = tf.gather_nd(max_overlaps_gather,
                                                 tf.where(tf.equal(max_overlaps_gather_mask, 1)))
        gt_picked_iou_5 = tf.gather_nd(gt_picked,
                                       tf.where(tf.equal(max_overlaps_gather_mask, 1)))
        segmap_iou_5 = tf.gather_nd(segmap,
                                    tf.where(tf.equal(max_overlaps_gather_mask, 1)))
        quad_boxes_iou_5 = tf.gather_nd(quad_boxes,
                                        tf.where(tf.equal(max_overlaps_gather_mask, 1)))

        gt_cropped = tf.py_func(makeseggt, [quad_boxes_iou_5, gt_picked_iou_5,
                                            max_overlaps_gather_iou_5, seg_shape[2], seg_shape[1],
                                            (FLAGS.crop_seg_width, FLAGS.crop_seg_height)], tf.float32)

        gt_score, gt_sep, pred_mask = tf.split(value=gt_cropped, num_or_size_splits=3, axis=3)
        pred_score, pred_sep = tf.split(value=segmap_iou_5, num_or_size_splits=2, axis=3)

        classification_loss = tf.losses.mean_squared_error(gt_score * pred_mask, pred_score * pred_mask)
        classification_loss += tf.losses.mean_squared_error(gt_sep * pred_mask, pred_sep * pred_mask)

        tf.summary.image('gt_score', gt_score)
        tf.summary.image('gt_sep', gt_sep)
        tf.summary.image('gt_mask', pred_mask)
        tf.summary.image('pred_score', pred_score)
        tf.summary.image('pred_sep', pred_sep)
        return classification_loss


class Textregressor_fpn(object):
    def __init__(self, is_training, input_shape):
        self.is_training = is_training
        self.input_shape = input_shape

    def cal_quad_offset(self, offset_ch, whole_quad_boxes, whole_roi_idx):
        dx, dy, dw, dh, dtheta = tf.split(offset_ch, 5, -1)

        dtheta = (tf.layers.dense(inputs=dtheta, units=1,
                                  activation=tf.nn.sigmoid, name="reg_theta") - 0.5) * (np.pi / 2.0)
        offset_anchor_inverse = tf.concat(
            [dx, dy, dw, dh, dtheta],
            axis=-1)

        dx = tf.squeeze(dx, axis=-1)
        dy = tf.squeeze(dy, axis=-1)
        dw = tf.squeeze(dw, axis=-1)
        dh = tf.squeeze(dh, axis=-1)
        dtheta = tf.squeeze(dtheta, axis=-1)
        gathered_quadboxes = tf.gather_nd(whole_quad_boxes, whole_roi_idx)

        box_w, box_h, box_center_x, box_center_y, box_theta_1 = quad_to_param(gathered_quadboxes)

        # (B, H, W, NUM_ANCHOR)
        pred_center_x = box_center_x + box_w * dx
        pred_center_y = box_center_y + box_h * dy
        pred_w = box_w * tf.exp(dw)
        pred_h = box_h * tf.exp(dh)

        pred_w = tf.clip_by_value(pred_w, 0, tf.cast(self.input_shape[2], 'float32') - 1)
        pred_h = tf.clip_by_value(pred_h, 0, tf.cast(self.input_shape[1], 'float32') - 1)

        pred_theta = box_theta_1 + dtheta
        pred_theta = tf.where(pred_theta > np.pi,  np.pi - pred_theta, pred_theta)
        pred_theta = tf.where(pred_theta < -np.pi, - pred_theta, pred_theta)
        sin = tf.sin(pred_theta)
        cos = tf.cos(pred_theta)  # [b, w, h, 1]

        pred_x1 = (- pred_w / 2) * cos - (- pred_h / 2) * sin + pred_center_x
        pred_y1 = (- pred_w / 2) * sin + (- pred_h / 2) * cos + pred_center_y

        pred_x2 = (pred_w / 2) * cos - (- pred_h / 2) * sin + pred_center_x
        pred_y2 = (pred_w / 2) * sin + (- pred_h / 2) * cos + pred_center_y

        pred_x3 = (pred_w / 2) * cos - (pred_h / 2) * sin + pred_center_x
        pred_y3 = (pred_w / 2) * sin + (pred_h / 2) * cos + pred_center_y

        pred_x4 = (- pred_w / 2) * cos - (pred_h / 2) * sin + pred_center_x
        pred_y4 = (- pred_w / 2) * sin + (pred_h / 2) * cos + pred_center_y

        offset_ch = tf.stack([pred_x1, pred_y1,
                     pred_x2, pred_y2,
                     pred_x3, pred_y3,
                     pred_x4, pred_y4], -1)
        return offset_ch, offset_anchor_inverse

    def build_network(self, input_list, quadboxes_list):
        cropped_feature_map_list = []
        roi_idx_list = []
        cropped_num_acc = tf.constant(0, dtype=tf.int64)

        for each_input, each_quadboxes in zip(input_list, quadboxes_list):
            cropped, roi_idx, _, _ = roi_rotate_quad(each_input, each_quadboxes,
                                                     (FLAGS.crop_height, FLAGS.crop_width),
                                                     self.input_shape)
            cropped = tf.reshape(cropped, [-1, FLAGS.crop_height, FLAGS.crop_width, each_input.shape[-1]])
            cropped_feature_map_list.append(cropped)
            cropped_num = tf.shape(each_quadboxes)[1]

            roi_idx_b, roi_idx_n = tf.split(roi_idx, 2 , axis=-1)

            roi_idx_n += cropped_num_acc
            roi_idx = tf.concat([roi_idx_b, roi_idx_n], axis=-1)
            cropped_num_acc += tf.cast(cropped_num, dtype=tf.int64)

            roi_idx_list.append(roi_idx)

        # (M, FLAGS.crop_height, FLAGS.crop_width, each_input.shape[-1])
        whole_crop = tf.concat(cropped_feature_map_list, axis=0)
        whole_quad_boxes = tf.concat(quadboxes_list, axis=1)
        whole_roi_idx = tf.concat(roi_idx_list, axis=0)

        whole_crop = feature_extraction(whole_crop, self.is_training)
        whole_crop = tf.layers.batch_normalization(whole_crop, training=self.is_training)
        whole_crop = tf.reduce_mean(whole_crop, [1, 2], name='cropped_gap', keep_dims=True)
        offset_ch = tf.layers.conv2d(inputs = whole_crop, filters=5, kernel_size=[1, 1], strides=[1, 1], padding="same")
        offset_ch = tf.reduce_mean(offset_ch, [1, 2], name='cropped_gap_2', keep_dims=False)
        classifier_ch = tf.layers.conv2d(inputs=whole_crop, filters=1+FLAGS.num_class, kernel_size=[1, 1], strides=[1, 1], padding="same", name='calssifier')
        classifier_ch = tf.reduce_mean(classifier_ch, [1, 2], name='cropped_gap_3', keep_dims=False)
        offset_ch, offset_anchor_inverse = self.cal_quad_offset(offset_ch, whole_quad_boxes, whole_roi_idx)

        return offset_ch, whole_roi_idx, classifier_ch, offset_anchor_inverse

    def loss_layer(self, offest_ch, gt_boxes, roi_idx, quadboxes, classifier_ch, offset_anchor_inverse, box_label):
        """
        :param offest_ch:
        :param gt_boxes: (B, N(300), 8)
        :param roi_idx: (M, 2) -> (2) -> (Batch idx, N idx)
        :return:
        """

        # (B, NMS * 6 stage, FLAGS.gt_max_box)
        overlaps = Rboxiou(boxes=quadboxes, quaryboxes=gt_boxes)

        # (B, NMS * 6 stage)
        overlaps_idx = tf.argmax(overlaps, -1)
        shape = tf.shape(overlaps_idx)
        batch_idx = tf.range(shape[0])
        batch_idx = tf.reshape(batch_idx, (shape[0], 1))
        # (B, NMS * 6 stage)
        b = tf.tile(batch_idx, (1, shape[1]))
        b = tf.cast(b, tf.int32)

        overlaps_idx = tf.cast(overlaps_idx, tf.int32)

        # (B, NMS * 6stage, 2)
        indexa = tf.stack([b, overlaps_idx], -1)
        gt_rearranged = tf.gather_nd(gt_boxes, indexa)
        label_rearrange = tf.gather_nd(box_label, indexa)

        max_overlaps = tf.reduce_max(overlaps, -1)
        max_overlaps_gather = tf.gather_nd(max_overlaps, roi_idx)
        mask = tf.where(max_overlaps_gather > 0.5, tf.ones_like(max_overlaps_gather),
                        tf.zeros_like(max_overlaps_gather))
        positive_mask = tf.where(max_overlaps_gather >= 0.5, tf.ones_like(max_overlaps_gather),
                                 tf.zeros_like(max_overlaps_gather))
        negative_mask = tf.where(max_overlaps_gather < 0.5, tf.ones_like(max_overlaps_gather),
                                 tf.zeros_like(max_overlaps_gather))

        softmax_classifier_ch = tf.nn.softmax(classifier_ch, axis=-1)
        background_prediction = softmax_classifier_ch[:, 0]
        foreground_prediction = softmax_classifier_ch[:, 1:]

        reduced_softmax_foreground_prediction = tf.reduce_max(foreground_prediction, axis=-1)

        #zero = tf.zeros_like(background_prediction)

        pos_num = tf.reduce_sum(positive_mask)
        neg_num = tf.reduce_sum(negative_mask)
        min_num = tf.minimum(pos_num, neg_num)
        max_num = tf.maximum(pos_num, neg_num)

        pos_choice_num = tf.where(min_num > tf.cast(256 * shape[0], tf.float32), min_num // 2, min_num)
        neg_choice_num = tf.clip_by_value(tf.cast(512 * shape[0], tf.float32) - pos_choice_num, 0, neg_num)
        pos_choice_num = tf.cast(pos_choice_num, tf.int32)
        neg_choice_num = tf.cast(neg_choice_num, tf.int32)

        random_neg_choice_num = tf.cast(tf.cast(neg_choice_num, tf.float32) * 1.00, tf.int32)
        ohem_neg_choice_num = tf.cast(tf.cast(neg_choice_num, tf.float32) * 0.00, tf.int32)

        pos_choice_mask, neg_choice_mask = select_pos_neg_mask(reduced_softmax_foreground_prediction, background_prediction,
                                                               negative_mask, positive_mask, pos_choice_num,
                                                               random_neg_choice_num, ohem_neg_choice_num)

        tf.summary.scalar('classifier/pos_num', pos_num)
        tf.summary.scalar('classifier/min_num', min_num)
        tf.summary.scalar('classifier/neg_choice_num', neg_choice_num)
        tf.summary.scalar('classifier/max_num', max_num)
        tf.summary.scalar('classifier/random_neg_choice_num', random_neg_choice_num)
        tf.summary.scalar('classifier/ohem_neg_choice_num', ohem_neg_choice_num)

        score_mask = tf.logical_or(tf.cast(pos_choice_mask, tf.bool), tf.cast(neg_choice_mask, tf.bool))
        score_mask = tf.cast(score_mask, tf.float32)
        gt_ch = tf.gather_nd(gt_rearranged, roi_idx)
        gt_class = tf.gather_nd(label_rearrange, roi_idx)
        masked_gt_class = tf.cast(pos_choice_mask, tf.int32) * gt_class
        masked_gt_class = tf.clip_by_value(masked_gt_class, 0, FLAGS.num_class)
        masked_gt_class = tf.cast(masked_gt_class, tf.int32)

        gathered_quadboxes = tf.gather_nd(quadboxes, roi_idx)

        gather_regressor_value = tf.gather_nd(offset_anchor_inverse,
                                              tf.where(tf.equal(pos_choice_mask, 1)))
        gather_gt_rearranged = tf.gather_nd(gt_ch,
                                            tf.where(tf.equal(pos_choice_mask, 1)))
        gather_loss_anchor_box = tf.gather_nd(gathered_quadboxes,
                                              tf.where(tf.equal(pos_choice_mask, 1)))

        gt_regressor = bbox_inverse_transform(gather_loss_anchor_box, gather_gt_rearranged)
        loss = tf.losses.mean_squared_error(gt_regressor, gather_regressor_value)

        batch_size = tf.shape(classifier_ch)[0]
        batch_linspace = tf.range(0, batch_size)
        batch_linspace = tf.reshape(batch_linspace, [-1])
        gt_gather_idx = tf.stack([batch_linspace, masked_gt_class], axis=-1)

        loss_log = tf.log(tf.nn.softmax(classifier_ch) + 1e-5)
        loss_gather_nd = tf.gather_nd(loss_log, gt_gather_idx)
        score_loss = tf.reduce_mean(-1. * loss_gather_nd * score_mask)
        #score_loss = tf.where(tf.math.is_nan(score_loss), tf.zeros_like(score_loss), score_loss)
        #loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)


        # debug = tf.py_func(debug_loss, ["score_loss", score_loss, "classifier_ch", classifier_ch,
        #                                 "offset_anchor_inverse", offset_anchor_inverse], tf.int64)
        # debug = tf.cast(debug, tf.float32)

        # debug2 = tf.py_func(debug_loss, ["loss", loss, "gt_regressor", gt_regressor,
        #                                 "gather_regressor_value", gather_regressor_value], tf.int64)
        # debug2 = tf.cast(debug2, tf.float32)

        tf.summary.scalar('regressor_loss', loss)
        tf.summary.scalar('classifier_loss', score_loss)
        debugimg = tf.py_func(debugclassifier,
                              [quadboxes, gt_ch, roi_idx, offest_ch, mask, reduced_softmax_foreground_prediction],
                              tf.float32)
        debugimg = tf.reshape(debugimg, [FLAGS.batch_size, FLAGS.input_size, FLAGS.input_size, 3])

        debugclassimg = tf.py_func(debugclass,
                                   [quadboxes, box_label, roi_idx, gt_boxes, mask, foreground_prediction],
                                   tf.float32)
        debugclassimg = tf.reshape(debugclassimg, [FLAGS.batch_size, FLAGS.input_size, FLAGS.input_size, 3])

        tf.summary.image('debugimg', debugimg)
        tf.summary.image('debugclassimg', debugclassimg)

        return score_loss + loss


def debug_loss(name_1, loss_1, name_2, loss_2, name_3, loss_3 ):
    name_1 = name_1.decode("utf-8")
    name_2 = name_2.decode("utf-8")
    name_3 = name_3.decode("utf-8")
    print("{} : {}".format(name_1, loss_1))
    print("{} : {}".format(name_2, loss_2))
    print("{} : {}".format(name_3, loss_3))
    return 0


class TextmapNetwork(object):
    def __init__(self, inputs, backbone, param, mode="train", gt=None):
        self.input = inputs
        self.input_shape = tf.shape(self.input)
        self.mode = mode
        self.backbone = backbone
        if self.mode == "train":
            self.is_training = True
        else:
            self.is_training = False
        self.gt = gt

        self.textregressor_fpn = Textregressor_fpn(is_training=self.is_training, input_shape=self.input_shape)
        self.textmask = Textmask(is_training=self.is_training, input_shape=self.input_shape)
        self.rpn_1 = Rpn(param, 512, is_training=self.is_training)
        self.rpn_2 = Rpn(param, 256, is_training=self.is_training)
        self.rpn_3 = Rpn(param, 128, is_training=self.is_training)
        self.rpn_4 = Rpn(param, 64, is_training=self.is_training)
        self.rpn_5 = Rpn(param, 32, is_training=self.is_training)

        self.rpn_obj_list = [self.rpn_1, self.rpn_2, self.rpn_3, self.rpn_4, self.rpn_5]

    def _build_base(self):
        if self.backbone == "resnet101":
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                logits, end_points = resnet_v1.resnet_v1_101(self.input,
                                                             is_training=self.is_training, scope='resnet_v1_101')
        elif self.backbone == "resnet50":
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                logits, end_points = resnet_v1.resnet_v1_50(self.input,
                                                            is_training=self.is_training, scope='resnet_v1_50')
        elif self.backbone == "inception_resnet":
            raise ValueError
        elif self.backbone == "custom_resnet":
            net = resnet.Resnet()
            logits, end_points = net.make_block(self.input, 50, self.is_training)
        else:
            raise ValueError

        return logits, end_points

    def build_pixel_based_network(self, feature_lists, input_shape, idx):
        multiple_list = [32.0, 16.0, 8.0, 4.0, 2.0]
        feature_lists_bn = tf.layers.batch_normalization(feature_lists[idx], training=self.is_training)
        small_score_map = tf.layers.conv2d(inputs=feature_lists_bn, filters=2,
                                           kernel_size=[1, 1], strides=[1, 1],
                                           padding="SAME")
        geo_map = tf.exp(tf.layers.conv2d(inputs=feature_lists_bn, filters=4,
                                          kernel_size=[1, 1], strides=[1, 1],
                                          padding="SAME"))
        theta_map = (tf.layers.conv2d(inputs=feature_lists_bn, filters=1,
                                      kernel_size=[1, 1], strides=[1, 1],
                                      padding="SAME", activation=tf.nn.sigmoid,
                                      name="theta_map_{}".format(idx + 1)) - 0.5) * np.pi / 2.0
        geo_concat = tf.concat([geo_map, small_score_map, theta_map], axis=-1)
        boxes, score, thres_score, w, h = getboxes(geo_concat, multiple_list[idx])
        reshape_boxes = tf.reshape(boxes, [input_shape[0], -1, 8])
        reshape_score = tf.reshape(score, [input_shape[0], -1, 1])

        reshape_boxes, reshape_score, reshape_idx = filter_score_idx(reshape_boxes, reshape_score, 0.1)

        reshape_boxes, reshape_score = tf.py_func(standard_nms_idx,
                                                  [reshape_boxes, reshape_score, reshape_idx,
                                                   input_shape[0],
                                                   FLAGS.num_nms],
                                                  [tf.float32, tf.float32])
        return reshape_boxes, reshape_score, geo_concat

    def top_k_and_gather(self, rpn_reg_proposal_box, rpn_reg_proposal_score):
        rpn_reg_proposal_box_shape = tf.shape(rpn_reg_proposal_box)
        rpn_reg_proposal_score_shape = tf.shape(rpn_reg_proposal_score)

        sort_value, sort_idx = tf.nn.top_k(tf.reshape(rpn_reg_proposal_score, [-1]),
                                           tf.cast(rpn_reg_proposal_box_shape[0] * FLAGS.num_refine_proposal, tf.int32))
        rpn_reg_proposal_box_1d = tf.reshape(rpn_reg_proposal_box, [-1, 8])
        gather_rpn_reg_proposal_box_1d = tf.gather(rpn_reg_proposal_box_1d, sort_idx)
        rpn_reg_proposal_box = tf.scatter_nd(tf.reshape(sort_idx, [-1, 1]), gather_rpn_reg_proposal_box_1d,
                                             tf.shape(rpn_reg_proposal_box_1d))

        rpn_reg_proposal_box = tf.reshape(rpn_reg_proposal_box, rpn_reg_proposal_box_shape)

        rpn_reg_proposal_score_1d = tf.reshape(rpn_reg_proposal_score, [-1, 1])
        gather_rpn_reg_proposal_score_1d = tf.gather(rpn_reg_proposal_score_1d, sort_idx)
        rpn_reg_proposal_score = tf.scatter_nd(tf.reshape(sort_idx, [-1, 1]), gather_rpn_reg_proposal_score_1d,
                                               tf.shape(rpn_reg_proposal_score_1d))

        rpn_reg_proposal_score = tf.reshape(rpn_reg_proposal_score, rpn_reg_proposal_score_shape)
        rpn_reg_proposal_list = tf.split(value=rpn_reg_proposal_box, num_or_size_splits=5, axis=1)
        return rpn_reg_proposal_list, rpn_reg_proposal_score

    def build_network(self):
        input_shape = tf.shape(self.input)
        self.logits, self.endpoint = self._build_base()

        self.endpoint["pool6"] = normalblock(self.endpoint["pool5"], 256, self.is_training, (2, 2), no_activation=True)
        self.endpoint["pool2"] = unpool(self.endpoint["pool2"])
        self.endpoint["pool3"] = unpool(self.endpoint["pool3"])
        self.endpoint["pool4"] = unpool(self.endpoint["pool4"])

        feature_num = 256
        feature_lists = []  # [self.endpoint["pool6"]]

        for i in range(5):
            output = self.endpoint["pool" + str(5 - i)]
            if i == 0:
                h_before = self.endpoint["pool" + str(6 - i)]
            h_before_upsample = upsample(h_before, tf.shape(output))
            new_output = tf.concat([h_before_upsample, output], -1)
            new_output = normalblock(new_output, feature_num, self.is_training, no_activation=True)
            new_output = tf.nn.sigmoid(new_output)

            h_before = new_output
            feature_lists.append(h_before)

        stage = len(self.rpn_obj_list)
        rpn_res_list = []
        reg_res_list = []

        for i in range(stage):
            with tf.variable_scope("rpn_{}".format(i+1)):
                rpn_res_list.append(self.rpn_obj_list[i].build_network([feature_lists[i]], tf.shape(self.input)))

        for i in range(stage):
            with tf.variable_scope("reg_{}".format(i+1)):
                reg_res_list.append(self.build_pixel_based_network(feature_lists, input_shape, i))

        rpn_proposal_box_list = []
        rpn_proposal_score_list = []

        reg_proposal_box_list = []
        reg_proposal_score_list = []

        stage_box_list = []
        stage_score_list = []
        geo_concat_list = []

        for each_rpn, each_reg in zip(rpn_res_list, reg_res_list):
            rpn_proposal_box_list.append(each_rpn[0][0])
            rpn_proposal_score_list.append(each_rpn[0][1])

            reg_proposal_box_list.append(each_reg[0])
            reg_proposal_score_list.append(each_reg[1])

            each_stage_box = tf.concat([each_rpn[0][0], each_reg[0]], axis=1)
            stage_box_list.append(each_stage_box)

            each_stage_score = tf.concat([each_rpn[0][1], each_reg[1]], axis=1)
            stage_score_list.append(each_stage_score)

            geo_concat_list.append(each_reg[2])

        rpn_proposal_box = tf.concat(rpn_proposal_box_list, axis=1)
        rpn_proposal_score = tf.concat(rpn_proposal_score_list, axis=1)

        rpn_debug = tf.py_func(makepoly_with_score,
                               [input_shape[0], rpn_proposal_box,
                                rpn_proposal_score, input_shape[1],
                                input_shape[2], self.input, 0.8], [tf.float32])
        rpn_debug = tf.reshape(rpn_debug, [input_shape[0], input_shape[1], input_shape[2], 3])
        tf.summary.image('rpn_debug', rpn_debug)

        reshape_rpn_boxes = tf.reshape(rpn_proposal_box, [input_shape[0], -1, 8])
        reshape_rpn_score = tf.reshape(rpn_proposal_score, [input_shape[0], -1, 1])

        rpn_reg_proposal_box = tf.concat(stage_box_list, axis=1)
        rpn_reg_proposal_score = tf.concat(stage_score_list, axis=1)

        reshape_boxes = tf.concat(reg_proposal_box_list, axis=1)
        reshape_score = tf.concat(reg_proposal_score_list, axis=1)

        debug_reg_whole = tf.py_func(makepoly_with_score,
                                     [input_shape[0], reshape_boxes,
                                      reshape_score, input_shape[1],
                                      input_shape[2], self.input, 0.8], [tf.float32])
        debug_reg_whole = tf.reshape(debug_reg_whole, [input_shape[0], input_shape[1], input_shape[2], 3])
        tf.summary.image('reg_debug', debug_reg_whole)

        rpn_reg_proposal_list, rpn_reg_proposal_score = self.top_k_and_gather(rpn_reg_proposal_box,
                                                                              rpn_reg_proposal_score)

        with tf.variable_scope("refine"):
            offset_ch_fpn_whole, roi_idx_fpn_whole, \
            classifier_ch_fpn_whole, offset_anchor_inverse_fpn_whole = self.textregressor_fpn.build_network(
                feature_lists, rpn_reg_proposal_list)

        with tf.variable_scope("refine", reuse=True):
            refined_offset_fpn_whole = tf.scatter_nd(roi_idx_fpn_whole, offset_ch_fpn_whole,
                                                     tf.cast(tf.shape(rpn_reg_proposal_box), tf.int64))
            refined_boxes_fpn_list = tf.split(value=refined_offset_fpn_whole, num_or_size_splits=5, axis=1)
            offset_ch_fpn_whole_refine, roi_idx_fpn_whole_refine, \
            classifier_ch_fpn_whole_refine, offset_anchor_inverse_fpn_whole_refine = self.textregressor_fpn.build_network(
                feature_lists, refined_boxes_fpn_list)

        geo_concat_list.append(reshape_boxes)
        geo_concat_list.append(reshape_score)
        geo_concat_list.append(reshape_rpn_boxes)
        geo_concat_list.append(reshape_rpn_score)

        return geo_concat_list, rpn_res_list[0], rpn_res_list[1], rpn_res_list[2], rpn_res_list[3], rpn_res_list[4],\
               [offset_ch_fpn_whole, roi_idx_fpn_whole, rpn_reg_proposal_box,
                classifier_ch_fpn_whole, offset_anchor_inverse_fpn_whole,
                offset_ch_fpn_whole_refine, roi_idx_fpn_whole_refine, refined_offset_fpn_whole,
                classifier_ch_fpn_whole_refine, offset_anchor_inverse_fpn_whole_refine]

    def add_loss(self, y_true_geo, y_pred_geo, training_mask, div, min, max):
        y_true_geo = y_true_geo[:, ::div, ::div, :]
        training_mask = training_mask[:, ::div, ::div, :]
        ones = tf.ones_like(training_mask, dtype=tf.bool)
        zeros = tf.zeros_like(training_mask, dtype=tf.bool)
        shape = tf.shape(y_true_geo)

        d1_gt, d2_gt, d3_gt, d4_gt, score_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=6, axis=3)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)

        size_clip = tf.where(tf.logical_and(area_gt >= min, area_gt < max), ones, zeros)
        specific_range = tf.cast(size_clip, tf.float32)

        gt_real_ch = specific_range * score_gt
        y_neg_map = 1 - gt_real_ch
        y_pos_map = gt_real_ch
        pos_num = tf.reduce_sum(gt_real_ch)
        neg_num = tf.reduce_sum(1 - gt_real_ch)
        pos_num = tf.cast(pos_num, tf.int32)
        neg_num = tf.cast(neg_num, tf.int32)

        min_num = tf.minimum(tf.cast(tf.cast(pos_num, tf.float32), tf.int32), neg_num)
        min_num = tf.minimum(256 * shape[0], min_num)

        pos_choice_num = tf.where(min_num > 128 * shape[0], min_num // 2, min_num)
        neg_choice_num = tf.clip_by_value(256 * shape[0] - pos_choice_num, 0, neg_num)
        pos_choice_mask, neg_choice_mask = select_pos_neg_reg_mask(y_pos_map, y_neg_map, tf.shape(training_mask), pos_choice_num, neg_choice_num, 0)

        update_mask = tf.logical_or(tf.cast(pos_choice_mask, tf.bool), tf.cast(neg_choice_mask, tf.bool))
        update_mask = tf.cast(update_mask, tf.float32)

        d1_pred, d2_pred, d3_pred, d4_pred, score_bk, score_pred, theta_pred = tf.split(value=y_pred_geo,
                                                                                        num_or_size_splits=7, axis=3)
        score_bce = tf.concat([score_bk, score_pred], axis=-1)
        classification_loss = tf.losses.sparse_softmax_cross_entropy(
            tf.cast(tf.squeeze(gt_real_ch, axis=-1), tf.int64),
            score_bce, tf.squeeze(update_mask*training_mask, axis=-1))
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
        w_union = tf.maximum(0.0, w_union)
        h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
        h_union = tf.maximum(0.0, h_union)

        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        L_AABB = -tf.log((area_intersect + 1.0) / (area_union + 1.0))
        L_theta = 1 - tf.cos(theta_pred - theta_gt)
        L_g = L_AABB + 20 * L_theta
        L_geometry_AABB = tf.reduce_mean(L_g * tf.cast(gt_real_ch  * update_mask, tf.float32) * training_mask)
        tf.summary.scalar('geometry_AABB', L_geometry_AABB)
        tf.summary.image('gt_real_ch', gt_real_ch)
        tf.summary.image('update_mask', update_mask)
        tf.summary.image('pos_choice_mask', tf.cast(pos_choice_mask, tf.float32))
        tf.summary.image('neg_choice_mask', tf.cast(neg_choice_mask, tf.float32))

        # debug = tf.py_func(debug_loss, ["L_geometry_AABB", L_geometry_AABB, "L_classification_loss", classification_loss,
        #                                 "L_AABB", tf.reduce_mean(L_AABB)], tf.int64)
        # debug = tf.cast(debug, tf.float32)

        return L_geometry_AABB + classification_loss

