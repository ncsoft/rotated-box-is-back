import tensorflow as tf
import numpy as np
import cv2

from nms import rbox_gpu_nms
from utils import bbox_inverse_transform
from roi_rotate import rotate_np
from debug_utils import viz_pos_neg_anchor

FLAGS = tf.app.flags.FLAGS
nms_kernel = tf.load_op_library('./nms/nms_kernel_op.so')


def Rboxiou(boxes, quaryboxes):
    return nms_kernel.Rboxiou(boxes=boxes, quaryboxes=quaryboxes)


def filter_score_idx(boxes, score, thres):
    score = tf.squeeze(score, axis=-1)
    thres_tensor = tf.fill(tf.shape(score), thres)
    score_idx = tf.where(tf.greater(score, thres_tensor))
    filter_score = tf.gather_nd(score, score_idx)
    filter_boxes = tf.gather_nd(boxes, score_idx)
    filter_score = tf.expand_dims(filter_score, axis=-1)
    return filter_boxes, filter_score, score_idx


def standard_nms_idx(boxes, score, info_idx, batchsize, k):
    np_after_boxes = np.zeros([batchsize, k, 8], dtype=np.float32)
    np_after_score = np.zeros([batchsize, k, 1], dtype=np.float32)
    np_nms_input = np.concatenate([boxes, score], -1)
    info_idx_shape = info_idx.shape
    if len(info_idx_shape) != 2 or info_idx_shape[0] == 0:
        return np_after_boxes, np_after_score
    batch_info = info_idx[:, 0]
    for i in range(batchsize):
        batch_idx = np.where(batch_info == i)

        np_nms_input_thres = np_nms_input[batch_idx]
        idx = np.argsort(np_nms_input_thres[:, -1])[::-1]
        cut_off = min(k*10, np_nms_input_thres.shape[0])
        np_nms_input_thres = np_nms_input_thres[idx]

        np_nms_input_thres = np_nms_input_thres[:cut_off]

        np_after_nms_idx = rbox_gpu_nms(np_nms_input_thres, 0.5)
        np_after_nms = np_nms_input_thres[np_after_nms_idx.astype(np.int32)]
        max_nms_length = np_after_nms.shape[0]
        if max_nms_length == 0:
            continue
        np_after_nms_score = np_after_nms[:, -1]
        max_nms_length = min(max_nms_length, k)

        np_after_boxes[i, :max_nms_length] = np_after_nms[:max_nms_length, :8]
        np_after_score[i, :max_nms_length, 0] = np_after_nms_score[:max_nms_length]

    return np_after_boxes, np_after_score


def get_uniform_sample(input_length, choice_num):
    uniform_distribution = tf.random.uniform(
        shape=input_length,
        minval=0,
        maxval=None,
        dtype=tf.float32,
        seed=None,
        name=None
    )
    sample_value, sample_idx = tf.nn.top_k(uniform_distribution, choice_num)
    return sample_value, sample_idx

def select_mask(y_map, original_mask_shape, random_choice_num, ohem_choice_num):
    # (B*N')
    y_map_1d = tf.reshape(y_map, [-1])

    y_map_first_idx = tf.cast(tf.where(tf.cast(y_map_1d, tf.bool)), tf.int32)
    # (M)
    y_map_gather = tf.gather_nd(y_map_1d, y_map_first_idx)
    y_map_gather_shape = tf.shape(y_map_gather)

    # (M)
    y_map_gather_1d = tf.reshape(y_map_gather, [-1])
    y_input_length = tf.shape(y_map_gather_1d)
    y_sample_value, y_sample_idx = get_uniform_sample(y_input_length, random_choice_num)

    # (M)
    y_map_gather_sample_recon = tf.scatter_nd(tf.reshape(y_sample_idx, [-1, 1]), tf.ones(random_choice_num),
                                              y_input_length)
    y_map_gather_recon = tf.reshape(y_map_gather_sample_recon, y_map_gather_shape)

    # (B*N')
    choice_mask = tf.scatter_nd(tf.reshape(y_map_first_idx, [-1, 1]), y_map_gather_recon,
                                tf.shape(y_map_1d))
    choice_mask = tf.reshape(choice_mask, original_mask_shape)

    if ohem_choice_num == 0:
        choice_mask = tf.cast(choice_mask, tf.int32)
        return choice_mask

    value, idx = tf.nn.top_k(y_map_1d, tf.cast(ohem_choice_num * 10, tf.int32))
    ohem_choice_mask = tf.scatter_nd(tf.reshape(idx, [-1, 1]), tf.ones(tf.cast(ohem_choice_num * 10, tf.int32)),
                                     tf.shape(y_map_1d))
    ohem_choice_mask = tf.reshape(ohem_choice_mask, original_mask_shape)
    ohem_choice_mask_1d = tf.reshape(ohem_choice_mask, [-1])
    ohem_choice_mask_map_first_idx = tf.cast(tf.where(tf.cast(ohem_choice_mask_1d, tf.bool)), tf.int32)

    ohem_choice_mask_map_gather = tf.gather_nd(ohem_choice_mask_1d, ohem_choice_mask_map_first_idx)
    ohem_choice_mask_map_gather_shape = tf.shape(ohem_choice_mask_map_gather)

    # (M)
    ohem_choice_mask_map_gather_1d = tf.reshape(ohem_choice_mask_map_gather, [-1])
    ohem_choice_input_length = tf.shape(ohem_choice_mask_map_gather_1d)
    ohem_sample_value, ohem_sample_idx = get_uniform_sample(ohem_choice_input_length, ohem_choice_num)

    # (M)
    ohem_map_gather_sample_recon = tf.scatter_nd(tf.reshape(ohem_sample_idx, [-1, 1]),
                                                 tf.ones(ohem_choice_num), ohem_choice_input_length)
    ohem_gather_recon = tf.reshape(ohem_map_gather_sample_recon, ohem_choice_mask_map_gather_shape)

    # (B*N')
    ohem_random_choice_mask = tf.scatter_nd(tf.reshape(ohem_choice_mask_map_first_idx, [-1, 1]),
                                            ohem_gather_recon, tf.shape(y_map_1d))
    ohem_random_choice_mask = tf.reshape(ohem_random_choice_mask, original_mask_shape)

    choice_mask = tf.logical_or(tf.cast(choice_mask, tf.bool), tf.cast(ohem_random_choice_mask, tf.bool))
    choice_mask = tf.cast(choice_mask, tf.int32)
    return choice_mask


def select_compare_mask(anchor_mask, compare_tensor, random_choice_num, ohem_choice_num):
    zero = tf.zeros_like(compare_tensor)
    # (B, N' ,1)
    y_map = tf.where(tf.cast(anchor_mask, tf.bool), compare_tensor + 1e-3, zero)
    return select_mask(y_map, tf.shape(compare_tensor), random_choice_num, ohem_choice_num)


def select_pos_neg_reg_mask(pos_reg_mask, neg_reg_mask, output_shape, pos_choice_num, neg_choice_num, ohem_neg_choice_num):
    neg_choice_mask = select_mask(neg_reg_mask, output_shape, neg_choice_num, ohem_neg_choice_num)
    pos_choice_mask = select_mask(pos_reg_mask, output_shape, pos_choice_num, 0)

    return pos_choice_mask, neg_choice_mask


def select_pos_neg_mask(neg_compare, pos_compare, neg_anchor_mask, pos_anchor_mask, pos_choice_num, neg_choice_num, ohem_neg_choice_num):
    neg_choice_mask = select_compare_mask(neg_anchor_mask, neg_compare, neg_choice_num, ohem_neg_choice_num)
    pos_choice_mask = select_compare_mask(pos_anchor_mask, pos_compare, pos_choice_num, 0)

    return pos_choice_mask, neg_choice_mask


def whole_rpn_loss(concat_anchor_box, concat_score_box, concat_anchor_offset, quadboxes):
    """
    :param concat_anchor_box: (B, N', 8*2)
    :param concat_score_box: (B, N', 2)
    :param concat_anchor_offset:
    :param quadboxes: (B, N'', 8)
    :return:
    """
    # (B, N', N'')
    pred_concat_anchor_box = concat_anchor_box[:, :, :8]  # pred
    concat_anchor_box = concat_anchor_box[:, :, 8:]  # anchor

    score_neg, score_pos = tf.split(tf.nn.softmax(concat_score_box, axis=-1), num_or_size_splits=2,
                                    axis=-1)  # = concat_score_box#
    overlaps = nms_kernel.Rboxiou(boxes=concat_anchor_box, quaryboxes = quadboxes)
    # (B, N')
    shape = tf.shape(overlaps)
    overlaps_idx = tf.argmax(overlaps, -1)
    batch_idx = tf.range(shape[0])
    batch_idx = tf.reshape(batch_idx, (shape[0], 1))
    b = tf.tile(batch_idx, (1, shape[1]))
    # (B,N')
    b = tf.cast(b, tf.int32)

    overlaps_idx = tf.cast(overlaps_idx, tf.int32)
    # (B, N', 2)
    indexa = tf.stack([b, overlaps_idx], -1)
    # (B, N', 8)
    gt_rearranged = tf.gather_nd(quadboxes, indexa)

    # (B, N'')
    gt_overlap_idx = tf.argmax(overlaps, -2)
    gt_max_overlaps = tf.reduce_max(overlaps, -2)
    gt_overlap_idx = tf.cast(gt_overlap_idx, tf.int32)
    b2 = tf.tile(batch_idx, (1, shape[2]))
    b2 = tf.cast(b2, tf.int32)
    exclude_zero_idx = tf.where(gt_max_overlaps > 0.01)

    indexb = tf.stack([b2, gt_overlap_idx], -1)
    exclude_indexb = tf.gather_nd(indexb, exclude_zero_idx)

    pos_th = 0.70
    neg_th = 0.30

    pred_overlaps = tf.reduce_max(overlaps, -1)

    anchor_mask = tf.where(pred_overlaps > pos_th, tf.ones_like(pred_overlaps),
                           tf.zeros_like(pred_overlaps))
    neg_anchor_mask = tf.where(pred_overlaps < neg_th, tf.ones_like(pred_overlaps),
                               tf.zeros_like(pred_overlaps))

    reshape_indexb = tf.reshape(exclude_indexb, [-1, 2])
    reshape_size = tf.shape(reshape_indexb)
    gt_mask = tf.scatter_nd(reshape_indexb, tf.ones(reshape_size[0]), tf.shape(pred_overlaps))

    mask = tf.logical_or(tf.cast(anchor_mask, tf.bool), tf.cast(gt_mask, tf.bool))

    mask = tf.cast(mask, tf.int32)
    mask = tf.expand_dims(mask, -1)
    neg_anchor_mask = tf.expand_dims(neg_anchor_mask, -1)
    pos_anchor_mask = mask
    pos_num = tf.reduce_sum(pos_anchor_mask)
    neg_num = tf.reduce_sum(neg_anchor_mask)
    neg_num = tf.cast(neg_num, tf.int32)
    min_num = tf.minimum(tf.cast(tf.cast(pos_num, tf.float32), tf.int32), neg_num)
    min_num = tf.minimum(256 * shape[0], min_num)

    pos_choice_num = tf.where(min_num > 128 * shape[0], min_num // 2, min_num)
    neg_choice_num = tf.clip_by_value(256 * shape[0] - pos_choice_num, 0, neg_num)
    random_neg_choice_num = tf.cast(tf.cast(neg_choice_num, tf.float32) * 1.00, tf.int32)
    ohem_neg_choice_num = tf.cast(tf.cast(neg_choice_num, tf.float32) * 0.00, tf.int32)

    pos_choice_mask, neg_choice_mask = select_pos_neg_mask(score_pos, score_pos, neg_anchor_mask,
                                                           pos_anchor_mask, pos_choice_num,
                                                           random_neg_choice_num, ohem_neg_choice_num)

    tf.summary.scalar('pos_num', pos_num)
    tf.summary.scalar('neg_num', neg_num)
    tf.summary.scalar('neg_choice_num', neg_choice_num)
    tf.summary.scalar('random_neg_choice_num', random_neg_choice_num)
    tf.summary.scalar('ohem_neg_choice_num', ohem_neg_choice_num)
    tf.summary.scalar('pos_choice_num', pos_choice_num)

    score_mask = tf.logical_or(tf.cast(pos_choice_mask, tf.bool), tf.cast(neg_choice_mask, tf.bool))
    score_mask = tf.cast(score_mask, tf.float32)

    gather_regressor_value = tf.gather_nd(concat_anchor_offset, tf.where(tf.equal(tf.squeeze(pos_choice_mask, axis=-1), 1)))
    gather_gt_rearranged = tf.gather_nd(gt_rearranged,
                                        tf.where(tf.equal(tf.squeeze(pos_choice_mask, axis=-1), 1)))
    gather_loss_anchor_box = tf.gather_nd(concat_anchor_box,
                                          tf.where(tf.equal(tf.squeeze(pos_choice_mask, axis=-1), 1)))

    gt_regerssor = bbox_inverse_transform(gather_loss_anchor_box, gather_gt_rearranged)

    box_loss = tf.losses.mean_squared_error(gt_regerssor, gather_regressor_value)
    score_loss = tf.losses.sparse_softmax_cross_entropy(tf.squeeze(pos_choice_mask, axis=-1), concat_score_box,
                                                        tf.squeeze(score_mask, axis=-1))

    tf.summary.scalar('box_loss', box_loss)
    tf.summary.scalar('score_loss', score_loss)

    viz_pos_img, vis_neg_img = tf.py_func(viz_pos_neg_anchor, [concat_anchor_box, score_pos, pos_choice_mask,
                                                               score_mask,
                                                               (FLAGS.batch_size, FLAGS.input_size, FLAGS.input_size, 3)], [tf.float32, tf.float32])

    tf.summary.image('viz_pos_img', viz_pos_img)
    tf.summary.image('vis_neg_img', vis_neg_img)

    viz_pos_pred_img, vis_neg_pred_img = tf.py_func(viz_pos_neg_anchor, [pred_concat_anchor_box, score_pos, pos_choice_mask, score_mask, (
    FLAGS.batch_size, FLAGS.input_size, FLAGS.input_size, 3)],
                                          [tf.float32, tf.float32])

    tf.summary.image('viz_pos_pred_img', viz_pos_pred_img)
    tf.summary.image('vis_neg_pred_img', vis_neg_pred_img)

    return score_loss + box_loss


def makeseggt(quadboxes, gt_contours, max_overlaps, w, h, crop_size):
    """
    :param quadboxes: (N, 8)
    :param gt_contours: (N , None, 2)
    :param max_overlaps:
    :param w:
    :param h:
    :return:
    """

    crop_width, crop_height = crop_size
    crop_img = []
    one_mask = np.ones([crop_height, crop_width])
    zero_mask = np.zeros([crop_height, crop_width])
    zero_mask_stack = np.stack([zero_mask, zero_mask, zero_mask], -1)

    if quadboxes.shape[0] == 0:
        return np.zeros([0, crop_height, crop_width, 3], dtype=np.float32)
    for quadbox, gt_contour, each_max_overlaps in zip(quadboxes, gt_contours, max_overlaps):
        if each_max_overlaps < 0.5:
            crop_img.append(zero_mask_stack)
            continue
        else:
            gt_contour = gt_contour.reshape([-1, 2]).astype(np.int32)
            gt_contour_idx = np.where(np.logical_and(gt_contour[:, 0] != 0, gt_contour[:, 1] != 0))
            gt_contour = gt_contour[gt_contour_idx]
            if len(gt_contour) == 0:
                crop_img.append(zero_mask_stack)
                continue
            quadbox = quadbox.reshape([4, 2]).astype(np.int32)
            contour_x_max = np.amax(gt_contour[:, 0])
            contour_x_min = np.amin(gt_contour[:, 0])
            contour_y_max = np.amax(gt_contour[:, 1])
            contour_y_min = np.amin(gt_contour[:, 1])

            x_max = np.amax(quadbox[:, 0])
            x_min = np.amin(quadbox[:, 0])
            y_max = np.amax(quadbox[:, 1])
            y_min = np.amin(quadbox[:, 1])

            if y_min >= y_max and x_min >= x_max:
                crop_img.append(zero_mask_stack)
                continue

            two_x_max = np.maximum(contour_x_max, x_max)
            two_x_min = np.minimum(contour_x_min, x_min)
            two_y_max = np.maximum(contour_y_max, y_max)
            two_y_min = np.minimum(contour_y_min, y_min)

            if two_y_min >= two_y_max or two_x_min >= two_x_max:
                crop_img.append(zero_mask_stack)
                continue

            gt_contour[:, 0] -= two_x_min
            gt_contour[:, 1] -= two_y_min

            quadbox[:, 0] -= two_x_min
            quadbox[:, 1] -= two_y_min

            x_max -= two_x_min
            x_min -= two_x_min
            y_max -= two_y_min
            y_min -= two_y_min

            two_x_min -= two_x_min
            two_x_max -= two_x_min
            two_y_min -= two_y_min
            two_y_max -= two_y_min

            vector = (quadbox[1] - quadbox[0]) + (quadbox[2] - quadbox[3])
            eps = 0.00001

            x_vector1 = vector[0] * (crop_width / float(x_max - x_min + eps))
            y_vector1 = vector[1] * (crop_height / float(y_max - y_min + eps))

            angle = np.arctan(y_vector1 / (x_vector1 + eps))
            final_angle = angle
            rec_length_maximum = np.maximum((two_x_max - two_x_min), (two_y_max - two_y_min))
            max_size = 100

            def norm_and_resize(inputs, roi_max_size, output_size):
                return output_size * inputs / float(roi_max_size)

            two_x_max = norm_and_resize(two_x_max, rec_length_maximum, max_size)
            two_x_min = norm_and_resize(two_x_min, rec_length_maximum, max_size)

            two_y_max = norm_and_resize(two_y_max, rec_length_maximum, max_size)
            two_y_min = norm_and_resize(two_y_min, rec_length_maximum, max_size)

            x_max = norm_and_resize(x_max, rec_length_maximum, max_size)
            x_min = norm_and_resize(x_min, rec_length_maximum, max_size)

            y_max = norm_and_resize(y_max, rec_length_maximum, max_size)
            y_min = norm_and_resize(y_min, rec_length_maximum, max_size)

            gt_contour = gt_contour.astype(np.float32)
            gt_contour[:, 0] /= rec_length_maximum
            gt_contour[:, 1] /= rec_length_maximum

            gt_contour[:, 0] *= max_size
            gt_contour[:, 1] *= max_size

            two_x_max = two_x_max.astype(np.int32)
            two_x_min = two_x_min.astype(np.int32)

            two_y_max = two_y_max.astype(np.int32)
            two_y_min = two_y_min.astype(np.int32)

            x_max = x_max.astype(np.int32)
            x_min = x_min.astype(np.int32)

            y_max = y_max.astype(np.int32)
            y_min = y_min.astype(np.int32)

            gt_contour = gt_contour.astype(np.int32)

            if two_y_max < 0 or two_x_max < 0:
                crop_img.append(zero_mask_stack)
                continue

            box_img = np.zeros([two_y_max - two_y_min, two_x_max - two_x_min], dtype=np.uint8)
            sep_img = np.zeros([two_y_max - two_y_min, two_x_max - two_x_min], dtype=np.uint8)

            cv2.fillPoly(box_img, [gt_contour], 1)

            sep_img = cv2.drawContours(sep_img, [gt_contour], 0, (1, 1, 1), 1)

            if y_min + 1 < y_max and x_min + 1 < x_max:
                box_img = box_img[y_min:y_max, x_min:x_max]
                sep_img = sep_img[y_min:y_max, x_min:x_max]
                box_img = cv2.resize(box_img, (crop_width, crop_height))
                sep_img = cv2.resize(sep_img, (crop_width, crop_height))

                box_img = rotate_np(box_img, final_angle)
                sep_img = rotate_np(sep_img, final_angle)
                crop_img.append(np.stack([box_img, sep_img, one_mask], -1))
            else:
                crop_img.append(zero_mask_stack)

    return np.array(crop_img, dtype=np.float32)


def py_func_first_postprocessing(boxes, classifier):
    boxes = np.reshape(boxes, [-1, 4, 2])
    if boxes.shape[0] != 0:
        quad_boxes_reshape = np.reshape(boxes, [-1, 8])
        classifier_reshape = np.reshape(classifier, [-1, 1])
        nms_after_box, nms_after_score, num_after_class = nms_per_class(quad_boxes_reshape, classifier_reshape, 2400)
        boxes = np.reshape(nms_after_box[:, :8], [-1, 4, 2])
        classifier = np.reshape(nms_after_score[:, -1], [-1, 1])

    return boxes, classifier


def py_func_postprocessing(boxes, classifier, np_roi_idx, return_class=False):
    roi_boxes = boxes[np_roi_idx[:, 0], np_roi_idx[:, 1], :]
    roi_classifier = classifier[np_roi_idx[:, 0], np_roi_idx[:, 1], :]
    roi_boxes = np.reshape(roi_boxes, [-1, 4, 2])
    if roi_boxes.shape[0] != 0:
        roi_boxes_reshape = np.reshape(roi_boxes, [-1, 8])
        classifier_reshape = np.reshape(roi_classifier, [-1, FLAGS.num_class + 1])
        nms_after_box, nms_after_score, num_after_class = nms_per_class(roi_boxes_reshape, classifier_reshape,
                                                                        2400, 0.2, FLAGS.num_class, FLAGS.thres, True)
        roi_boxes = np.reshape(nms_after_box[:, :8], [-1, 4, 2])
        if return_class:
            roi_classifier = np.reshape(num_after_class[:, -1], [-1, 1])
            roi_classifier = roi_classifier.astype(np.int32)
        else:
            roi_classifier = np.reshape(nms_after_score[:, -1], [-1, 1])

    return roi_boxes, roi_classifier


def nms_per_class(boxes, score, k, iou=0.2, num_class=1, thres=0.3, refine=False):
    np_after_boxes = np.zeros([k, 8], dtype=np.float32)
    np_after_score = np.zeros([k, 1], dtype=np.float32)
    np_after_class = np.zeros([k, 1], dtype=np.int32)

    cum_max_nms_length = 0

    for class_idx in range(num_class):
        if refine:
            np_nms_input_thres = np.concatenate([boxes, np.round(np.expand_dims(score[:, class_idx + 1], axis=-1), 2)],
                                                -1)
        else:
            np_nms_input_thres = np.concatenate([boxes, np.round(np.expand_dims(score[:, class_idx], axis=-1), 2)], -1)

        np_nms_thres_idx = np.where(np_nms_input_thres[:, -1] > thres)
        np_nms_input_thres = np_nms_input_thres[np_nms_thres_idx]
        idx = np.argsort(np_nms_input_thres[:, -1])[::-1]
        cut_off = min(k * 5, np_nms_input_thres.shape[0])
        np_nms_input_thres = np_nms_input_thres[idx]

        np_nms_input_thres = np_nms_input_thres[:cut_off]
        np_after_idx = rbox_gpu_nms(np_nms_input_thres,
                                    iou)
        np_after_nms = np_nms_input_thres[np_after_idx.astype(np.int32)]
        max_nms_length = np_after_nms.shape[0]
        if max_nms_length == 0:
            continue
        np_after_nms_score = np_after_nms[:, -1]

        np_after_boxes[cum_max_nms_length:(max_nms_length + cum_max_nms_length)] = np_after_nms[:max_nms_length, :8]
        np_after_score[cum_max_nms_length:(max_nms_length + cum_max_nms_length), 0] = np_after_nms_score[:max_nms_length]
        np_after_class[cum_max_nms_length:(max_nms_length + cum_max_nms_length), 0] = class_idx+1

        cum_max_nms_length += max_nms_length
    cum_max_nms_length = min(cum_max_nms_length, k)
    np_after_boxes = np_after_boxes[:cum_max_nms_length]
    np_after_score = np_after_score[:cum_max_nms_length]
    np_after_class = np_after_class[:cum_max_nms_length]
    return np_after_boxes, np_after_score, np_after_class
