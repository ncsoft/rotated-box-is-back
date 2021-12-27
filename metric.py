import tensorflow as tf
import numpy as np
from utils import cal_iou_np

FLAGS = tf.app.flags.FLAGS





def update_gt_class(y, idx, test_image_num, class_num):
    gt_matrix = np.zeros(shape=[test_image_num, class_num], dtype=np.float32)
    for label in y:
        gt_matrix[int(idx), int(label)-1] += 1
    return gt_matrix


def update_tp_fp(gt_label, gt_quadboxes, rec_boxes, classifier, idx, test_image_num, class_num,
                 iou_threshold=0.5):
    tp_matrix = np.zeros(shape=[test_image_num, class_num], dtype=np.float32)
    fp_matrix = np.zeros(shape=[test_image_num, class_num], dtype=np.float32)
    idx = int(idx)
    gt_quadboxes = gt_quadboxes[:, 0, :]
    rec_boxes = rec_boxes.reshape([-1, 8])
    classifier = classifier.reshape([-1, 1])

    rec_boxes_removed = []
    classifier_removed = []

    for each_rec_boxes, each_classifier in zip(rec_boxes, classifier):
        if np.sum(each_rec_boxes) == 0:
            continue
        else:
            rec_boxes_removed.append(each_rec_boxes)
            classifier_removed.append(each_classifier)

    rec_boxes = np.reshape(np.array(rec_boxes_removed), [-1, 8])
    classifier = np.reshape(np.array(classifier_removed), [-1, 1])

    gt_num = gt_label.shape[0]
    positive_num = 0
    true_positive_num = 0
    matching_checking = {}
    overlaps = cal_iou_np(gt_quadboxes, rec_boxes)
    print("idx : {}".format(idx))
    if len(overlaps) != 0:
        # (GT dim)
        # pred_idx_base_on_gt = np.argmax(overlaps, axis=-1)
        # pred_max_iou_base_on_gt = np.amax(overlaps, axis=-1)
        gt_idx_base_on_pred = np.argmax(overlaps, axis=0)
        gt_max_iou_base_on_pred = np.amax(overlaps, axis=0)

        for each_gt_idx, iou, each_pred_label in zip(gt_idx_base_on_pred, gt_max_iou_base_on_pred, classifier):
            positive_num += 1

            if iou > iou_threshold:  # and gt_label[each_gt_idx] == each_pred_label:
                if matching_checking.get(each_gt_idx, None) is None:
                    true_positive_num += 1
                    matching_checking[each_gt_idx] = 1

        false_positive = positive_num - true_positive_num

        print("positive_num : {}".format(positive_num))
        print("false_positive : {}".format(false_positive))
        print("true_positive_num : {}".format(true_positive_num))

        print("precision : {}".format(true_positive_num / float(positive_num + 0.00001)))
        print("recall : {}".format(true_positive_num / float(gt_num + 0.00001)))
        tp_matrix[idx, 0] = true_positive_num
        fp_matrix[idx, 0] = false_positive

    return tp_matrix, fp_matrix


class F1_Metric(tf.keras.metrics.Metric):
    def __init__(self, class_num, test_image_num=10000, mode="recall", name='mAPmetric', **kwargs):
        super(F1_Metric, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", shape=(test_image_num, class_num), initializer='zero')
        self.false_positives = self.add_weight(name="fp", shape=(test_image_num, class_num), initializer='zero')
        self.gt_counter_per_class = self.add_weight(name='gt_per_class', shape=(test_image_num, class_num),
                                                    initializer='zero')
        self.test_image_num = test_image_num
        self.num_class = class_num
        self.mode = mode

        self.cnt = self.add_weight(name="cnt", initializer='zero')

    def update_state(self, y, pred):
        """
        batch size 1 supported
        """
        gt_label = y["label"][0]
        gt_quadboxes = y["quad_boxes"][0]

        rec_boxes = pred["rec_boxes"]
        classifier = pred["classifier"]

        gt_label_idx = tf.where(gt_label > 0)
        gt_label_gather = tf.gather(gt_label, gt_label_idx)
        gt_quadboxes_gather = tf.gather(gt_quadboxes, gt_label_idx)

        gt_matrix = tf.py_func(update_gt_class, [gt_label_gather, self.cnt, self.test_image_num, self.num_class],
                               tf.float32, name='update_gt_class')
        self.gt_counter_per_class.assign_add(gt_matrix)

        tp, fp = tf.py_func(update_tp_fp, [gt_label_gather, gt_quadboxes_gather, rec_boxes, classifier, self.cnt,
                            self.test_image_num, self.num_class], [tf.float32, tf.float32],
                            name='update_tp_fp')
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.cnt.assign_add(1)

    def result(self):
        whole_tp = tf.reduce_sum(self.true_positives, axis=[0, 1])
        whole_fp = tf.reduce_sum(self.false_positives, axis=[0, 1])
        whole_gt = tf.reduce_sum(self.gt_counter_per_class, axis=[0, 1])
        if self.mode == "recall":
            recall = whole_tp / whole_gt
            return recall
        elif self.mode == "precision":
            precision = whole_tp / (whole_tp + whole_fp)
            return precision
        else:
            recall = whole_tp / whole_gt
            precision = whole_tp / (whole_tp + whole_fp)
            f_score = 2/(1/recall + 1/precision)
            return f_score

    def reset_states(self):
        for i in range(self.num_class):
            self.true_positives[i].assign(0)
            self.false_positives[i].assign(0)
            self.gt_counter_per_class[i].assign(0)
