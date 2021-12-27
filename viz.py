import cv2
import time
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from rpn_util.rpn_params import get_param
from intermediate_processing import py_func_postprocessing, py_func_first_postprocessing
from debug_utils import makepoly_with_score_color


tf.app.flags.DEFINE_string('test_data_path', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_string('gpu_list', '0', 'select running gpu index')
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/east_icdar2015_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_string('output_dir', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')
tf.app.flags.DEFINE_integer('min_size', 1100, '')
tf.app.flags.DEFINE_integer('max_size', 2000, '')
tf.app.flags.DEFINE_integer('batch_size', 1, '')
tf.app.flags.DEFINE_float('thres', 0.4, '')

import model_refine as model
FLAGS = tf.app.flags.FLAGS


def get_images():
    """
    find image files in test data path
    :return: list of files found
    """
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG', 'gif']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    files.sort()
    return files


def resize_image_min_max(im, min_side_len=512, max_side_len=1024):
    """
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param min_side_len: set image size to minimum height or width
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    """
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    ratio = float(min_side_len) / resize_h if resize_h < resize_w else float(min_side_len) / resize_w

    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32) * 32
    if resize_h == 0 or resize_w == 0:
        min_length = min(h, w)
        resize_w = int(w/float(min_length))*32
        resize_h = int(h/float(min_length))*32

    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def trainmode_to_evalmode(rpn, offset=5):
    box2 = rpn[0 + offset]
    roi_idx2 = rpn[1 + offset]
    refined_boxes = rpn[2 + offset]
    classifier_ch2 = rpn[3 + offset]

    refined_boxes2 = tf.scatter_nd(roi_idx2, box2, tf.cast(tf.shape(refined_boxes), tf.int64))
    refined_boxes2_shape = tf.shape(refined_boxes2)
    refined_score2 = tf.scatter_nd(roi_idx2, tf.nn.softmax(classifier_ch2, axis=-1),
                                   tf.cast([refined_boxes2_shape[0], refined_boxes2_shape[1],
                                            FLAGS.num_class + 1], tf.int64))

    return refined_boxes2, refined_score2


def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.train.get_global_step()
        net = model.TextmapNetwork(input_images, "resnet50", get_param(), mode="predict")
        net_output = net.build_network()

        text_reg = net_output[-1]

        text_reg_boxes, text_reg_scores = trainmode_to_evalmode(text_reg)
        first_text_reg_boxes, first_text_reg_scores = trainmode_to_evalmode(text_reg, 0)

        rrpn_score = net_output[0][-1]
        rrpn_box = net_output[0][-2]

        reg_score = net_output[0][-3]
        reg_box = net_output[0][-4]

        tf_boxes, tf_score = tf.py_func(py_func_postprocessing,
                                        [text_reg_boxes, text_reg_scores, text_reg[1]],
                                        [tf.float32, tf.float32], name='py_func_postprocessing')
        tf_first_boxes, tf_first_score = tf.py_func(py_func_postprocessing,
                                                    [first_text_reg_boxes, first_text_reg_scores, text_reg[1]],
                                                    [tf.float32, tf.float32],
                                                    name='py_func_first_postprocessing')

        tf_rrpn_boxes, tf_rrpn_score = tf.py_func(py_func_first_postprocessing,
                                                  [rrpn_box, rrpn_score],
                                                  [tf.float32, tf.float32], name='py_func_first_stage')
        tf_reg_boxes, tf_reg_score = tf.py_func(py_func_first_postprocessing,
                                                [reg_box, reg_score],
                                                [tf.float32, tf.float32], name='py_func_first_stage')

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            for i, im_fn in enumerate(im_fn_list):
                print("process : {} {}".format(i, im_fn))
                im = Image.open(im_fn).convert('RGB')
                im = np.array(im)[:, :, ::-1]

                im_resized, (ratio_h, ratio_w) = resize_image_min_max(im, FLAGS.min_size, FLAGS.max_size)
                im_resize_shape = im_resized.shape

                np_boxes, np_score,\
                np_first_boxes, np_first_score,\
                np_rrpn_boxes, np_rrpn_score,\
                np_reg_boxes, np_reg_score = sess.run([tf_boxes, tf_score,
                                                       tf_first_boxes, tf_first_score,
                                                       tf_rrpn_boxes, tf_rrpn_score,
                                                       tf_reg_boxes, tf_reg_score],
                                                      feed_dict={input_images: [im_resized]})

                rec_boxes = np.reshape(np_boxes.copy(), [-1, 4, 2])
                rec_score = np.reshape(np_score.copy(), [-1, 1])

                rec_boxes[:, :, 0] = rec_boxes[:, :, 0] / ratio_w + 1
                rec_boxes[:, :, 1] = rec_boxes[:, :, 1] / ratio_h + 1
                rec_boxes = rec_boxes.astype(np.int32)
                res_file = os.path.join(
                    FLAGS.output_dir,
                    '{}.txt'.format(
                        os.path.basename(im_fn).split('.')[0]))

                img_h, img_w, ch = im.shape

                with open(res_file, 'w') as f:
                    for box, score in zip(rec_boxes, rec_score):
                        if score[0] > 0:
                            f.write('{0},{1},{2},{3},{4},{5},{6},{7},{8:0.2f}\r\n'.format(
                                box[0, 0], box[0, 1],
                                box[1, 0], box[1, 1],
                                box[2, 0], box[2, 1],
                                box[3, 0], box[3, 1],
                                score[0]
                             ))

                if not FLAGS.no_write_images:
                    im_resized_first_image = im_resized.copy()
                    im_resized_first_image = makepoly_with_score_color(1, np.reshape(np_rrpn_boxes, [1, -1, 8]),
                                                                       np.reshape(np_rrpn_score, [1, -1]),
                                                                       im_resize_shape[0], im_resize_shape[1],
                                                                       im_resized_first_image[np.newaxis, :, :, :],
                                                                       (255, 255, 0), FLAGS.thres)
                    im_resized_first_image = makepoly_with_score_color(1, np.reshape(np_reg_boxes, [1, -1, 8]),
                                                                       np.reshape(np_reg_score, [1, -1]),
                                                                       im_resize_shape[0], im_resize_shape[1],
                                                                       im_resized_first_image,
                                                                       (0, 255, 255), FLAGS.thres)

                    resized_result = makepoly_with_score_color(1, np.reshape(np_first_boxes, [1, -1, 8]),
                                                               np.reshape(np_first_score, [1, -1]),
                                                               im_resize_shape[0], im_resize_shape[1],
                                                               im_resized[np.newaxis, :, :, :],
                                                               (0, 0, 255), FLAGS.thres)
                    resized_result = makepoly_with_score_color(1, np.reshape(np_boxes, [1, -1, 8]),
                                                               np.reshape(np_score, [1, -1]),
                                                               im_resize_shape[0], im_resize_shape[1],
                                                               resized_result,
                                                               (0, 255, 0), FLAGS.thres)

                    im_result = makepoly_with_score_color(1, np.reshape(rec_boxes, [1, -1, 8]),
                                                          np.reshape(np_score, [1, -1]), img_h,
                                                          img_w, im[np.newaxis, :, :, :],
                                                          (0, 255, 0), FLAGS.thres)

                    concat_image = np.concatenate([im_resized_first_image[0, :, :, ],
                                                   resized_result[0, :, :, :]], axis=0)
                    result_file = os.path.join(
                        FLAGS.output_dir,
                        '{}_resized_result.jpg'.format(
                            os.path.basename(im_fn).split('.')[0]))
                    cv2.imwrite(result_file, concat_image)

                    score_file = os.path.join(
                        FLAGS.output_dir,
                        '{}_result.jpg'.format(
                            os.path.basename(im_fn).split('.')[0]))
                    cv2.imwrite(score_file, im_result[0, :, :, ])


if __name__ == '__main__':
    tf.app.run()
