import numpy as np
import tensorflow as tf
import model_refine as model
from tensorflow.contrib import slim
from rpn_util.rpn_params import get_param
from intermediate_processing import whole_rpn_loss, py_func_postprocessing
from metric import F1_Metric

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_integer('input_size', 1024, '')
tf.app.flags.DEFINE_integer('batch_size', 7, '')
tf.app.flags.DEFINE_integer('num_readers', 16, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_epochs', 50020, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/rb/', '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')
tf.app.flags.DEFINE_string('test_data_path', 'ch8_test_debug_images',
                           'training dataset to use')
tf.app.flags.DEFINE_string('test_gt_path', 'ch8_test_debug_gt',
                           'training gt to use')
tf.app.flags.DEFINE_string('warmup_path', None,
                           'warmup to use')
tf.app.flags.DEFINE_float('thres', 0.4, '')

FLAGS = tf.app.flags.FLAGS

import data_loader


def debug_loss(name_1, loss_1, name_2, loss_2, name_3, loss_3, name_4, loss_4 ):
    name_1 = name_1.decode("utf-8")
    name_2 = name_2.decode("utf-8")
    name_3 = name_3.decode("utf-8")
    name_4 = name_4.decode("utf-8")
    print("{} : {}, {} : {}, {} : {}, {} : {}".format(name_1, loss_1, name_2, loss_2, name_3, loss_3, name_4, loss_4))
    return 0


def loss_fn(net, net_output, images, geo_maps, training_masks, quad_boxes, contours, label, reuse_variables=None):
    reg = net_output[0]
    rpn_1 = net_output[1]
    rpn_2 = net_output[2]
    rpn_3 = net_output[3]
    rpn_4 = net_output[4]
    rpn_5 = net_output[5]
    text_reg = net_output[6]

    with tf.variable_scope("reg_1"):
        reg_loss = net.add_loss(geo_maps, reg[0], training_masks, 32, 256 * 256, 2000 * 2000)
    with tf.variable_scope("reg_2"):
        reg_loss += net.add_loss(geo_maps, reg[1], training_masks, 16, 128 * 128, 512 * 512)
    with tf.variable_scope("reg_3"):
        reg_loss += net.add_loss(geo_maps, reg[2], training_masks, 8, 64 * 64, 256 * 256)
    with tf.variable_scope("reg_4"):
        reg_loss += net.add_loss(geo_maps, reg[3], training_masks, 4, 32 * 32, 128 * 128)
    with tf.variable_scope("reg_5"):
        reg_loss += net.add_loss(geo_maps, reg[4], training_masks, 2, 1, 64 * 64)

    concat_anchor_box = tf.concat([rpn_1[1][0], rpn_2[1][0], rpn_3[1][0], rpn_4[1][0], rpn_5[1][0], ],
                                  axis=1)
    concat_score_box = tf.concat([rpn_1[1][1], rpn_2[1][1], rpn_3[1][1], rpn_4[1][1], rpn_5[1][1], ],
                                 axis=1)
    concat_anchor_offset = tf.concat([rpn_1[1][2], rpn_2[1][2], rpn_3[1][2], rpn_4[1][2], rpn_5[1][2], ],
                                     axis=1)

    model_loss = whole_rpn_loss(concat_anchor_box, concat_score_box, concat_anchor_offset, quad_boxes)

    refine_loss = net.textregressor_fpn.loss_layer(text_reg[0], quad_boxes, text_reg[1], text_reg[2], text_reg[3],
                                                   text_reg[4], label)
    refine_loss_2 = net.textregressor_fpn.loss_layer(text_reg[5], quad_boxes, text_reg[6], text_reg[7], text_reg[8],
                                                   text_reg[9], label)


    debug = tf.py_func(debug_loss, ["reg_loss", reg_loss, "rpn_loss", model_loss,
                                    "refine_loss", refine_loss, "refine_loss_2", refine_loss_2], tf.int64)
    debug = tf.cast(debug, tf.float32)

    model_loss = tf.where(tf.math.is_nan(model_loss), tf.zeros_like(model_loss), model_loss)
    refine_loss = tf.where(tf.math.is_nan(refine_loss), tf.zeros_like(refine_loss), refine_loss)
    refine_loss_2 = tf.where(tf.math.is_nan(refine_loss_2), tf.zeros_like(refine_loss_2), refine_loss_2)
    reg_loss = tf.where(tf.math.is_nan(reg_loss), tf.zeros_like(reg_loss), reg_loss)

    model_loss += reg_loss
    model_loss += refine_loss
    model_loss += refine_loss_2
    model_loss += debug

    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # add summary
    if reuse_variables is None:
        tf.summary.image('input', images)
        # tf.summary.image('score_map_pred', f_score[:,:,:,0:1])
        # tf.summary.image('sep_map_pred', f_score[:, :, :, 1:2])
        tf.summary.image('geo_map_4_pred_4', reg[3][:, :, :, 5:6])
        tf.summary.image('geo_map_4_pred_5', reg[4][:, :, :, 5:6])

        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('reg_loss', reg_loss)

    return total_loss


def train_input_fn(img_list, txt_list):
    dataset = tf.data.Dataset.from_tensor_slices((img_list, txt_list))
    dataset = dataset.repeat(FLAGS.max_epochs*FLAGS.batch_size)
    dataset = dataset.map(
        lambda filepath, txtpath: tf.py_func(data_loader.get_tf_data, [filepath, txtpath, -1],
                                             [tf.float32, tf.float32, tf.float32,
                                              tf.float32, tf.float32, tf.int32, tf.int64]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #  dataset = dataset.apply(tf.contrib.data.ignore_errors())
    dataset = dataset.shuffle(100)
    #  dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.padded_batch(FLAGS.batch_size,
                                   (
                                       [None, None, 3], [None, None, 6], [None, None, 1],
                                       [FLAGS.max_text, 8],
                                       [FLAGS.max_text, FLAGS.max_polygon, 2],
                                       [FLAGS.max_text], [3]))

    dataset = dataset.map(
        map_func=lambda input_images, input_geo_maps,
                        input_training_masks, input_quad_boxes,
                        input_contours, input_label, input_shape:
        (
            {'input_images': input_images, 'input_geo_maps': input_geo_maps,
             'input_training_masks': input_training_masks, 'input_quad_boxes': input_quad_boxes,
             'input_contours': input_contours, 'input_label': input_label, 'input_shape':input_shape}, input_quad_boxes))

    return dataset


def test_input_fn(img_list, txt_list):
    dataset = tf.data.Dataset.from_tensor_slices((img_list, txt_list))
    dataset = dataset.map(
        lambda filepath, txtpath: tf.py_func(data_loader.get_test_tf_data, [filepath, txtpath, FLAGS.input_size],
                                             [tf.float32, tf.float32, tf.float32,
                                              tf.float32, tf.float32, tf.int32, tf.int64]))
    dataset = dataset.batch(1)

    dataset = dataset.map(
        map_func=lambda input_images, input_geo_maps, input_training_masks, input_quad_boxes, input_contours,
                        input_label, input_shape:
        (
            {'input_images': input_images, 'input_geo_maps': input_geo_maps,
             'input_training_masks': input_training_masks, 'input_quad_boxes': input_quad_boxes,
             'input_contours': input_contours, 'input_label': input_label, 'input_shape': input_shape},
            input_quad_boxes))

    return dataset


def trainmode_to_evalmode(rpn):
    box2 = rpn[0]
    roi_idx2 = rpn[1]
    refined_boxes = rpn[2]
    classifier_ch2 = rpn[3]
    refined_boxes2 = tf.scatter_nd(roi_idx2, box2, tf.cast(tf.shape(refined_boxes), tf.int64))
    refined_boxes2_shape = tf.shape(refined_boxes2)
    refined_score2 = tf.scatter_nd(roi_idx2, tf.nn.softmax(classifier_ch2, axis=-1),
                                   tf.cast([refined_boxes2_shape[0], refined_boxes2_shape[1],
                                            FLAGS.num_class + 1], tf.int64))

    return refined_boxes2, refined_score2





def custom_model(features, mode, params, labels):

    input_images = features["input_images"]
    input_geo_maps = features["input_geo_maps"]
    input_training_masks = features["input_training_masks"]
    input_quad_boxes = features["input_quad_boxes"]
    input_contours = features["input_contours"]
    input_label = features["input_label"]
    input_shape = features["input_shape"]

    if mode == tf.estimator.ModeKeys.TRAIN:
        networt_mode = "train"
        input_images = tf.reshape(input_images,
                                  [FLAGS.batch_size, tf.reduce_max(input_shape[:, 0]), tf.reduce_max(input_shape[:, 1]),
                                   3])
        input_geo_maps = tf.reshape(input_geo_maps,
                                    [FLAGS.batch_size, tf.reduce_max(input_shape[:, 0]),
                                     tf.reduce_max(input_shape[:, 1]), 6])
        input_training_masks = tf.reshape(input_training_masks,
                                          [FLAGS.batch_size, tf.reduce_max(input_shape[:, 0]),
                                           tf.reduce_max(input_shape[:, 1]), 1])
        input_quad_boxes = tf.reshape(input_quad_boxes, [FLAGS.batch_size, FLAGS.max_text, 8])
        input_contours = tf.reshape(input_contours, [FLAGS.batch_size, FLAGS.max_text, FLAGS.max_polygon, 2])
        input_label = tf.reshape(input_label, [FLAGS.batch_size, FLAGS.max_text])
    elif mode == tf.estimator.ModeKeys.EVAL:
        networt_mode = "eval"
        input_images = tf.reshape(input_images,
                                  [1, tf.reduce_max(input_shape[:, 0]), tf.reduce_max(input_shape[:, 1]),
                                   3])
        input_geo_maps = tf.reshape(input_geo_maps,
                                    [1, tf.reduce_max(input_shape[:, 0]),
                                     tf.reduce_max(input_shape[:, 1]),
                                     6])
        input_training_masks = tf.reshape(input_training_masks,
                                          [1, tf.reduce_max(input_shape[:, 0]),
                                           tf.reduce_max(input_shape[:, 1]),
                                           1])
        input_quad_boxes = tf.reshape(input_quad_boxes, [1, FLAGS.max_text, 8])
        input_contours = tf.reshape(input_contours, [1, FLAGS.max_text, FLAGS.max_polygon, 2])
        input_label = tf.reshape(input_label, [1, FLAGS.max_text])

    else:
        networt_mode = "predict"
        input_images = tf.reshape(input_images,
                                  [FLAGS.batch_size, tf.reduce_max(input_shape[:, 0]), tf.reduce_max(input_shape[:, 1]),
                                   3])
        input_geo_maps = tf.reshape(input_geo_maps,
                                    [FLAGS.batch_size, tf.reduce_max(input_shape[:, 0]),
                                     tf.reduce_max(input_shape[:, 1]),
                                     6])
        input_training_masks = tf.reshape(input_training_masks,
                                          [FLAGS.batch_size, tf.reduce_max(input_shape[:, 0]),
                                           tf.reduce_max(input_shape[:, 1]),
                                           1])
        input_quad_boxes = tf.reshape(input_quad_boxes, [FLAGS.batch_size, FLAGS.max_text, 8])
        input_contours = tf.reshape(input_contours, [FLAGS.batch_size, FLAGS.max_text, FLAGS.max_polygon, 2])
        input_label = tf.reshape(input_label, [FLAGS.batch_size, FLAGS.max_text])

    with tf.variable_scope(tf.get_variable_scope(), reuse=None):
        net = model.TextmapNetwork(input_images, "resnet50", get_param(), mode=networt_mode, gt=input_quad_boxes)
        net_output = net.build_network()

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'segmap': net_output[0][0],
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = loss_fn(net, net_output,
                   input_images, input_geo_maps,
                   input_training_masks, input_quad_boxes,
                   input_contours, input_label)

    if mode == tf.estimator.ModeKeys.EVAL:

        text_reg = net_output[6]

        text_reg_boxes, text_reg_scores = trainmode_to_evalmode(text_reg)
        boxes, label = tf.py_func(py_func_postprocessing, [text_reg_boxes, text_reg_scores, text_reg[1], True],
                                  [tf.float32, tf.int32], name='py_func_postprocessing')


        predictions = {"rec_boxes": boxes,
                       "classifier": label}
        ground_truth = {"quad_boxes": input_quad_boxes,
                        "label": input_label}

        recall_metric = F1_Metric(class_num=FLAGS.num_class, mode="recall")
        recall_metric.update_state(y=ground_truth, pred=predictions)

        precision_metric = F1_Metric(class_num=FLAGS.num_class, mode="precision")
        precision_metric.update_state(y=ground_truth, pred=predictions)

        metrics = {"precision": precision_metric, "recall": recall_metric}

        return tf.estimator.EstimatorSpec(
            mode, eval_metric_ops=metrics, loss=loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=50000, decay_rate=0.94,
                                                   staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        all_variable = tf.contrib.framework.get_variables_to_restore()
        summary_variable_list = ["refine/mini_feature_extraction/batch_normalization/moving_mean",
                                 "refine/mini_feature_extraction/batch_normalization/moving_variance",
                                 "refine/mini_feature_extraction/batch_normalization_1/moving_mean",
                                 "refine/mini_feature_extraction/batch_normalization_1/moving_variance",
                                 "refine/mini_feature_extraction/batch_normalization_2/moving_mean",
                                 "refine/mini_feature_extraction/batch_normalization_2/moving_variance",
                                 "refine/mini_feature_extraction/batch_normalization_3/moving_mean",
                                 "refine/mini_feature_extraction/batch_normalization_3/moving_variance",
                                 "refine/mini_feature_extraction/batch_normalization_4/moving_mean",
                                 "refine/mini_feature_extraction/batch_normalization_4/moving_variance",
                                 ]
        for each_variable in all_variable:
            each_variable_name = each_variable.name.split(":")[0]
            if each_variable_name in summary_variable_list:
                print(each_variable_name)
                tf.summary.scalar(each_variable_name, tf.reduce_mean(each_variable))

        opt = tf.train.AdamOptimizer(learning_rate)
        opt = tf.contrib.estimator.clip_gradients_by_norm(opt, clip_norm=1)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(loss, global_step=global_step)

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


if __name__ == "__main__":
    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MkDir(FLAGS.checkpoint_path)
    else:
        if not FLAGS.restore:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
            tf.gfile.MkDir(FLAGS.checkpoint_path)

    images_list, txt_gt_list = data_loader.get_images()
    test_images_list, test_txt_list = data_loader.get_images(FLAGS.test_data_path, FLAGS.test_gt_path)

    hooks = [tf.train.ProfilerHook(output_dir=FLAGS.checkpoint_path, save_secs=60, show_memory=False)]

    distribution = tf.contrib.distribute.MirroredStrategy()

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.allow_soft_placement = True
    session_config.log_device_placement = True

    config = tf.estimator.RunConfig(save_summary_steps=FLAGS.save_summary_steps,
                                    keep_checkpoint_max=3,
                                    log_step_count_steps=10,
                                    train_distribute=distribution,
                                    session_config=session_config,)

    if FLAGS.warmup_path is None:
        ws = None
    else:
        print("warmup!!")
        ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=FLAGS.warmup_path)

    rb = tf.estimator.Estimator(
        model_fn=custom_model,
        model_dir=FLAGS.checkpoint_path,
        config=config,
        warm_start_from=ws
    )

    #  classifier.evaluate(input_fn=lambda: test_input_fn(test_images_list, test_txt_list))  # , hooks=hooks)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(images_list, txt_gt_list))
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: test_input_fn(test_images_list, test_txt_list), steps=None,
                                      throttle_secs=60 * 60)

    tf.estimator.train_and_evaluate(rb, train_spec=train_spec, eval_spec=eval_spec)

