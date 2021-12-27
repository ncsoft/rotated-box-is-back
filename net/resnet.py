import tensorflow as tf
import numpy as np


def bottle_layer(x, conv2_list, seq, is_training):
    info = conv2_list[-1]
    info2 = conv2_list[0]
    if seq == 0 :
        stride = (2, 2)
    else:
        stride = (1, 1)
    with tf.variable_scope("sub"+str(seq)):
        if seq == 0:
            stride = (2, 2)
            add_layer = tf.layers.conv2d(inputs=x, filters=info["filter"], kernel_size=[1, 1],
                                         strides=stride, padding=info2["padding"])
            add_layer = tf.layers.batch_normalization(add_layer, training=is_training, name="bn_"+str(seq))
        else:
            add_layer = x
        for i, info in enumerate(conv2_list):
            if i == 0:
                x = tf.layers.conv2d(inputs=x, filters=info["filter"], kernel_size=info["kernel_size"],
                                     strides=stride, padding=info["padding"], name="conv2d_" + str(i))
            else:
                x = tf.layers.conv2d(inputs=x,filters=info["filter"], kernel_size=info["kernel_size"],
                                     strides=info["stride"], padding=info["padding"], name="conv2d_"+str(i))
            x = tf.layers.batch_normalization(x, training=is_training, name="bn_"+str(seq)+"_"+str(i))
            if i != len(conv2_list) - 1:
                x = tf.nn.relu(x)
    return tf.nn.relu(x + add_layer)


def get_info(ch, kernel_size, stirde=(1, 1), padding="same"):
    info = {}
    info["filter"] = ch
    info["kernel_size"] = kernel_size
    info["stride"] = stirde
    info["padding"] = padding
    return info


def make_info_block_list(mode):
    if mode == 18:
        num_list = [2, 2, 2, 2]
        num_ch = [[64, 64], [128, 128], [256, 256], [512, 512]]
        num_kernel = [[3, 3], [3, 3], [3, 3], [3, 3]]

    elif mode == 34:
        num_list = [3, 4, 6, 3]
        num_ch = [[64, 64], [128, 128], [256, 256], [512, 512]]
        num_kernel = [[3, 3], [3, 3], [3, 3], [3, 3]]

    elif mode == 50:
        num_list = [3, 4, 6, 3]
        num_ch = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]
        num_kernel = [[1, 3, 1], [1, 3, 1], [1, 3, 1], [1,3, 1]]

    elif mode == 101:
        num_list = [3, 4, 23, 3]
        num_ch = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]
        num_kernel = [[1, 3, 1], [1, 3, 1], [1, 3, 1], [1, 3, 1]]
    elif mode == 152:
        num_list = [3, 8, 36, 3]
        num_ch = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]
        num_kernel = [[1, 3, 1], [1, 3, 1], [1, 3, 1], [1, 3, 1]]
    else:
        assert 1, "mode is not correct"

    whole_block = []
    for i, num in enumerate(num_list):
        blocks = []
        for j, ch in enumerate(num_ch[i]):
            blocks.append(get_info(ch, [num_kernel[i][j], num_kernel[i][j]] ))
        whole_block.append(blocks)
    return whole_block, num_list


class Resnet(object):
    def __init__(self):
        self.image = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.label = tf.placeholder(tf.float32, shape=[None, None])

    def make_block(self, input, mode, is_training):
        whole_block, num_list = make_info_block_list(mode)
        endpoint = {}
        for i, blocks in enumerate(whole_block):
            with tf.variable_scope("block"+str(i)):
                for loop in range(num_list[i]):
                    output = bottle_layer(input, blocks, loop, is_training)
                    input = output
            endpoint["pool" + str(i + 2)] = output
        return input, endpoint

    def make_tail(self, input):
        input = tf.reduce_mean(input, axis=[1, 2])
        return input

    def make_classifier(self,input, num_class):
        return tf.layers.dense(input, num_class, name='fc')

    def build_graph(self, input, mode, num_class, is_training):
        input,_ = self.make_block(input, mode, is_training)
        input = self.make_tail(input)
        output = self.make_classifier(input, num_class)
        return output

    def add_loss(self, output, labels):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=output)
        return loss


