# coding:utf-8
import glob
import cv2
import time
import os
import numpy as np
import codecs
import tensorflow as tf
import math
import xml.etree.ElementTree as ET
import time

tf.app.flags.DEFINE_string('training_data_path', '/data/ocr/icdar2015/',
                           'training dataset to use')
tf.app.flags.DEFINE_string('training_gt_path', '/data/ocr/icdar2015/',
                           'training gt to use')
tf.app.flags.DEFINE_float('min_crop_side_ratio', 0.5,
                          'when doing random crop from input image, the'
                          'min length of min(H, W')
tf.app.flags.DEFINE_integer('min_text_size', 0,
                            'if the text size is smaller than this, we ignore it during training')
tf.app.flags.DEFINE_integer('rotate', 1, "rotation")
tf.app.flags.DEFINE_integer('flip', 1, "flip")
tf.app.flags.DEFINE_integer('blur', 1, "blur")
tf.app.flags.DEFINE_integer('max_text', 100, "max_text")
tf.app.flags.DEFINE_integer('max_polygon', 100, "max_polygon")
FLAGS = tf.app.flags.FLAGS
DIV = 1


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p_copy = p.copy()
    p_copy = sorted(p_copy, key=lambda x: x[0])
    p_copy = p_copy[:2]
    p_copy = sorted(p_copy, key=lambda x: x[1])
    min_point = p_copy[0]
    for i, point in enumerate(p):
        if point[0] == min_point[0] and point[1] == min_point[1]:
            min_axis = i
            # print("{} min_axis".format(min_axis))
            break

    min_axis_plus = (min_axis + 1) % 4
    min_axis_minus = (min_axis - 1) % 4
    max_axis = (min_axis + 2) % 4

    max_point = p[max_axis].copy() - min_point
    min_axis_minus_point = p[min_axis_minus].copy() - min_point

    if max_point[0] * min_axis_minus_point[1] - max_point[1] * min_axis_minus_point[0] < 0:
        p = p[[min_axis, min_axis_minus, (min_axis_minus - 1) % 4, (min_axis_minus - 2) % 4, ]]
    else:
        p = p[[min_axis, min_axis_plus, (min_axis_plus + 1) % 4, (min_axis_plus + 2) % 4, ]]
    return p


def recal_theta(p):
    vector = (p[1] - p[0]) + (p[2] - p[3])
    eps = 0.00001
    theta = np.arctan((vector[1]) / ((vector[0]) + eps))
    vector_2 = (p[2] - p[1]) + (p[3] - p[0])
    theta_2 = np.arctan((vector_2[1]) / ((vector_2[0]) + eps))

    if np.linalg.norm(vector) > np.linalg.norm(vector_2):
        if np.abs(theta) * (180 / np.pi) > 50:
            if np.sign(theta) >= 0:
                return p[[-1, 0, 1, 2]]
            else:
                # vertical
                return p[[1, 2, 3, 0]]
        else:
            # horizontal
            return p
    else:
        if np.abs(theta_2) * (180 / np.pi) > 50:
            # vertical
            return p
        else:
            # horizontal
            return p[[1, 2, 3, 0]]


def rotate_about_center(src, angle, scale=1.):
    """https://www.oschina.net/translate/opencv-rotation"""
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4), rot_mat


def rotate_image(im, text_poly):
    angle = np.random.rand() * 15
    if np.random.rand() > 0.5:
        angle *= -1
    im, m = rotate_about_center(im, angle)
    for i, box in enumerate(text_poly):
        for j, point in enumerate(box):
            r_point = np.dot(m, np.array([point[0], point[1], 1]))
            text_poly[i][j] = r_point

    return im, text_poly


def flip_hori_image(im, text_poly):
    im_shape = im.shape
    img_width = im_shape[1]

    for i, box in enumerate(text_poly):
        for j, point in enumerate(box):
            x = point[0]
            y = point[1]
            new_x = img_width - x
            text_poly[i][j] = np.array([new_x, y])
    im = cv2.flip(im, 1)
    return im, text_poly


def flip_ver_image(im, text_poly):
    im_shape = im.shape
    img_height = im_shape[0]

    for i, box in enumerate(text_poly):
        for j, point in enumerate(box):
            x = point[0]
            y = point[1]
            new_y = img_height - y
            text_poly[i][j] = np.array([x, new_y])

    im = cv2.flip(im, 0)
    return im, text_poly


def get_gt(p, data_path, gt_path):
    basename = os.path.basename(p)
    basedir = os.path.dirname(p)
    norm_training_data_path = os.path.normpath(data_path)
    norm_training_gt_path = os.path.normpath(gt_path)
    basedir = basedir.replace(norm_training_data_path, norm_training_gt_path)
    txt_path = os.path.join(basedir, basename.replace(basename.split(".")[-1], "txt"))
    xml_path = os.path.join(basedir, basename.replace(basename.split(".")[-1], "xml"))
    if os.path.exists(txt_path):
        return txt_path
    elif os.path.exists(xml_path):
        return xml_path
    else:
        return False


def get_images(data_path_base=FLAGS.training_data_path, gt_path_base=FLAGS.training_gt_path):
    files = []
    gt_paths = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(data_path_base):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    del_box = []
    whole_len = len(files)
    for i in range(whole_len):
        print("{} in {}".format(i, whole_len))
        gt_path = get_gt(files[i], data_path_base, gt_path_base)
        if gt_path is False:
            result = None
        else:
            result = load_annotation(gt_path)
        if result is None:
            print("i is deleted : " + str(i))
            del_box.append(i)
        else:
            gt_paths.append(gt_path)
    del_box.sort()
    del_box.reverse()
    for i in del_box:
        del files[i]
    return files, gt_paths


def load_annotation(p):
    """
    load annotation from the text file
    :param p: text file path for icdar format
    :return text_polys, text_tags:
    """
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return None, None
    ends = p.split(".")[-1]
    if ends == "txt":
        f = codecs.open(p, mode='r', encoding='utf-8-sig')
        while True:
            line = f.readline()
            if not line:
                break
            boxes = line.split(",")
            boxes = [box.rstrip().strip() for box in boxes]
            label = boxes[-1]
            boxes_length = len(boxes)

            if boxes_length >= 8:
                polygon_list = []
                polygon_length = 0
                for each_char in boxes:
                    if each_char[0] == '-':
                        decision_char = each_char[1:]
                    else:
                        decision_char = each_char

                    if decision_char.isdigit():
                        polygon_length += 1
                    else:
                        break

                assert polygon_length % 2 == 0, "gt point is not x,y pair"
                polygon_length = polygon_length // 2


                for i in range(polygon_length):
                    polygon_list.append([int(boxes[2 * i]), int(boxes[2 * i + 1])])
                text_polys.append(polygon_list)

            else:
                box = [boxes[0], boxes[1], boxes[2], boxes[3]]
                box = [int(bo) for bo in box]
                x1 = box[0]
                y1 = box[1]
                x3 = box[2]
                y3 = box[3]
                x2 = x3
                y2 = y1
                x4 = x1
                y4 = y3
                text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            label = label.strip()
            if label == '*':  # "#" or "###" in label:
                text_tags.append(0)
            else:
                text_tags.append(1)
    elif ends == "xml":
        tree = ET.parse(p)
        objs = tree.findall('object')
        num_objs = len(objs)
        if num_objs == 0:
            return None, None

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            if bbox.find('x1') is None:
                # Make pixel indexes 0-based
                x1 = float(bbox.find('xmin').text)
                y1 = float(bbox.find('ymin').text)
                x3 = float(bbox.find('xmax').text)
                y3 = float(bbox.find('ymax').text)

                x2 = x3
                y2 = y1

                x4 = x1
                y4 = y3
            else:
                x1 = float(bbox.find('x1').text)
                y1 = float(bbox.find('y1').text)
                x2 = float(bbox.find('x2').text)
                y2 = float(bbox.find('y2').text)
                x3 = float(bbox.find('x3').text)
                y3 = float(bbox.find('y3').text)
                x4 = float(bbox.find('x4').text)
                y4 = float(bbox.find('y4').text)
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            text_tags.append(1)
    else:
        return None, None
    return text_polys, text_tags


def crop_area(im, polys, tags, crop_background=False, max_tries=50):
    """
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    """
    h, w, _ = im.shape
    pad_h = h // 10
    pad_w = w // 10
    h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
    w_array = np.zeros((w + pad_w * 2), dtype=np.int32)
    selected_polys = []
    selected_tags = []

    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx + pad_w:maxx + pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny + pad_h:maxy + pad_h] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, polys, tags
    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w - 1)
        xmax = np.clip(xmax, 0, w - 1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h - 1)
        ymax = np.clip(ymax, 0, h - 1)
        if xmax - xmin < FLAGS.min_crop_side_ratio * w or ymax - ymin < FLAGS.min_crop_side_ratio * h:
            # area too small
            continue
        if len(polys) != 0:
            for poly, tag in zip(polys, tags):
                cnt = 0
                for point in poly:
                    if point[0] < xmin or point[0] > xmax or point[1] < ymin or point[1] > ymax:
                        cnt = 1
                        break

                if cnt == 0:
                    selected_polys.append(poly)
                    selected_tags.append(tag)

        if len(selected_polys) == 0:
            # no text in this area
            if crop_background:
                return im[ymin:ymax + 1, xmin:xmax + 1, :], selected_polys, selected_tags
            else:
                continue
        im = im[ymin:ymax + 1, xmin:xmax + 1, :]
        polys = selected_polys
        tags = selected_tags
        for poly in polys:
            for point in poly:
                point[0] -= xmin
                point[1] -= ymin

        return im, polys, tags

    return im, polys, tags


def caldistancepoint(a, b):
    return np.linalg.norm(a - b, axis=-1)


def callinetopoint(a, b, c):
    np_debug = np.where(np.linalg.norm(b - a, axis=-1) == 0)
    if np_debug[0].shape[0] > 0:
        return np.zeros_like(np.linalg.norm(b - a, axis=-1))
    result = np.abs(np.cross(b - a, c - a) / (np.linalg.norm(b - a, axis=-1)))
    result[result == np.inf] = 0
    result[result == -np.inf] = 0
    result[result == np.NAN] = 0
    return result


def caltheta(origin, point):
    fixed_point = point - origin
    return np.arctan2(fixed_point[:, 1], fixed_point[:, 0])


def check_point_validation(a, b):
    np_zero = np.where(np.linalg.norm(b - a) == 0)
    if np_zero[0].shape[0] > 0:
        return True
    else:
        return False


def validate_poly(poly):
    if check_point_validation(poly[0], poly[1]):
        return True
    elif check_point_validation(poly[1], poly[2]):
        return True
    elif check_point_validation(poly[2], poly[3]):
        return True
    elif check_point_validation(poly[3], poly[0]):
        return True
    elif check_point_validation(poly[0], poly[2]):
        return True
    elif check_point_validation(poly[1], poly[3]):
        return True
    else:
        return False


def generate_gt(im_size, polys, tags):
    h, w = im_size
    score_map = np.zeros((h, w), dtype=np.uint8)
    geo_map_up = np.zeros((h, w), dtype=np.float32)
    geo_map_down = np.zeros((h, w), dtype=np.float32)
    geo_map_left = np.zeros((h, w), dtype=np.float32)
    geo_map_right = np.zeros((h, w), dtype=np.float32)
    geo_map_theta = np.zeros((h, w), dtype=np.float32)

    # mask used during traning, to ignore some hard areas
    training_mask = np.ones((h, w), dtype=np.uint8)
    quad_boxes = np.zeros((FLAGS.max_text, 8), dtype=np.int32)
    box_label = np.zeros(FLAGS.max_text, dtype=np.int32)
    contours = np.zeros((FLAGS.max_text, FLAGS.max_polygon, 2), dtype=np.int32)
    cnt = 0
    for poly_idx, poly_tag in enumerate(zip(polys, tags)):
        poly = poly_tag[0]
        tag = poly_tag[1]

        np_poly = np.array([poly], dtype=np.int32)

        rect = cv2.minAreaRect(np_poly[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        rpoly = sort_poly(box)
        rpoly = recal_theta(rpoly)

        if np.array_equal(rpoly[0], rpoly[1]) or np.array_equal(rpoly[1], rpoly[2]) or \
                np.array_equal(rpoly[2], rpoly[3]) or np.array_equal(rpoly[3], rpoly[0]):
            # print("triangle")
            continue

        if tag > 0:
            if cnt < FLAGS.max_text:
                quad_boxes[cnt, 0] = rpoly[0, 0]
                quad_boxes[cnt, 1] = rpoly[0, 1]
                quad_boxes[cnt, 2] = rpoly[1, 0]
                quad_boxes[cnt, 3] = rpoly[1, 1]
                quad_boxes[cnt, 4] = rpoly[2, 0]
                quad_boxes[cnt, 5] = rpoly[2, 1]
                quad_boxes[cnt, 6] = rpoly[3, 0]
                quad_boxes[cnt, 7] = rpoly[3, 1]
                box_label[cnt] = int(tag)
                max_polygon = min(rpoly.shape[0], FLAGS.max_polygon)
                contours[cnt, :max_polygon] = rpoly[:max_polygon]
                cnt += 1

        x_center = np.mean(rpoly[:, 0])
        y_center = np.mean(rpoly[:, 1])

        signedarea = 0
        for i in range(rpoly.shape[0]):
            first_idx = i % 4
            sencod_idx = (i + 1) % 4
            x1, y1 = rpoly[first_idx]
            x2, y2 = rpoly[sencod_idx]
            signedarea += x1 * y2 - x2 * y1
        signedarea /= 2.0
        signed_length = np.sqrt(signedarea)

        area_length = 2 * (signed_length // float(32)) + 4

        score_rec_xmin = x_center - area_length
        score_rec_ymin = y_center - area_length

        score_rec_xmax = x_center + area_length
        score_rec_ymax = y_center + area_length

        score_rec = np.array([[score_rec_xmin, score_rec_ymin], [score_rec_xmax, score_rec_ymin],
                              [score_rec_xmax, score_rec_ymax], [score_rec_xmin, score_rec_ymax]], dtype=np.int32)

        cv2.fillPoly(score_map, [score_rec], 1)

        rect_map = np.zeros((h, w), dtype=np.uint16)
        cv2.fillPoly(rect_map, [score_rec], 255)
        xy_poly = np.where(rect_map == 255)

        np_xy_poly = np.transpose(np.array(xy_poly))
        np_lt_poly = np.repeat(rpoly[0][np.newaxis, :], np_xy_poly.shape[0], axis=0)[:, ::-1]
        np_rt_poly = np.repeat(rpoly[1][np.newaxis, :], np_xy_poly.shape[0], axis=0)[:, ::-1]
        np_rb_poly = np.repeat(rpoly[2][np.newaxis, :], np_xy_poly.shape[0], axis=0)[:, ::-1]
        np_lb_poly = np.repeat(rpoly[3][np.newaxis, :], np_xy_poly.shape[0], axis=0)[:, ::-1]

        geo_map_up[xy_poly] = callinetopoint(np_lt_poly, np_rt_poly, np_xy_poly)
        geo_map_left[xy_poly] = callinetopoint(np_lt_poly, np_lb_poly, np_xy_poly)
        geo_map_down[xy_poly] = callinetopoint(np_lb_poly, np_rb_poly, np_xy_poly)
        geo_map_right[xy_poly] = callinetopoint(np_rb_poly, np_rt_poly, np_xy_poly)
        theta_vector = (np_rt_poly - np_lt_poly) + (np_rb_poly - np_lb_poly)
        theta_vector_2 = (np_rb_poly - np_rt_poly) + (np_lb_poly - np_lt_poly)

        x_vector = theta_vector[:, 1]  # / (xmax - xmin)
        eps = np.full(x_vector.shape, 0.00001, dtype=np.float32)
        div_x_vector = np.where(x_vector == 0, eps, x_vector)
        x_vector_2 = theta_vector_2[:, 1]  # / (xmax - xmin)
        div_x_vector_2 = np.where(x_vector_2 == 0, eps, x_vector_2)

        div_first = np.arctan((theta_vector[:, 0]) / div_x_vector)
        div_second = np.arctan((theta_vector_2[:, 0]) / div_x_vector_2)
        div_second_recal = np.where(div_second < 0, np.pi / 2.0 + div_second, div_second - np.pi / 2.0)
        long_theta = np.where(np.linalg.norm(theta_vector, axis=-1) > np.linalg.norm(theta_vector_2, axis=-1),
                              div_first, div_second_recal)
        geo_map_theta[xy_poly] = long_theta

        if tag == 0:
            cv2.fillPoly(training_mask, np_poly, 0)
    return np.stack([geo_map_up, geo_map_left,
                     geo_map_down, geo_map_right,
                     score_map, geo_map_theta], -1)[::DIV, ::DIV, :], \
           training_mask[:, :, np.newaxis], quad_boxes, contours, box_label


def multi_poly(text_polys, rd_scale_x, rd_scale_y):
    for poly in text_polys:
        for point in poly:
            point[0] *= rd_scale_x
            point[1] *= rd_scale_y
            point[0] = int(point[0])
            point[1] = int(point[1])
    return text_polys


def add_poly(text_polys, w, h):
    for poly in text_polys:
        for point in poly:
            point[0] += w
            point[1] += h
            point[0] = int(point[0])
            point[1] = int(point[1])
    return text_polys


def get_tf_data(img_path, txt_path, input_size=-1):
    img_path = img_path.decode("utf-8")
    txt_path = txt_path.decode("utf-8")

    im = cv2.imread(img_path)
    if input_size == -1:
        input_size = np.random.choice([320, 448, 512, 1024], 1)[0]
    h, w, _ = im.shape

    if not os.path.exists(txt_path):
        print("text file {} does not exist".format(txt_path))
        raise ValueError

    text_polys, text_tags = load_annotation(txt_path)

    if np.random.rand() > 0.5 and FLAGS.flip:
        im, text_polys = flip_hori_image(im, text_polys)
    if np.random.rand() > 0.5 and FLAGS.flip:
        im, text_polys = flip_ver_image(im, text_polys)
    if FLAGS.rotate:
        im, text_polys = rotate_image(im, text_polys)

    if np.random.rand() > 0.5:
        im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)

    if np.random.rand() > 0.7 and FLAGS.blur:
        k = np.random.randint(5, 20)
        im = cv2.blur(im, (k, k))
    h, w, _ = im.shape

    new_h, new_w, _ = im.shape
    resize_h = input_size
    resize_w = input_size
    im = cv2.resize(im, dsize=(resize_w, resize_h))
    resize_ratio_3_x = resize_w / float(new_w)
    resize_ratio_3_y = resize_h / float(new_h)
    text_polys = multi_poly(text_polys, resize_ratio_3_x, resize_ratio_3_y)
    new_h, new_w, _ = im.shape
    geo_map, training_mask, quad_boxes, contours, box_label = generate_gt((new_h, new_w), text_polys, text_tags)

    return im[:, :, ::-1].astype(np.float32), geo_map.astype(np.float32), training_mask.astype(np.float32), \
           quad_boxes.astype(np.float32), contours.astype(np.float32), box_label.astype(np.int32), im.shape


def get_test_tf_data(img_path, txt_path, input_size=512):
    img_path = img_path.decode("utf-8")
    txt_path = txt_path.decode("utf-8")

    im = cv2.imread(img_path)

    h, w, _ = im.shape

    if not os.path.exists(txt_path):
        print("text file {} does not exist".format(txt_path))
        raise ValueError

    text_polys, text_tags = load_annotation(txt_path)

    h, w, _ = im.shape

    new_h, new_w, _ = im.shape
    max_h_w_i = np.max([new_h, new_w, input_size])
    im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
    value = np.random.rand()
    if value < 0.25:
        im_padded[:new_h, :new_w, :] = im.copy()

    elif value < 0.50:
        im_padded[:new_h, -new_w:] = im.copy()
        text_polys = add_poly(text_polys, max_h_w_i - new_w, 0)
    elif value < 0.75:
        im_padded[-new_h:, :new_w] = im.copy()
        text_polys = add_poly(text_polys, 0, max_h_w_i - new_h)
    else:
        im_padded[-new_h:, -new_w:] = im.copy()
        text_polys = add_poly(text_polys, max_h_w_i - new_w, max_h_w_i - new_h)
    im = im_padded
    # resize the image to input size
    new_h, new_w, _ = im.shape
    resize_h = input_size
    resize_w = input_size
    im = cv2.resize(im, dsize=(resize_w, resize_h))
    resize_ratio_3_x = resize_w / float(new_w)
    resize_ratio_3_y = resize_h / float(new_h)
    text_polys = multi_poly(text_polys, resize_ratio_3_x, resize_ratio_3_y)
    new_h, new_w, _ = im.shape
    geo_map, training_mask, quad_boxes, contours, box_label = generate_gt((new_h, new_w), text_polys, text_tags)

    return im[:, :, ::-1].astype(np.float32), geo_map.astype(np.float32), training_mask.astype(np.float32), \
           quad_boxes.astype(np.float32), contours.astype(np.float32), box_label.astype(np.int32), im.shape
