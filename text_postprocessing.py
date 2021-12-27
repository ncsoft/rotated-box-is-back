from argparse import ArgumentParser
import numpy as np
import os
import cv2
from nms import rbox_gpu_nms


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
                      required=True,
                      type=str)
    args.add_argument("-o", "--output", help="", default="./rec_output",
                      type=str)
    args.add_argument("-m", "--mode", help="", default="quad",
                      type=str)
    args.add_argument("-t", "--threshold", help="", default=0.3,
                      type=float)
    args.add_argument("-e", "--exclude", help="", default=False,
                      type=bool)
    return parser


def get_file_by_ext(test_data_path, exts=['txt',]):
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    print(test_data_path)
    for parent, dirnames, filenames in os.walk(test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break

    print('Find {} images'.format(len(files)))
    files = sorted(files)

    return files


def main():
    args = build_argparser().parse_args()
    text_file_path = get_file_by_ext(args.input)

    whole_instance = 0
    after_nms_instance = 0
    threshold_filtered = 0
    os.makedirs(args.output, exist_ok=True)
    for each_text_file in text_file_path:
        base_name = os.path.basename(each_text_file)
        output_file_name = os.path.join(args.output, base_name)

        with open(each_text_file, "r") as f:
            whole_string = f.readlines()
        with open(output_file_name, "w") as fo:
            after_nms = []
            for each_string in whole_string:
                each_string = each_string.split(",")
                x1 = int(each_string[0])
                x2 = int(each_string[2])
                x3 = int(each_string[4])
                x4 = int(each_string[6])

                y1 = int(each_string[1])
                y2 = int(each_string[3])
                y3 = int(each_string[5])
                y4 = int(each_string[7])

                quad = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

                signedarea = 0
                for i in range(quad.shape[0]):
                    first_idx = i % 4
                    sencod_idx = (i+1) % 4
                    x1, y1 = quad[first_idx]
                    x2, y2 = quad[sencod_idx]
                    signedarea += x1 * y2 - x2 * y1
                if signedarea < 0:
                    print(each_text_file)

                else:
                    score = min(float(each_string[-1]), 1.0)
                    score = max(score, 0.0)
                    quad = quad.reshape([-1])
                    score = np.array(score).reshape([-1])

                    after_nms.append(np.concatenate([quad, score], axis=-1))

            after_nms = np.array(after_nms)
            whole_instance += after_nms.shape[0]
            np_after_nms_idx = rbox_gpu_nms(after_nms.astype('float32'),
                                            0.2)
            after_nms = after_nms[np_after_nms_idx.astype(np.int32)]
            after_nms_instance += after_nms.shape[0]
            for each_boxes in after_nms:
                quad = each_boxes[:8].reshape([4, 2]).astype(np.int32)
                score = each_boxes[-1]
                if np.array_equal(quad, np.zeros_like(quad)) and score == 0:
                    print("null")
                else:
                    if score > args.threshold:
                        if args.exclude:
                            if args.mode == "quad":
                                fo.write('{0},{1},{2},{3},{4},{5},{6},{7}\r\n'.format(
                                    quad[0, 0], quad[0, 1],
                                    quad[1, 0], quad[1, 1],
                                    quad[2, 0], quad[2, 1],
                                    quad[3, 0], quad[3, 1]
                                ))
                            else:
                                xmin = np.amin(quad[:, 0])
                                xmax = np.amax(quad[:, 0])
                                ymin = np.amin(quad[:, 1])
                                ymax = np.amax(quad[:, 1])
                                fo.write('{0},{1},{2},{3}\r\n'.format(
                                    xmin, ymin, xmax, ymax
                                ))

                        else:
                            if args.mode == "quad":
                                fo.write('{0},{1},{2},{3},{4},{5},{6},{7},{8:0.2f}\r\n'.format(
                                    quad[0, 0], quad[0, 1],
                                    quad[1, 0], quad[1, 1],
                                    quad[2, 0], quad[2, 1],
                                    quad[3, 0], quad[3, 1],
                                    score
                                ))
                            else:
                                xmin = np.amin(quad[:, 0])
                                xmax = np.amax(quad[:, 0])
                                ymin = np.amin(quad[:, 1])
                                ymax = np.amax(quad[:, 1])
                                fo.write('{0},{1},{2},{3},{4:0.2f}\r\n'.format(
                                    xmin, ymin, xmax, ymax, score
                                ))

                    else:
                        threshold_filtered += 1
    print("Before nms and threshold filtering num : {}".format(whole_instance))
    print("nms : {}".format(after_nms_instance))
    print("nms diff : {}".format(whole_instance - after_nms_instance))
    print("threshold filtering diff : {}".format(threshold_filtered))
    print("After nms and threshold filtering num : {}".format(after_nms_instance - threshold_filtered))


if __name__ == "__main__":
    main()
