# coding=utf-8
import cv2
import numpy as np
import random
import os, os.path
from argparse import ArgumentParser
ALPHA = 5


def file_extension(path): 
    pathes=os.path.splitext(path)
    if len(pathes) > 1:
        return pathes[1]
    return "txt"

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--original', dest='ori', required=True)
    parser.add_argument('--image', dest='img', required=True)
    parser.add_argument('--result', dest='res', required=True)
    parser.add_argument('--alpha', dest='alpha', default=ALPHA)
    parser.add_argument('--compress', dest='compress', default=False)
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    ori = options.ori
    img = options.img
    res = options.res
    compress = options.compress

    alpha = float(options.alpha)
    if not os.path.isfile(ori):
        parser.error("original image %s does not exist." % ori)
    if not os.path.isfile(img):
        parser.error("image %s does not exist." % img)
    decode(ori, img, res, alpha, compress)


def decode(ori_path, img_path, res_path, alpha, compress):
    ori = cv2.imread(ori_path)
    img = cv2.imread(img_path)
    ori_f = np.fft.fft2(ori)
    img_f = np.fft.fft2(img)
    height, width = ori.shape[0], ori.shape[1]
    watermark = (ori_f - img_f) / alpha
    watermark = np.real(watermark)
    res = np.zeros(watermark.shape)
    random.seed(height + width)
    x = list(range(int(height / 2)))
    y = list(range(width))
    random.shuffle(x)
    random.shuffle(y)
    for i in range(int(height / 2)):
        for j in range(width):
            res[x[i]][y[j]] = watermark[i][j]
    suffix = file_extension(res_path)
    if suffix == '.jpg1' or suffix == '.jpeg1':
        # https://docs.opencv.org/master/d4/da8/group__imgcodecs.html
        if compress:
            cv2.imwrite(res_path, res, [int(cv2.IMWRITE_JPEG_OPTIMIZE), 1])
        else:
            cv2.imwrite(res_path, res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    elif  suffix == '.png1':
        if compress:
            # For PNG, it can be the compression level from 0 to 9. A higher value means a smaller size and longer compression time. 
            cv2.imwrite(res_path, res, [int(cv2.IMWRITE_PNG_COMPRESSION), 5])
        else:
            # https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#gaa60044d347ffd187161b5ec9ea2ef2f9
            cv2.imwrite(res_path, res, [int(cv2.IMWRITE_PNG_STRATEGY), cv2.IMWRITE_PNG_STRATEGY_DEFAULT ])
    else:
        print("WRONG OUTPUT TYPE", suffix)
        cv2.imwrite(res_path, res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

if __name__ == '__main__':
    main()
