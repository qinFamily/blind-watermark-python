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
    return "null"

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--image', dest='img', required=True)
    parser.add_argument('--watermark', dest='wm', required=True)
    parser.add_argument('--result', dest='res', required=True)
    parser.add_argument('--alpha', dest='alpha', default=ALPHA)
    parser.add_argument('--compress', dest='compress', default=False)
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    img = options.img
    wm = options.wm
    res = options.res
    alpha = float(options.alpha)
    compress = options.compress

    if not os.path.isfile(img):
        parser.error("image %s does not exist." % img)
    if not os.path.isfile(wm):
        parser.error("watermark %s does not exist." % wm)
    encode(img, wm, res, alpha, compress)


def encode(img_path, wm_path, res_path, alpha, compress):
    img = cv2.imread(img_path)
    img_f = np.fft.fft2(img)
    height, width, channel = np.shape(img)
    watermark = cv2.imread(wm_path)
    wm_height, wm_width = watermark.shape[0], watermark.shape[1]
    x, y = list(range(int(height/3))), list(range(width))
    # print(x)
    # sys.exit(0)
    random.seed(height + width)
    random.shuffle(x)
    random.shuffle(y)
    tmp = np.zeros(img.shape)
    for i in range(int(height / 3)):
        for j in range(width):
            if x[i] < wm_height and y[j] < wm_width:
                tmp[i][j] = watermark[x[i]][y[j]]
                tmp[height - 1 - i][width - 1 - j] = tmp[i][j]
    res_f = img_f + alpha * tmp
    res = np.fft.ifft2(res_f)
    res = np.real(res)
    
    # cv2.imwrite(res_path, res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    suffix = file_extension(res_path)
    if suffix == '.jpg' or suffix == '.jpeg':
        # https://docs.opencv.org/master/d4/da8/group__imgcodecs.html
        if compress:
            cv2.imwrite(res_path, res, [int(cv2.IMWRITE_JPEG_OPTIMIZE), 1])
        else:
            cv2.imwrite(res_path, res, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    elif  suffix == '.png':
        if compress:
            # For PNG, it can be the compression level from 0 to 9. A higher value means a smaller size and longer compression time. 
            cv2.imwrite(res_path, res, [int(cv2.IMWRITE_PNG_COMPRESSION), 5])
        else:
            # https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#gaa60044d347ffd187161b5ec9ea2ef2f9
            cv2.imwrite(res_path, res, [int(cv2.IMWRITE_PNG_STRATEGY), cv2.IMWRITE_PNG_STRATEGY_DEFAULT ])
    else:
        print("WRONG OUTPUT TYPE", suffix)


if __name__ == '__main__':
    # print(file_extension("a.txt.jpg.png"))
    main()
