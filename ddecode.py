# -*- coding: utf-8 -*-

import cv2
import numpy as np
import random, sys
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
    parser.add_argument('-o', dest='ori', required=True)
    parser.add_argument('-i', dest='img', required=True)
    parser.add_argument('-r', dest='res', required=True)
    parser.add_argument('-a', dest='alpha', default=ALPHA)
    parser.add_argument('-c', dest='compress', default=False)
    parser.add_argument('-d', dest='decodeMethod', default="origin")

    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    ori = options.ori
    img = options.img
    res = options.res
    compress = options.compress
    decodeMethod = options.decodeMethod

    alpha = float(options.alpha)
    if not os.path.isfile(ori):
        parser.error("original image %s does not exist." % ori)
    if not os.path.isfile(img):
        parser.error("image %s does not exist." % img)
    decode(ori, img, res, alpha, compress, decodeMethod)

#使用灰度图像
def gray_image(image):
    # return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return cv2.medianBlur(image, 3)

# https://docs.opencv.org/2.3/modules/imgproc/doc/object_detection.html
def template_image(target, tpl):

    # 使用灰度图像
    target1 = gray_image(target)
    tpl1 = gray_image(tpl)
    
    th, tw = tpl.shape[:2]
    result = cv2.matchTemplate(target1, tpl1, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # print("min_val, max_val, min_loc, max_loc", min_val, max_val, min_loc, max_loc)
    tl = min_loc

    br = (tl[0] + tw, tl[1] + th)
    # cv2.rectangle(target, tl, br, [0, 0, 0])
    return tl, br

def merge(src, dst, center):
    # Create a rough mask around the airplane.
    src_mask = np.zeros(src.shape, src.dtype)

    # 当然我们比较懒得话，就不需要下面两行，只是效果差一点。
    # 不使用的话我们得将上面一行改为 mask = 255 * np.ones(obj.shape, obj.dtype) <-- 全白
    poly = np.array([ [4,80], [30,54], [151,63], [254,37], [298,90], [272,134], [43,122] ], np.int32)
    cv2.fillPoly(src_mask, [poly], (255, 255, 255))

    # 这是 飞机 CENTER 所在的地方
    # center = (800,100)

    # Clone seamlessly.
    output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)

    # 保存结果
    # cv2.imwrite("images/opencv-seamless-cloning-example.jpg", output)
    cv2.imshow("merge", output)
    return output
    
def delete(ori, left, top, right, bottom):

    rows, cols = ori.shape[:2]
    # print("rows, cols", rows, cols)

    # height top bottom
    for i in range(rows):
        # width left right
        for j in range(cols):
            if j >= left and j <= right and i >= top and i <= bottom:
                # print(i,j)
                ori[i,j] = (0,0,0)
    cv2.imshow('del',ori)
    return ori

def mergeWithPosition(ori, src, left, top, right, bottom):
    
    rows, cols = ori.shape[:2]
    print("rows, cols", rows, cols, "left, top, right, bottom", left, top, right, bottom)

    # height top bottom
    for i in range(rows):
        # width left right
        for j in range(cols):
            if j >= left and j <= right and i >= top and i <= bottom:
                # print(i, j)
                ori[i][j] = src[i-top-1][j-left-1]
    cv2.imshow('mergeWithPosition',ori)
    return ori

import eencode as ec

def decode(ori_path, img_path, res_path, alpha, compress, decodeMethod="origin"):
    ori = cv2.imread(ori_path)
    # cv2.imshow("ori", ori)

    h, w =ori.shape[:2]
    
    img = cv2.imread(img_path)
    # cv2.imshow("img", img)
    ih, iw = img.shape[:2]

    (left, top), (right, bottom) = template_image(ori, img)
    # print((left, top), (right, bottom))

    if ih != h and iw != w:
        encori = ec.encode(ori_path, "mark.png", "", 20)
        # cv2.imwrite("econd.png",encori,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        center=(left + int((right-left)/2), top + int((bottom-top)/2))
        # print("center",center)
        ori = delete(encori, left, top, right, bottom)
        
        newimg = mergeWithPosition(encori, img, left, top, right, bottom)
        # if newimg is not None:
        img = newimg
    
    ori_f = np.fft.fft2(ori)
    img_f = np.fft.fft2(img)

    height, width = ori.shape[0], ori.shape[1]
    watermark = (ori_f - img_f) / alpha
    watermark = np.real(watermark)
    res = np.zeros(watermark.shape)
    random.seed(height + width)
    x = list(range(int(height / 1)))
    y = list(range(width))
    random.shuffle(x)
    random.shuffle(y)
    for i in range(int(height / 1)):
        for j in range(width):
            res[x[i]][y[j]] = watermark[i][j]

    cv2.imwrite(res_path, res, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return



def saveImg(resImg, res_path):
    
    suffix = file_extension(res_path)
    if suffix == '.jpg' or suffix == '.jpeg':
        # https://docs.opencv.org/master/d4/da8/group__imgcodecs.html
        if compress:
            cv2.imwrite(res_path, resImg, [int(cv2.IMWRITE_JPEG_OPTIMIZE), 1])
        else:
            cv2.imwrite(res_path, resImg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    elif  suffix == '.png':
        if compress:
            # For PNG, it can be the compression level from 0 to 9. A higher value means a smaller size and longer compression time. 
            cv2.imwrite(res_path, resImg, [int(cv2.IMWRITE_PNG_COMPRESSION), 5])
        else:
            # https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#gaa60044d347ffd187161b5ec9ea2ef2f9
            cv2.imwrite(res_path, resImg, [int(cv2.IMWRITE_PNG_STRATEGY), cv2.IMWRITE_PNG_STRATEGY_DEFAULT ])
    else:
        print("WRONG OUTPUT TYPE", suffix)
        cv2.imwrite(res_path, resImg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

if __name__ == '__main__':
    main()

    cv2.waitKey(0)
    cv2.destroyAllWindows()