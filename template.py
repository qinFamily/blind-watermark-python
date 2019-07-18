# -*- coding=utf-8 -*-

import cv2 as cv
import numpy as np

from argparse import ArgumentParser

# https://blog.csdn.net/qq_41603898/article/details/82219291
# https://docs.opencv.org/2.3/modules/imgproc/doc/object_detection.html
def template_image(target, tpl):

    # 二值化
    # target1 = threshold(target)
    # tpl1 = threshold(tpl)

    target1 = gray_image(target)
    tpl1 = gray_image(tpl)

    cv.imshow("modul", tpl1)
    cv.imshow("yuan", target1)
    '''
    TM_SQDIFF_NORMED标准平方差匹配
        标准差是越小为0代表匹配上了
    TM_CCORR_NORMED标准相关性匹配
        相关性是越接近1代表匹配上了
    TM_CCOEFF_NORMED标准相关性系数匹配
        相关性越接近1越好
    '''
    # methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED, cv.CV_TM_CCORR, cv,CV_TM_CCOEFF]
    methods = [cv.TM_SQDIFF_NORMED]
    th, tw = tpl.shape[:2]
    for md in methods:
        result = cv.matchTemplate(target1, tpl1, md)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        print(md, min_val, max_val, min_loc, max_loc)
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0] + tw, tl[1] + th)
        cv.rectangle(target, tl, br, [0, 0, 0])
        cv.imshow("pipei"+np.str(md), target)
 
'''
--------------------- 
作者：yangyang688 
来源：CSDN 
原文：https://blog.csdn.net/yangyang688/article/details/82998065 
版权声明：本文为博主原创文章，转载请附上博文链接！
'''

#局部阈值
def threshold(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # cv.imshow("原来", gray)
    # 1 gray 2 maxValue 3 局部的方式：高斯(权重，越近越大)，均值(取均值) 4 阈值类型，和全局一样
    # 5 blockSize 必须是奇数,图像分成块的大小 6 常量(假设用均值，其他值-均值>10才设置为黑色或者白色)
    # binary1 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 10)
    # cv.imshow("局部1", binary1)
    # 高斯处理
    binary2 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)
    # cv.imshow("局部2", binary2)
    return binary2
 

 
#求出图像均值作为阈值来二值化
def custom_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # cv.imshow("gray", gray)
    h, w = gray.shape[:2]
    m = np.reshape(gray, [1, w*h])#化为一维数组
    mean = m.sum() / (w*h)
    print("mean: ", mean)
    # ret, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    return cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 10)

#使用灰度图像
def gray_image(image):
    # return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return cv.medianBlur(image, 3)

# img = cv.imread("22.jpg")
# threshold(img)

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('-i', dest='img', default="2.png")
    parser.add_argument('-t', dest='templete', default='1.png')
    return parser

if __name__ == '__main__':
    parser = build_parser()
    options = parser.parse_args()
    img = options.img
    templete = options.templete

    target = cv.imread(img)
    tpl = cv.imread(templete)
    template_image(target, tpl)
    cv.waitKey(0)
    cv.destroyAllWindows()
