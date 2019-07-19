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
    parser.add_argument('-i', dest='img', required=True)
    parser.add_argument('-o', dest='res', required=True)
    parser.add_argument('-f', dest='flip', default=0)
    parser.add_argument('-d', dest='degree', default=30)
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    img = options.img
    res = options.res
    flip = int(options.flip)
    degree = float(options.degree)

    if not os.path.isfile(img):
        parser.error("image %s does not exist." % img)
    # encode(img, res, flip)
    imgmat = cv2.imread(img, 1)
    # rotate_img(imgmat, 45, res)
    rotate_arbitrarily_angle(imgmat, res, degree)

'''旋转'''
def rotate_img(img, rotate_angle, outputdir):

    if not os.path.exists(outputdir) and not os.path.isdir(outputdir):  #a判断当前路径是否为绝对路径或者是否为路径
        os.mkdir(outputdir)  #生成单级路径

    rows, cols = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).shape    #cvtcolor 是颜色转换参数

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotate_angle, 1)
    dst = cv2.warpAffine(img, M, (np.max(cols, rows),np.max(cols, rows)))

    cv2.imwrite(outputdir + os.path.sep + 'pic_' + str(rotate_angle) + '.jpg', dst)

# https://blog.csdn.net/u013263891/article/details/83932479

def rotate_arbitrarily_angle(src, dst, angle):
    
    radian = (float) (angle /180.0 * np.pi)
    # 填充图像
    rows, cols = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).shape    #cvtcolor 是颜色转换参数
    # print(rows, cols, max(rows, cols))
    maxBorder =(int) (max(cols, rows)) # 即为sqrt(2)*max  / np.sin(radian)
    # print("rows =", rows, ", cols =", cols, ", max =", max(rows, cols), ", maxBorder =", maxBorder)
    dx = int((maxBorder - cols)/2)
    dy = int((maxBorder - rows)/2)
    # print("dx, dy",dx, dy)


    # 固定值边框，统一都填充0也称为zero padding

    '''
        给源图像增加边界

        cv2.copyMakeBorder(src,top, bottom, left, right ,borderType,value)
        src:源图像
        top,bottem,left,right: 分别表示四个方向上边界的长度
        borderType: 边界的类型
            有以下几种：
            BORDER_REFLICATE　　　  # 直接用边界的颜色填充， aaaaaa | abcdefg | gggg
            BORDER_REFLECT　　　　  # 倒映，abcdefg | gfedcbamn | nmabcd
            BORDER_REFLECT_101　　 # 倒映，和上面类似，但在倒映时，会把边界空开，abcdefg | egfedcbamne | nmabcd
            BORDER_WRAP　　　　  　# 额。类似于这种方式abcdf | mmabcdf | mmabcd
            BORDER_CONSTANT　　　　# 常量，增加的变量通通为value色 [value][value] | abcdef | [value][value][value]
    
    '''
    out = cv2.copyMakeBorder(src, dy, dy, dx, dx, cv2.BORDER_CONSTANT, value=0)
    newrows, newcols = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).shape
    # print("newrows, newcols", newrows, newcols)
    # 旋转
    center = ((float)(newcols/2) , (float) (newrows/2))
    affine_matrix = cv2.getRotationMatrix2D( center, angle, 1.0 )
    # 仿射
    out1 = cv2.warpAffine(out, affine_matrix, (newcols, newrows) )
    # 透视变换
    # pts3 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    # pts4 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    # M_perspective = cv2.getPerspectiveTransform(pts3,pts4)
    # out1 = cv2.warpPerspective(out, M_perspective, (newcols, newrows) )

    cv2.imwrite(dst + os.path.sep + 'pic_' + str(angle) + '.jpg', out1)

    # //计算图像旋转之后包含图像的最大的矩形
    sinVal = abs(np.sin(radian))
    cosVal = abs(np.cos(radian))
    ((int)(src.cols * cosVal +src.rows * sinVal),(int)(src.cols * sinVal + src.rows * cosVal) )
    int x = (out1.cols - targetSize.width) / 2
    int y = (out1.rows - targetSize.height) / 2
    Rect rect(x, y, targetSize.width, targetSize.height)
    dst = Mat(dst,rect)


def encode(img_path, res_path, flip):
    img = cv2.imread(img_path)
    # height, width, channel = np.shape(img)
    wm_height, wm_width = img.shape[0], img.shape[1]
    print(wm_height, wm_width)

    # 水印矩阵转置
    trans_img = cv2.transpose(img)
    print(trans_img)
    # 0: 沿X轴翻转； >0: 沿Y轴翻转； <0: 沿X轴和Y轴翻转
    new_img = cv2.flip(trans_img, flip)
    print(new_img)
    cv2.imwrite(res_path, new_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90 ])

if __name__ == '__main__':
    # print(file_extension("a.txt.jpg.png"))
    main()
