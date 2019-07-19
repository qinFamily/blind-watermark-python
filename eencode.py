# coding=utf-8
import cv2
import numpy as np
import random, sys
import os, os.path
from argparse import ArgumentParser
ALPHA = 5

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('-i', dest='img', required=True)
    parser.add_argument('-w', dest='wm', required=True)
    parser.add_argument('-r', dest='res', required=True)
    parser.add_argument('-a', dest='alpha', default=ALPHA)
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    img = options.img
    wm = options.wm
    res = options.res
    alpha = float(options.alpha)

    if not os.path.isfile(img):
        parser.error("image %s does not exist." % img)
    if not os.path.isfile(wm):
        parser.error("watermark %s does not exist." % wm)
    encode(img, wm, res, alpha)

# https://blog.csdn.net/jazywoo123/article/details/17353069
#读取jpg，添加alpha通道
def addAlpha(image):
    temp_image = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
                temp = np.append(image[i][j], 255)
                temp_image.append(temp)
    return np.array(temp_image)

def imread(path):
    '''
    # https://docs.opencv.org/2.3/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#cv2.imread
    flags：读入图片的标志 
        cv2.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道
        cv2.IMREAD_GRAYSCALE：读入灰度图片
        cv2.IMREAD_UNCHANGED：顾名思义，读入完整图片，包括alpha通道
    '''
    img = cv2.imread(path, flags=cv2.IMREAD_UNCHANGED)
    print(img.ndim)
    if img.ndim != 4:
        newimg=addAlpha(img)
        newimg.resize((img.shape[0],img.shape[1],4))
        return newimg
    return img

# 灰度
# newrows, newcols = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).shape #cvtcolor 是颜色转换参数
'''
定义裁剪函数，四个参数分别是：
左上角横坐标x0
左上角纵坐标y0
裁剪宽度w
裁剪高度h
'''
crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]


def rotate_bound(image, angle):
    '''
        
    --------------------- 
    作者：阿尔法先生 
    来源：CSDN 
    原文：https://blog.csdn.net/weixin_43730228/article/details/84979806 
    版权声明：本文为博主原创文章，转载请附上博文链接！
    '''
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    shuchu=cv2.warpAffine(image, M, (nW, nH))
    return shuchu

#平铺 将img的图片平铺到newWidth, newHeight的图像中
def fillEmpty(img, dstHeight, dstWidth):
    h, w = img.shape[:2]
    
    newHeight = h * int(np.ceil(dstHeight/h))
    newWidth  = w * int(np.ceil(dstWidth/w))

    #generate a blank photo
    newImg = np.zeros(shape=(newHeight,newWidth,3),dtype=np.uint8)
    # cv2.imshow('img0',newImg)
    #copy each pixels'value
    img_x = 0
    img_y = 0
    for now_y in range(newHeight):
        for now_x in range(newWidth):
            newImg[now_y,now_x,0] = img[img_y,img_x,0]
            newImg[now_y,now_x,1] = img[img_y,img_x,1]
            newImg[now_y,now_x,2] = img[img_y,img_x,2]
            img_x +=1
            #超过原图列数范围，归0，重新开始复制
            if img_x >= w:
                img_x = 0
        img_y +=1
        if img_y >=h:
            img_y = 0
        # print('.')
    # 变形
    # newImg = cv2.resize(newImg, (dstWidth, dstHeight))
    # 裁剪
    newImg = newImg[0:dstHeight, 0:dstWidth]
    # cv2.namedWindow('fillEmpty')
    cv2.imshow('fillEmpty',newImg)
    cv2.imwrite("newmark.png",newImg)


def encode(img_path, wm_path, res_path, alpha):
    img = cv2.imread(img_path)
    img_f = np.fft.fft2(img)
    height, width, channel = np.shape(img)
    watermark = cv2.imread(wm_path)
    
    ## 水印图像旋转
    # newWatermark = rotate(watermark, 30)
    newWatermark = rotate_bound(watermark, -30)
    cv2.imshow('rotate_bound', newWatermark)

    # 平铺
    fillEmpty(newWatermark, height, width)

    wm_height, wm_width = newWatermark.shape[0], newWatermark.shape[1]
    
    x, y = list(range(int(height / 2))), list(range(width))
    random.seed(height + width)
    random.shuffle(x)
    random.shuffle(y)
    tmp = np.zeros(img.shape)
    for i in range(int(height / 2)):
        for j in range(width):
            if x[i] < wm_height and y[j] < wm_width:
                tmp[i][j] = newWatermark[x[i]][y[j]]
                tmp[height - 1 - i][width - 1 - j] = tmp[i][j]
    res_f = img_f + alpha * tmp
    res = np.fft.ifft2(res_f)
    res = np.real(res)
    
    if len(res_path) > 0 :
        # cv2.imwrite(res_path, res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imwrite(res_path, res, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    else:
        return res
    

if __name__ == '__main__':
    # print(file_extension("a.txt.jpg.png"))
    main()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
https://www.cnblogs.com/hanxiaosheng/p/9559996.html    


读入一副图像，给图片加文本
# img=cv2.imread('1.jpg',cv2.IMREAD_COLOR)
img=cv2.imread('1.png',cv2.IMREAD_COLOR)    # 打开文件
font = cv2.FONT_HERSHEY_DUPLEX  # 设置字体
# 图片对象、文本、像素、字体、字体大小、颜色、字体粗细
imgzi = cv2.putText(img, "zhengwen", (1100, 1164), font, 5.5, (0, 0, 0), 2,)
'''

'''
    img = cv2.imread(img_path)
    img_f = np.fft.fft2(img)
    height, width, channel = np.shape(img)
   
    ##  水印
    watermark = cv2.imread(wm_path)
    ## 水印图像旋转
    newWatermark = rotate(watermark, 30)
    wm_height, wm_width = newWatermark.shape[0], newWatermark.shape[1]
    # print(wm_height, wm_width)

    x, y = list(range(int(height/5))), list(range(width))
    
    random.seed(height + width)
    random.shuffle(x)
    random.shuffle(y)
    tmp = np.zeros(img.shape)
    for i in range(int(height / 5)):
        for j in range(width):
            if x[i] < wm_height and y[j] < wm_width:
                tmp[i][j] = newWatermark[x[i]][y[j]]
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
'''