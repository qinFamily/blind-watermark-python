# coding=utf-8
import cv2
import numpy as np
import random
import os, os.path
from argparse import ArgumentParser
ALPHA = 5

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('-i', dest='img', required=True)
    parser.add_argument('-w', dest='wm', required=True)
    parser.add_argument('-o', dest='res', required=True)
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

def rotate(src, angle):
    radian = (float) (angle /180.0 * np.pi)
    # 填充图像
    rows, cols = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).shape    #cvtcolor 是颜色转换参数
    maxBorder =(int) (max(rows, cols)) # 即为sqrt(2)*max  / np.sin(radian)
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
    cv2.imshow("rotate", out1)
    return out1

    # 透视变换
    # pts3 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    # pts4 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    # M_perspective = cv2.getPerspectiveTransform(pts3,pts4)
    # out1 = cv2.warpPerspective(out, M_perspective, (newcols, newrows) )

    # cv2.imwrite(outputName + os.path.sep + 'pic_' + str(angle) + '.jpg', out1)

    # //计算图像旋转之后包含图像的最大的矩形
    # sinVal = abs(np.sin(radian))
    # cosVal = abs(np.cos(radian))
    # ((int)(src.cols * cosVal +src.rows * sinVal),(int)(src.cols * sinVal + src.rows * cosVal) )
    # int x = (out1.cols - targetSize.width) / 2
    # int y = (out1.rows - targetSize.height) / 2
    # Rect rect(x, y, targetSize.width, targetSize.height)
    # dst = Mat(dst,rect)


def encode(img_path, wm_path, res_path, alpha):
    img = cv2.imread(img_path)
    img_f = np.fft.fft2(img)
    height, width, channel = np.shape(img)
    watermark = cv2.imread(wm_path)
    
    ## 水印图像旋转
    newWatermark = rotate(watermark, 30)

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