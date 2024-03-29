# -*- coding: utf-8 -*-

import math
 
#define PI 3.1415
 
#复数类
class complex:
    def __init__(self):
        self.real = 0.0
        self.image = 0.0
 
#复数乘法
def mul_ee(complex0, complex1):
    complex_ret = complex()
    complex_ret.real = complex0.real * complex1.real - complex0.image * complex1.image
    complex_ret.image = complex0.real * complex1.image + complex0.image * complex1.real
    return complex_ret
 
#复数加法
def add_ee(complex0, complex1):
    complex_ret = complex()
    complex_ret.real = complex0.real + complex1.real
    complex_ret.image = complex0.image + complex1.image
    return complex_ret
 
#复数减法
def sub_ee(complex0, complex1):
    complex_ret = complex()
    complex_ret.real = complex0.real - complex1.real
    complex_ret.image = complex0.image - complex1.image
    return complex_ret
 
#对输入数据进行倒序排列
def forward_input_data(input_data, num):    
    j = int(num / 2)
    print(j)
    for i in range(1, num - 2):        
        if(i < j):
            complex_tmp = input_data[i]
            input_data[i] = input_data[j]
            input_data[j] = complex_tmp
            print("forward x[%d] <==> x[%d]" % (i, j))
        k = int(num / 2)
        while (j >= k):
            j = j - k
            k = int(k / 2)
        j = j + k
 
#实现1D FFT
def fft_1d(in_data, num):
    PI = 3.1415926
    forward_input_data(in_data, num) #倒序输入数据    
 
    #计算蝶形级数，也就是迭代次数
    M = 1 #num = 2^m
    tmp = num / 2;
    while (tmp != 1):
        M = M + 1
        tmp = tmp / 2
    print("FFT level：%d" % M)
 
    complex_ret = complex()
    for L in range(1, M + 1):
        B = int(math.pow(2, L -1)) #B为指数函数返回值，为float，需要转换integer
        for J in range(0, B):
            P = math.pow(2, M - L) * J            
            for K in range(J, num, int(math.pow(2, L))):
                print("L:%d B:%d, J:%d, K:%d, P:%f" % (L, B, J, K, P))
                complex_ret.real = math.cos((2 * PI / num) *  P)
                complex_ret.image = -math.sin((2 * PI / num) * P)
                complex_mul = mul_ee(complex_ret, in_data[K + B])
                complex_add = add_ee(in_data[K], complex_mul)
                complex_sub = sub_ee(in_data[K], complex_mul)
                in_data[K] = complex_add
                in_data[K + B] = complex_sub
                print("A[%d] real: %f, image: %f" % (K, in_data[K].real, in_data[K].image))
                print("A[%d] real: %f, image: %f" % (K + B, in_data[K + B].real, in_data[K + B].image))
 
def test_fft_1d():
    in_data = [2,3,4,5,7,9,10,11] #待测试的8点元素
    #变量data为长度为8、元素为complex类实例的list，用于存储输入数据
    data = [(complex()) for i in range(len(in_data))]
    #将8个测试点转换为complex类的形式，存储在变量data中
    for i in range(len(in_data)):
        data[i].real = in_data[i]
        data[i].image = 0.0
         
    #输出FFT需要处理的数据
    print("The input data:")
    for i in range(len(in_data)):
        print("x[%d] real: %f, image: %f" % (i, data[i].real, data[i].image))
          
    fft_1d(data, 8)
 
    #输出经过FFT处理后的结果
    print("The output data:")
    for i in range(len(in_data)):
        print("X[%d] real: %f, image: %f" % (i, data[i].real, data[i].image))
    
#test the 1d fft
test_fft_1d()