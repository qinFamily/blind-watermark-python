#读取文件每一行，写到图片中间位置，并以读取的字符命名水印图片
#encoding:utf-8
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
src = Image.open('rm1-guss.png')
file = open('name.txt','r',encoding='utf8')
fnt = ImageFont.truetype(r'./nunai.ttf',32,encoding='nunai')
# fnt = ImageFont.truetype(r'C:\\Windows\\Fonts\\STKAITI.TTF',32)

for line in file:
	print(line)
	img = src.copy()
	draw = ImageDraw.Draw(img)
	line = line.strip().lstrip().rstrip('\n')
	draw.text((img.size[0]/2,img.size[1]/2),line,(0,255,0),font=fnt)
	str = (r'./',('.jpg'))
	str = u''+line.join(str)+''
	img.save(str)
	print(str)
file.close()

'''
--------------------- 
作者：2℃ 
来源：CSDN 
原文：https://blog.csdn.net/weixin_39455973/article/details/85943752 
版权声明：本文为博主原创文章，转载请附上博文链接！
'''
