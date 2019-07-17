# git
```shell
git clone git@github.com:qinFamily/blind-watermark-python.git
```


# blind-watermark-python

# Usage
```shell
python encode.py --image <image file> --watermark <watermark file> --result <result file>

python decode.py --original <original image file> --image <image file> --result <result file>

Use --alpha to change the alpha (default 5.0).
```
# Example
## encode:
original image<br>
![image](https://github.com/qinFamily/blind-watermark-python/blob/master/imgs/ori.png)

watermark<br>
![image](https://github.com/qinFamily/blind-watermark-python/blob/master/imgs/watermark.png)


```shell
python encode.py --image imgs/ori.png --watermark imgs/watermark.png --result imgs/res.png
```
result<br>
![image](https://github.com/qinFamily/blind-watermark-python/blob/master/imgs/res.png)

## decode:
```shell
python decode.py --original imgs/ori.png --image imgs/res.png --result imgs/extract.png
```
watermark<br>
![image](https://github.com/qinFamily/blind-watermark-python/blob/master/imgs/extract.png)
