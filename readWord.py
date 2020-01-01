# Created on Dec 31st 2019
# Author: Changzhi Cai
# Contact me: caichangzhi97@gmail.com

# import package
from PIL import Image
import pytesseract
import cv2
import os

preprocess = 'blur'

# read the image
image = cv2.imread('scan.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

if preprocess == "thresh":
    gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

if preprocess == "blur":
    gray = cv2.medianBlur(gray,3)

# output file    
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename,gray)

# read text
text = pytesseract.image_to_string(Image.open(filename))
print(text)
os.remove(filename)

# show the result
cv2.imshow("Image",image)
cv2.imshow("Output",gray)
cv2.waitKey(0)                                   
