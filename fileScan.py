# Created on Dec 31st 2019
# Author: Changzhi Cai
# Contact me: caichangzhi97@gmail.com

# import package
import numpy as np
import cv2

def resize(image,width = None,height = None,inter = cv2.INTER_AREA):
	dim = None
	(h,w) = image.shape[:2]
	if width is None and height is None:
		return image
	if width is None:
		r = height/float(h)
		dim = (int(w*r),height)
	else:
		r = width/float(w)
		dim = (width,int(h * r))
	resized = cv2.resize(image,dim,interpolation=inter)
	return resized

def order_points(pts):
    
    # totally 4 points
	rect = np.zeros((4,2), dtype = "float32")

    # point: 0 - left_up, 1 - right_up, 2 - right_down, 3 - left_down
    # calculate point 0 and point 2
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

    # calculate point 1 and point 3
	diff = np.diff(pts,axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def four_point_transform(image,pts):
    
    # get import points
	rect = order_points(pts)
	(tl,tr,br,bl) = rect

    # calculate the width and take the max value
	widthA = np.sqrt(((br[0]-bl[0])**2)+((br[1]-bl[1])**2))
	widthB = np.sqrt(((tr[0]-tl[0])**2)+((tr[1]-tl[1])**2))
	maxWidth = max(int(widthA),int(widthB))

    # calculate the height and take the max value
	heightA = np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))
	heightB = np.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2))
	maxHeight = max(int(heightA),int(heightB))

    # loaction after transformation
	dst = np.array([
		[0,0],
		[maxWidth-1,0],
		[maxWidth-1,maxHeight-1],
		[0,maxHeight-1]],dtype = "float32")

    # calculate transformed matrix
	M = cv2.getPerspectiveTransform(rect,dst)
	warped = cv2.warpPerspective(image,M,(maxWidth, maxHeight))

	return warped

# read the image
image = cv2.imread("receipt.jpg")
ratio = image.shape[0]/500.0
orig = image.copy()

# resize the image
image = resize(orig,height = 500)

# do the preprocess
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)
edged = cv2.Canny(gray,75,200)

# show the result after preprocessing
print("Step 1: Edge Detection")
cv2.imshow("image",image)
cv2.imshow("edged",edged)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# contour detection
cnts = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]
cnts = sorted(cnts,key = cv2.contourArea,reverse = True)[:5]

# go through the contours
for c in cnts:
    
    # calculate contour approximation
    # c is the imported point set
    # epsilon is the maximum distance from the original contour to the approximate contour
    # True means it is closed
    peri = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,0.02*peri,True)
    
    # if it detects 4 points, take it
    if len(approx) == 4:
        screenCnt = approx
        break

# show the result
print("Step 2: Get Contour")
cv2.drawContours(image,[screenCnt],-1,(0,255,0),2)
cv2.imshow("Outline",image)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# perspective transformation
warped = four_point_transform(orig,screenCnt.reshape(4,2)*ratio)

# do the binary process
warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped,100,255,cv2.THRESH_BINARY)[1]
cv2.imwrite('scan.jpg',ref)

# show the results
print("Step 3: Transform")
cv2.imshow("Original",resize(orig,height = 650))
cv2.imshow("Scanned", resize(ref,height = 650))
cv2.waitKey(1000)
cv2.destroyAllWindows()