import cv2

ret, img = cv2.VideoCapture(0).read()
# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
print('pre', img)
cv2.imwrite('../image/same.jpg', img)
cv2.imwrite('../image/same.bmp', img)
jpg = cv2.imread('../image/same.jpg')
print('jpg', jpg)
bmp = cv2.imread('../image/same.bmp')
print('bmp', bmp)