from ocr import NepaliOCR
import cv2

nepali = NepaliOCR('weights/nepali_test.pth')
img = cv2.imread('fuck.png')

print(nepali.predict(img))
