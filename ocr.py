#test OCR
import cv2
import matplotlib as plt
img = cv2.imread('example.jpg') # wczytanie obrazka
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # konwersja na czarno białe
thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1] # wszystkie wartości powyżej 127 zamieniane na 255(biały)
plt.imshow(out, 'gray')
plt.show() # wyświetlanie obrazka

from pytesseract import image_to_string
output = image_to_string(thresh, lang='eng', config='--psm 7')
print('Output: ', output)