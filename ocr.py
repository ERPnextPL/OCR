from pdf2image import convert_from_path
import pytesseract
import numpy as np
import cv2
import os

pdf_path = r"C:\internal_projects\OCR\polisa_pzu.pdf"
poppler_path = r"C:\Program Files\poppler-25.07.0\Library\bin"
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Jan\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"Nie znaleziono pliku: {pdf_path}")

# Konwertuj PDF -> lista obrazów PIL
pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path,first_page=1, last_page=1)

for i, page in enumerate(pages):
    # Zamień obraz PIL na format OpenCV
    image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

    # OCR
    text = pytesseract.image_to_string(image, lang='pol')
    print(f"\n=== Strona {i+1} ===\n{text}")

    # Opcjonalnie wyświetl obraz
    cv2.imshow(f"Strona {i+1}", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()






from pdf2image import convert_from_path
import pytesseract
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


pdf_path = r"C:\internal_projects\OCR\polisa_pzu.pdf"
poppler_path = r"C:\Program Files\poppler-25.07.0\Library\bin"
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Jan\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"Nie znaleziono pliku: {pdf_path}")

# Konwertuj PDF -> lista obrazów PIL
pages = convert_from_path(pdf_path, dpi=200, poppler_path=poppler_path,first_page=1, last_page=1)
  



for i, page in enumerate(pages):
    # Zamień obraz PIL na format OpenCV
    image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2GRAY)
    # image = images[0:500, 0:1200] # y1:y2, x1:x2
    regions = {
    "numer polisy": (70, 360, 910, 970),
    "Leasingobiorca": (1060, 1380, 1390, 1220),
    "Firma": (110, 140, 235, 255),
    # "Inne": (400, 450, 1400, 1500),
  }
    
    print("\n=== WYNIKI OCR Z WYBRANYCH OBSZARÓW ===")
    for name, (y1, y2, x1, x2) in regions.items():
        roi = image[y1:y2, x1:x2]
        # texty = pytesseract.image_to_string(roi, lang='pol')
        # print(f"{name}: {text.strip()}")
        _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, lang='pol')
        print(f"=== {name} ===")
        print(f"{name}: {text.strip()}")

        # OCR
        # text = pytesseract.image_to_string(image, lang='pol')
        # print(f"\n=== Strona {i+1} ===\n{text}")

        # Opcjonalnie wyświetl obraz
        # ax.axis('on')
        # cv2.imshow(f"Strona {i+1}", image)
        cv2.waitKey(0)


# plt.show(image)



def show_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Kliknięto w punkt: x={x}, y={y}")
scale = 0.5  # 50% oryginalnego rozmiaru
resized = cv2.resize(image, None, fx=scale, fy=scale)
cv2.namedWindow("Strona")
cv2.setMouseCallback("Strona", show_coordinates)
# cv2.imshow("Strona", image)

cv2.imshow("Strona", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
