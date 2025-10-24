from pdf2image import convert_from_path
import pytesseract
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


# pdf_path = r"C:\internal_projects\OCR\polisa_pzu.pdf"
pdf_path = r"polisa_pzu.pdf"
poppler_path = r"C:\Program Files\poppler-25.07.0\Library\bin"
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Jan\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"Nie znaleziono pliku: {pdf_path}")

# Konwertuj PDF -> lista obrazów PIL
pages = convert_from_path(pdf_path, dpi=200, poppler_path=poppler_path,first_page=1, last_page=1)
  
numer_polisy = None


for page in pages:
    # Zamień obraz PIL na format OpenCV
    image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2GRAY)
    # image = images[0:500, 0:1200] # y1:y2, x1:x2
    regions = {
        "Numer polisy": (915, 950, 225, 360),
        "Firma ubez.": (220, 265, 75, 171),
        "Imię": (920, 948, 628, 819)
  }
    preview = image.copy()
    for name, (y1, y2, x1, x2) in regions.items():
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(preview, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Podgląd obszarów OCR", preview)
    cv2.waitKey(0)
    print("\n=== WYNIKI OCR Z WYBRANYCH OBSZARÓW ===")

    for name, (y1, y2, x1, x2) in regions.items():
        roi = image[y1:y2, x1:x2]

        # Przetwarzanie wstępne (poprawa jakości OCR)
        # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)

        # OCR
        text = pytesseract.image_to_string(thresh, lang='pol')
        print(f"{name}: {text.strip()}")

        if name == "Numer polisy":
            numer_polisy = text
        # Pokaż wycięty fragment
        # cv2.imshow(name, roi)
        # cv2.waitKey(0)

cv2.destroyAllWindows()


print(f"\nOdczytany numer polisy: {numer_polisy}")
# def show_coordinates(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(f"Kliknięto w punkt: x={x}, y={y}")
# scale = 0.5  # 50% oryginalnego rozmiaru
# resized = cv2.resize(image, None, fx=scale, fy=scale)
# cv2.namedWindow("Strona")
# cv2.setMouseCallback("Strona", show_coordinates)
# # cv2.imshow("Strona", image)

# cv2.imshow("Strona", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
