import cv2
import numpy as np
import os
import shutil
from typing import List, Tuple, Dict
from pytesseract import Output
import pytesseract


def _resolve_tesseract_path() -> str:
    """Locate the Tesseract executable and configure pytesseract accordingly."""
    # Prefer the command already configured on the pytesseract module.
    configured_cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
    resolved_cmd = shutil.which(configured_cmd)
    if resolved_cmd:
        pytesseract.pytesseract.tesseract_cmd = resolved_cmd
        return resolved_cmd

    candidates = [
        os.environ.get("TESSERACT_CMD"),
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]

    for candidate in candidates:
        if not candidate:
            continue
        # Allow users to use environment variables or relative paths
        expanded = os.path.expanduser(os.path.expandvars(candidate))
        if os.path.isfile(expanded):
            pytesseract.pytesseract.tesseract_cmd = expanded
            return expanded
        resolved = shutil.which(expanded)
        if resolved:
            pytesseract.pytesseract.tesseract_cmd = resolved
            return resolved

    raise RuntimeError(
        "Tesseract OCR executable not found. Install it from "
        "https://github.com/UB-Mannheim/tesseract/wiki and/or set the "
        "TESSERACT_CMD environment variable to point to tesseract.exe."
    )


class DocumentRegionDetector:
    """Detector for identifying distinct data regions in structured documents."""
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.original = None
        self.processed = None
        self.regions = []
        
    def load_image(self) -> np.ndarray:
        """Load and validate input image."""
        self.original = cv2.imread(self.image_path)
        if self.original is None:
            raise ValueError(f"Cannot load image from {self.image_path}")
        return self.original
    
    def preprocess_image(self) -> np.ndarray:
        """Apply preprocessing pipeline for optimal region detection."""
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Highlight the rectangular borders and structural edges of the document.
        edges = cv2.Canny(blur, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # Expand and connect edges so that individual boxes become solid blobs.
        dilated = cv2.dilate(edges, kernel, iterations=1)
        self.processed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

        return self.processed
    
    def detect_regions(
        self,
        min_area: int = 1000,
        max_area_ratio: float = 0.3,
        padding: int = 5
    ) -> List[Dict]:
        """Find contours and extract bounding boxes for data regions."""
        if self.processed is None:
            raise ValueError("Call preprocess_image() before detect_regions().")

        contours, hierarchy = cv2.findContours(
            self.processed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        image_height, image_width = self.processed.shape[:2]
        image_area = image_height * image_width

        regions = []
        for idx, contour in enumerate(contours):
            contour_area = cv2.contourArea(contour)

            # Reject noise that is too small to represent a section.
            if contour_area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            bounding_area = w * h

            if max_area_ratio is not None and bounding_area > image_area * max_area_ratio:
                continue

            # Expand the bounding box slightly to make sure we cover the whole block.
            x1 = max(x - padding, 0)
            y1 = max(y - padding, 0)
            x2 = min(x + w + padding, image_width)
            y2 = min(y + h + padding, image_height)
            padded_w = x2 - x1
            padded_h = y2 - y1

            width_ratio = padded_w / float(image_width)
            height_ratio = padded_h / float(image_height)
            if width_ratio < 0.05 or height_ratio < 0.05:
                continue

            regions.append({
                'id': idx,
                'bbox': (x1, y1, padded_w, padded_h),
                'area': contour_area,
                'center': (x1 + padded_w // 2, y1 + padded_h // 2)
            })

        # Sort regions by vertical position (top to bottom), then horizontal.
        regions.sort(key=lambda r: (round(r['bbox'][1] / 10) * 10, r['bbox'][0]))

        self.regions = regions
        return regions
    
    # def assign_labels(self) -> List[Dict]:
    #     """Assign semantic labels to detected regions based on position."""
    #     labels = [
    #         "Logo",
    #         "Samochod",
    #         "Malgorzata",
    #         "Dane polisy",
    #         "Ubezpieczajacy", 
    #         "Wlasciciel pojazdu",
    #         "Ubezpieczony pojazd",
    #         "Leasingodawca",
    #         "Leasingobiorca",
    #         "Platnosci"
    #     ]
        
    #     for idx, region in enumerate(self.regions):
    #         if idx < len(labels):
    #             region['label'] = labels[idx]
    #         else:
    #             region['label'] = f"Region {idx + 1}"
                
    #     return self.regions
    
    def visualize_regions(self, show_labels=True, show_window=True):
        output = self.original.copy()
        for region in self.regions:
            x, y, w, h = region['bbox']
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
            if show_labels and 'label' in region:
                cv2.putText(output, region['label'], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if show_window:
            cv2.imshow("Detected Regions", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return output

    
    def extract_region_images(self) -> List[Tuple[str, np.ndarray]]:
        """Extract individual region images from original."""
        extracted = []
        
        for region in self.regions:
            x, y, w, h = region['bbox']
            roi = self.original[y:y+h, x:x+w]
            label = region.get('label', f"region_{region['id']}")
            extracted.append((label, roi))
            
        return extracted
    
    def save_regions(self, output_dir: str = "./regions"):
        """Save individual region images to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        for region in self.regions:
            x, y, w, h = region['bbox']
            roi = self.original[y:y+h, x:x+w]
            label = region.get('label', f"region_{region['id']}")
            
            # Clean filename
            filename = label.replace(' ', '_').lower()
            filepath = os.path.join(output_dir, f"{filename}.png")
            cv2.imwrite(filepath, roi)
            
        print(f"Saved {len(self.regions)} regions to {output_dir}")
    
    def print_region_info(self):
        """Display information about detected regions."""
        print(f"\nDetected {len(self.regions)} regions:")
        print("-" * 60)
        
        for region in self.regions:
            x, y, w, h = region['bbox']
            label = region.get('label', 'Unknown')
            print(f"{label:25} | Position: ({x:4}, {y:4}) | Size: {w:4}x{h:4} | Area: {region['area']:6.0f}")

    def detect_regions_with_ocr(self, lang='pol+eng', conf_threshold=60, only_headers=False, header_keywords=None):
        """
        Detect regions using Tesseract OCR.

        :param lang: Language passed to Tesseract (e.g. 'pol')
        :param conf_threshold: Minimum confidence threshold for recognized text (0-100)
        :param only_headers: When True, keep only regions that match header_keywords
        :param header_keywords: Optional list of header keywords (lowercase)
        """
        if header_keywords is None:
            # Default keywords derived from the sample document structure
            header_keywords = [
                "dane polisy", "ubezpieczajacy", "wlasciciel pojazdu", "ubezpieczony pojazd", "leasingodawca", "leasingobiorca", "platnosci"
            ]
        header_keywords = [h.lower() for h in header_keywords]

        try:
            _resolve_tesseract_path()
        except RuntimeError as exc:
            print("[OCR] {}".format(exc))
            return []

        # Use a grayscale version of the original image for OCR
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        try:
            ocr_result = pytesseract.image_to_data(gray, lang=lang, output_type=Output.DICT)
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "Tesseract OCR executable could not be located. "
                "Install it and/or configure the TESSERACT_CMD environment variable."
            ) from exc

        regions = []
        n_boxes = len(ocr_result['text'])
        for i in range(n_boxes):
            text = ocr_result['text'][i]
            if text is None:
                continue
            text = text.strip().lower()
            if not text:
                continue

            try:
                conf_value = float(ocr_result['conf'][i])
            except (ValueError, TypeError):
                continue
            if conf_value < conf_threshold:
                continue

            if only_headers and text not in header_keywords:
                continue

            x = ocr_result['left'][i]
            y = ocr_result['top'][i]
            w = ocr_result['width'][i]
            h = ocr_result['height'][i]
            regions.append({
                "id": i,
                "bbox": (x, y, w, h),
                "label": text,
                "area": w * h,
                "conf": conf_value
            })

        self.regions = regions
        print("OCR detected {} regions (only_headers={}).".format(len(regions), only_headers))
        return regions

    def visualize_regions_ocr(self, show_window=True):
        """
        Wizualizuje OCR-owe ramki na oryginalnym obrazie.
        """
        output = self.original.copy()
        for region in self.regions:
            x, y, w, h = region['bbox']
            label = region.get('label', '')
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 128, 255), 3)
            cv2.putText(output, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if show_window:
            cv2.imshow("OCR Regions (Nagłówki/Text)", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        cv2.imwrite("ocr_detected_regions.jpg", output)
        return output
 

    # def read_text_from_regions(self, lang='pol', output_file="regions_text.txt") -> List[Dict]:
    #     """
    #     Odczytuje tekst z każdej wykrytej sekcji (regionu) za pomocą OCR,
    #     wypisuje wyniki w terminalu i zapisuje je do pliku tekstowego.
    #     """
    #     try:
    #         _resolve_tesseract_path()
    #     except RuntimeError as exc:
    #         print("[OCR ERROR]", exc)
    #         return []

    #     extracted_texts = []
    #     all_text_output = []

    #     for region in self.regions:
    #         x, y, w, h = region['bbox']
    #         roi = self.original[y:y+h, x:x+w]

    #         # Konwersja na odcienie szarości dla lepszego OCR
    #         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #         text = pytesseract.image_to_string(gray, lang=lang)

    #         text = text.strip()
    #         label = region.get('label', f"region_{region['id']}")

    #         extracted_texts.append({
    #             "label": label,
    #             "text": text
    #         })

    #         # Wypisz w terminalu
    #         print(f"\n=== {label.upper()} ===")
    #         print(text if text else "(brak tekstu wykrytego)")

    #         # Dodaj do bufora zapisu
    #         all_text_output.append(f"=== {label.upper()} ===\n{text if text else '(brak tekstu wykrytego)'}\n")

    #     # Zapisz wszystko do pliku
    #     try:
    #         with open(output_file, "w", encoding="utf-8") as f:
    #             f.write("\n\n".join(all_text_output))
    #         print(f"\n📄 Zapisano wyniki OCR do pliku: {output_file}")
    #     except Exception as e:
    #         print(f"[ERROR] Nie udało się zapisać wyników do pliku: {e}")

    #     return extracted_texts
    def read_text_from_regions_enhanced(self, lang='pol+eng', scale_factor=2.0, output_file="regions_text.txt") -> List[Dict]:
        """
        Odczytuje tekst z każdej wykrytej sekcji w kolejności:
        1. Wycięcie regionu
        2. Powiększenie obrazu
        3. Oczyszczenie i binaryzacja (usuwanie szumów)
        4. OCR z Tesseract
        Wyniki są wypisywane w terminalu i zapisywane do pliku tekstowego.
        """
        try:
            _resolve_tesseract_path()
        except RuntimeError as exc:
            print("[OCR ERROR]", exc)
            return []

        extracted_texts = []
        all_text_output = []

        for region in self.regions:
            x, y, w, h = region['bbox']
            roi = self.original[y:y+h, x:x+w]

            # --- 1️⃣ Powiększenie obrazu ---
            if scale_factor != 1.0:
                roi = cv2.resize(roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

            # --- 2️⃣ Konwersja do szarości ---
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # gray = cv2.equalizeHist(gray1) # additional contrast

            # --- 3️⃣ Usunięcie szumów i poprawa kontrastu ---
            # Filtr Gaussa usuwa drobne zakłócenia
            # denoised = cv2.GaussianBlur(gray, (5, 5), 0)
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            # Binaryzacja adaptacyjna – mocno podbija kontrast tekstu
            cleaned = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            # cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
            # cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)


            # --- 3️⃣.1️⃣ Zapis oczyszczonego regionu do pliku (debug) ---
            label = region.get('label', f"region_{region['id']}")
            cv2.imwrite(f"cleaned\cleaned_{label.replace(' ', '_').lower()}.png", cleaned)

            # --- 4️⃣ OCR ---
            # config = r'-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@._-'
            # text = pytesseract.image_to_string(cleaned, lang=lang, config=config)

            text = pytesseract.image_to_string(cleaned, lang=lang)
            text = text.strip()

            label = region.get('label', f"region_{region['id']}")
            extracted_texts.append({
                "label": label,
                "text": text
            })

            # --- 5️⃣ Logowanie i zapis ---
            print(f"\n=== {label.upper()} ===")
            print(text if text else "(brak tekstu wykrytego)")

            all_text_output.append(f"=== {label.upper()} ===\n{text if text else '(brak tekstu wykrytego)'}\n")

        # Zapis wyników OCR do pliku
        # try:
        #     with open(output_file, "w", encoding="utf-8") as f:
        #         f.write("\n\n".join(all_text_output))
        #     print(f"\n📄 Zapisano wyniki OCR do pliku: {output_file}")
        # except Exception as e:
        #     print(f"[ERROR] Nie udało się zapisać wyników do pliku: {e}")

        return extracted_texts


    def assign_labels_by_keywords(self, lang='pol+eng') -> List[Dict]:
        """
        Przypisuje etykiety do regionów na podstawie słów kluczowych rozpoznanych przez OCR.
        """
        try:
            _resolve_tesseract_path()
        except RuntimeError as exc:
            print("[OCR ERROR]", exc)
            return []

        # 🔹 Słowa kluczowe dla dopasowań
        label_keywords = {
            # "Logo": ["logo"],
            # "Samochod": ["samochód", "pojazd"],
            "Dane polisy": [ "numer polisy", "data polisy"],
            "Ubezpieczajacy": ["ubezpieczający", "ubezpieczenia"],
            "Wlasciciel pojazdu": ["właściciel", "pojazdu", "posiadacz","REGON"],
            "Ubezpieczony pojazd": ["ubezpieczony", "pojazd"],
            "Leasingodawca": ["leasingodawca","Leasingodawca"],
            "Leasingobiorca": ["leasingobiorca"],
            "Platnosci": ["Platności", "składka", "Platnosci", "platnosc", "odbiorca"]
        }

        for region in self.regions:
            x, y, w, h = region['bbox']
            roi = self.original[y:y+h, x:x+w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, lang=lang).lower()

            region['ocr_text'] = text.strip()

            best_label = None
            for label, keywords in label_keywords.items():
                for kw in keywords:
                    if kw in text:
                        best_label = label
                        break
                if best_label:
                    break

            # Jeśli nie znaleziono dopasowania – nadaj nazwę tymczasową
            region['label'] = best_label if best_label else f"Region_{region['id']}"

        print("\n📑 Etykiety przypisane na podstawie słów kluczowych OCR:")
        for region in self.regions:
            print(f" - {region['label']:<20} | Tekst: {region.get('ocr_text', '')[:60]}")

        
        return self.regions


def main():
    """Main execution function."""
    # Initialize detector
    detector = DocumentRegionDetector("data\polisa.jpg")
    
    # Processing pipeline
    print("Loading image...")
    detector.load_image()
    
    print("Preprocessing image...")
    detector.preprocess_image()
    
    print("Detecting regions...")
    detector.detect_regions(min_area=1000)
    
    # print("Assigning labels...")
    # detector.assign_labels()

    print("Przypisywanie etykiet na podstawie słów kluczowych OCR...")
    detector.assign_labels_by_keywords(lang='pol+eng')


    # print("\nOdczytywanie tekstu z wykrytych sekcji...")
    # detector.read_text_from_regions(lang='pol')
    print("\nOdczytywanie tekstu z wykrytych sekcji (ulepszony OCR)...")
    detector.read_text_from_regions_enhanced(lang='pol+eng', scale_factor=2.0)

    # Display results
    detector.print_region_info()

    # Visualize
    result = detector.visualize_regions(show_labels=True)
    # OCR - tylko nagłówki (z polskimi frazami)
    # detector.detect_regions_with_ocr(only_headers=True)
    # detector.visualize_regions_ocr()
    
    # # OCR - wszystkie teksty (bardziej "szczegółowe" ramki)
    # detector.detect_regions_with_ocr(only_headers=False)
    # detector.visualize_regions_ocr()

    print("\nProcessing complete!", result)


if __name__ == "__main__":
    main()
