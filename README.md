# 🚗 License Plate Recognition using OpenCV, EasyOCR, and Tesseract

This project performs automatic license plate recognition (ALPR) from vehicle images using Python, OpenCV, and OCR engines (EasyOCR or Tesseract).

## 🧰 Requirements

Install the required packages with the following commands:

```bash
pip install easyocr opencv-python-headless pytesseract imutils matplotlib
sudo apt install tesseract-ocr  # Only for Tesseract OCR
```

## 📂 How It Works

1. **Image Preprocessing:** Applies grayscale conversion, filtering, and edge detection.
2. **Contour Detection:** Identifies potential license plate areas via shape approximations.
3. **Plate Cropping:** Extracts and enhances the region containing the plate.
4. **OCR:** Uses either EasyOCR or Tesseract to read the text from the plate.

## 🧪 Usage in Google Colab

![Image](https://github.com/user-attachments/assets/01493ca6-4320-4c19-9ed0-3edbaefd2713)


```python
from google.colab import files
uploaded = files.upload()
```

Then call the function:

```python
image_path = list(uploaded.keys())[0]
license_plate_number = recognize_license_plate(image_path, reader='easyocr', enhance_ocr=True)

if license_plate_number:
    print(f"License Plate Number: {license_plate_number}")
else:
    print("License plate not found in the image.")
```

## 📸 Visualization

After recognition, the image is displayed using `matplotlib` to visually confirm detection.

## 🛠 Function Overview

```python
def recognize_license_plate(image_path, reader='easyocr', enhance_ocr=True):
    """
    Recognizes license plate from an image.
    - image_path: Path to vehicle image.
    - reader: Choose 'easyocr' or 'tesseract'.
    - enhance_ocr: Apply image enhancement to improve OCR.
    """
```

## ⚙️ Notes

* EasyOCR is more robust in low-quality or skewed images.
* Tesseract may require more tuning (`--psm`, `--oem`) for optimal results.
* This is a simple proof-of-concept and may not handle all real-world cases perfectly.
