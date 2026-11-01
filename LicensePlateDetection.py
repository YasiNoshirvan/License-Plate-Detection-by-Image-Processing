
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import imutils
import easyocr
from pathlib import Path


# Change these two settings to test another car or language.
plate_language = 'fa'  # 'fa' for Iranian plates, 'en' for English plates
image_name = 'car_plate3.jpg'  # Example: 'car_plate2.jpg'

project_dir = Path(__file__).resolve().parent
image_path = Path('D:/Training Image Processing/image') / image_name
output_dir = project_dir / 'results'
output_dir.mkdir(exist_ok=True)

car_img = cv.imread(str(image_path))
if car_img is None:
    raise FileNotFoundError(f'Image not found: {image_path}')

car_img_gray = cv.cvtColor(car_img, cv.COLOR_BGR2GRAY)

# Noise removal and edge detection
bilateral_filtered = cv.bilateralFilter(car_img_gray, 11, 15, 15)
edges = cv.Canny(bilateral_filtered, 30, 200)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(car_img, cv.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')

# Find a possible license plate among rectangular contours
contours = cv.findContours(edges.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours_refind = imutils.grab_contours(contours)
contours_sorted = sorted(contours_refind, key=cv.contourArea, reverse=True)[:30]

plate_location = None
for contour in contours_sorted:
    perimeter = cv.arcLength(contour, True)
    contour_approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(contour_approx) == 4:
        x, y, w, h = cv.boundingRect(contour_approx)
        aspect_ratio = w / h

        if 2.5 < aspect_ratio < 6:
            plate_location = contour_approx
            break

if plate_location is None:
    print('License plate not detected. Try another image or adjust the contour settings.')
else:
    # Keep the original mask-and-crop approach
    plate_mask0 = np.zeros(car_img_gray.shape, np.uint8)
    plate_mask = cv.drawContours(plate_mask0, [plate_location], 0, 255, -1)
    plate_img = cv.bitwise_and(car_img, car_img, mask=plate_mask)

    # np.where returns (row, column) = (y, x)
    rows, cols = np.where(plate_mask == 255)
    y1, y2 = np.min(rows), np.max(rows)
    x1, x2 = np.min(cols), np.max(cols)

    plate_width = x2 - x1 + 1
    plate_height = y2 - y1 + 1

    # Iranian plates often need extra space on the right for the province section
    if plate_language == 'fa':
        pad_left = int(plate_width * 0.20)
        pad_right = int(plate_width * 0.40)
    else:
        pad_left = int(plate_width * 0.05)
        pad_right = int(plate_width * 0.05)

    y1 = max(0, y1 - int(plate_height * 0.20))
    y2 = min(car_img_gray.shape[0] - 1, y2 + int(plate_height * 0.20))
    x1 = max(0, x1 - pad_left)
    x2 = min(car_img_gray.shape[1] - 1, x2 + pad_right)

    cropped_image = car_img_gray[y1:y2 + 1, x1:x2 + 1]

    # Increase size and contrast to help OCR on small plates
    bigger_plate = cv.resize(cropped_image, None, fx=3, fy=3,
                             interpolation=cv.INTER_CUBIC)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    ocr_image = clahe.apply(bigger_plate)

    languages = ['fa', 'en'] if plate_language == 'fa' else ['en']
    reader = easyocr.Reader(languages, gpu=False)
    plate_info = reader.readtext(ocr_image)

    valid_results = [item for item in plate_info if item[2] >= 0.4]

    if valid_results:
        plate_text = ' '.join(item[1] for item in valid_results)
        print('Detected plate text:', plate_text)

        for _, text, confidence in valid_results:
            print(f'{text} (confidence: {confidence:.2f})')
    else:
        plate_text = ''
        print('Plate detected')


    # Draw detection on original image
    result_img = car_img.copy()
    cv.polylines(result_img, [plate_location], True, (0, 255, 0), 2)
    cv.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv.imwrite(str(output_dir / 'plate_detected.jpg'), result_img)
    cv.imwrite(str(output_dir / 'plate_crop.jpg'), cropped_image)
    cv.imwrite(str(output_dir / 'plate_ocr_input.jpg'), ocr_image)
    (output_dir / 'recognized_text.txt').write_text(plate_text, encoding='utf-8')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(result_img, cv.COLOR_BGR2RGB))
    plt.title('Detected License Plate')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(ocr_image, cmap='gray')
    plt.title('Cropped Plate for OCR')
    plt.axis('off')

plt.tight_layout()
plt.show()
