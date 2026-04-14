import cv2
import numpy as np
import imutils
from imutils import contours
import joblib

def test():
    model = joblib.load('svm_model.pkl')
    IMG_SIZE = (20, 20)
    
    img_path = "xemayBigPlate239_jpg.rf.4dae4ddf5de24cfb64b78c65b5b6442a.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print("Ko thay anh")
        return

    img = imutils.resize(img, width=600)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(keypoints)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    
    location = None
    for contour in cnts:
        chu_vi = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * chu_vi, True)
        if len(approx) == 4:
            location = approx
            break

    if location is None:
        cropped_plate = gray
        y1, x1 = 0, 0
    else:
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_plate = gray[x1:x2+1, y1:y2+1]

    blur = cv2.GaussianBlur(cropped_plate, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    thresh[0:3, :] = 0
    thresh[-3:, :] = 0
    thresh[:, 0:3] = 0
    thresh[:, -3:] = 0

    char_cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_chars = []
    plate_height, plate_width = cropped_plate.shape

    for c in char_cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = w * h
        if (plate_height * 0.35 < h < plate_height * 0.9) and (area > 100) and (0.1 < aspect_ratio < 1.2): 
            valid_chars.append(c)

    if not valid_chars:
        print("KHONG CAT DUOC CHU!")
        return

    valid_chars = contours.sort_contours(valid_chars, method="left-to-right")[0]

    bien_so_doc_duoc = ""
    for idx, c in enumerate(valid_chars):
        (x, y, w, h) = cv2.boundingRect(c)
        
        if w < 5 or h < 15:
            continue
            
        roi = thresh[y:y+h, x:x+w]
        
        border = 4 
        roi_padded = cv2.copyMakeBorder(roi, border, border, border, border, cv2.BORDER_CONSTANT, value=0)
        
        roi_resized = cv2.resize(roi_padded, IMG_SIZE)
        cv2.imwrite(f"char_{idx}.jpg", roi_resized)
        
        features = roi_resized.flatten().reshape(1, -1)
        prediction = model.predict(features)
        
        print(f"Char {idx}: at x={x}, y={y}, w={w}, h={h} -> pred: {prediction[0]}")
        bien_so_doc_duoc += prediction[0]

    print(f"Final output: {bien_so_doc_duoc}")

test()
