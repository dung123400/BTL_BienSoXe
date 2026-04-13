import cv2
import numpy as np
import imutils
from imutils import contours
import joblib

# 1. TẢI "BỘ NÃO" SVM ĐÃ HUẤN LUYỆN LÊN
print("Đang tải mô hình AI...")
model = joblib.load('svm_model.pkl')
IMG_SIZE = (20, 20) # Phải giống hệt kích thước lúc Train

# 2. HÀM XỬ LÝ ẢNH ĐỂ ĐỌC BIỂN SỐ
def doc_bien_so(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Không tìm thấy ảnh!")
        return

    img = imutils.resize(img, width=600)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- PHẦN 1: TÌM KHUNG BIỂN SỐ ---
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

    # --- CƠ CHẾ BẢO HIỂM MỚI ---
    if location is None:
        print("Cảnh báo: Không tìm thấy khung viền! Tự động giả định toàn bộ ảnh là biển số.")
        cropped_plate = gray # Lấy luôn toàn bộ ảnh xám làm biển số
        y1, x1 = 0, 0 # Gán tạm tọa độ để tý nữa không bị lỗi
    else:
        print("Đã khoanh vùng được biển số!")
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_plate = gray[x1:x2+1, y1:y2+1]

    # --- PHẦN 2 (BẢN CHỐT): CẮT TỪNG CHỮ CÁI TRÊN BIỂN SỐ ĐÃ CẮT ---
    blur = cv2.GaussianBlur(cropped_plate, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 🌟 MẸO PRO: KỸ THUẬT XÓA VIỀN (CLEAR BORDER)
    # Tô màu đen (0) cho 4 mép của ảnh dày 3 pixel để cắt đứt các viền rác bám vào chữ
    thresh[0:3, :] = 0
    thresh[-3:, :] = 0
    thresh[:, 0:3] = 0
    thresh[:, -3:] = 0

    char_cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_chars = []
    plate_height, plate_width = cropped_plate.shape

    # 🌟 LƯỚI LỌC SIÊU CẤP (Chỉ bắt chữ, loại bỏ dấu chấm, gạch ngang, ốc vít)
    for c in char_cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = w * h
        
        # ĐIỀU KIỆN 1: Chiều cao (h) phải chiếm từ 35% đến 90% biển số
        # -> Dấu chấm, gạch ngang, ốc vít lùn tịt sẽ bị vứt ngay lập tức!
        # ĐIỀU KIỆN 2: Không lấy những cục rác quá nhỏ (area > 100)
        # ĐIỀU KIỆN 3: Tỷ lệ bề ngang/dọc (aspect_ratio) từ 0.1 đến 1.2
        if (plate_height * 0.35 < h < plate_height * 0.9) and (area > 100) and (0.1 < aspect_ratio < 1.2): 
            valid_chars.append(c)

    if not valid_chars:
        print("Không cắt được chữ nào chuẩn trên biển số!")
        cv2.imshow("Thu xem may tinh thay gi", thresh)
        cv2.waitKey(0)
        return

    # Sắp xếp các ký tự từ TRÁI sang PHẢI
    valid_chars = contours.sort_contours(valid_chars, method="left-to-right")[0]

    # --- PHẦN 3 (Đã Fix chuẩn): ĐƯA VÀO AI ĐỂ NHẬN DIỆN ---
    bien_so_doc_duoc = ""
    for c in valid_chars:
        (x, y, w, h) = cv2.boundingRect(c)
        
        # Bỏ qua các chấm nhiễu nhỏ
        if w < 5 or h < 15:
            continue
            
        # Cắt sát mép chữ cái
        roi = thresh[y:y+h, x:x+w]
        
        # Thêm lề (padding) để chữ không bị méo. Chú ý: value=0 (Màu đen)
        border = 4 
        roi_padded = cv2.copyMakeBorder(roi, border, border, border, border, cv2.BORDER_CONSTANT, value=0)
        
        # Lúc này roi_padded đang là Chữ Trắng - Nền Đen, khớp 100% với AI mới học
        roi_resized = cv2.resize(roi_padded, IMG_SIZE)
        
        # Duỗi thẳng và dự đoán
        features = roi_resized.flatten().reshape(1, -1)
        prediction = model.predict(features)
        
        bien_so_doc_duoc += prediction[0]
        cv2.rectangle(img, (y1+x, x1+y), (y1+x+w, x1+y+h), (0, 255, 0), 2)

    print(f"\n=============================")
    print(f"KẾT QUẢ: {bien_so_doc_duoc}")
    print(f"=============================\n")

    # Hiển thị ảnh
    cv2.imshow("Bien so da cat", cropped_plate)
    cv2.imshow("Anh nhin tu AI", thresh)
    cv2.imshow("Ket qua", img)
    cv2.waitKey(0)

# --- CHẠY THỬ ---
# Nhớ đổi tên file ảnh dưới đây thành ảnh của bạn nhé!
doc_bien_so("xemayBigPlate239_jpg.rf.4dae4ddf5de24cfb64b78c65b5b6442a.jpg")