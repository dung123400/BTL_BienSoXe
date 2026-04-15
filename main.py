import cv2
import numpy as np
import imutils
from imutils import contours
import joblib
import sys

sys.stdout.reconfigure(encoding='utf-8')

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
    # Bỏ GaussianBlur để giữ độ nét mỏng của các chữ số như '7' hoặc '1', tránh bị đứt nét
    _, thresh = cv2.threshold(cropped_plate, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 🌟 MẸO PRO: KỸ THUẬT XÓA VIỀN (CLEAR BORDER)
    # Tô màu đen (0) cho 4 mép của ảnh dày 3 pixel để cắt đứt các viền rác bám vào chữ
    thresh[0:3, :] = 0
    thresh[-3:, :] = 0
    thresh[:, 0:3] = 0
    thresh[:, -3:] = 0

    # Dùng RETR_LIST để đảm bảo tìm được chữ ngay cả khi vùng bị nhiễu nền hoặc chưa cắt sát khung viền
    char_cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_chars = []
    plate_height, plate_width = cropped_plate.shape

    # 🌟 LƯỚI LỌC DÀNH CHO BIỂN SỐ XE MÁY 2 DÒNG
    for c in char_cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = w * h
        
        # Biển 2 dòng / Không cắt trúng viền: chữ chỉ cao từ 8% - 55% của ảnh. Biển 1 dòng: chữ cao đến 90%.
        # Tỷ lệ bề ngang/dọc (aspect_ratio) phải < 0.85 (Chữ/số thường ốm, chỉ có méo/nhiễu/cục to mới tạo hình vuông > 0.85)
        if (plate_height * 0.08 <= h <= plate_height * 0.90) and (area > 50) and (0.1 < aspect_ratio <= 0.85): 
            valid_chars.append(c)

    # Lọc bỏ các contour ảo nằm gọn bên trong một contour khác (ví dụ: 2 lỗ rỗng bên trong số 8)
    final_valid_chars = []
    for i, cA in enumerate(valid_chars):
        xA, yA, wA, hA = cv2.boundingRect(cA)
        is_contained = False
        for j, cB in enumerate(valid_chars):
            if i == j: continue
            xB, yB, wB, hB = cv2.boundingRect(cB)
            # Kiểm tra xem box A có nằm hoàn toàn trong box B không
            if xB <= xA and yB <= yA and (xB + wB) >= (xA + wA) and (yB + hB) >= (yA + hA):
                is_contained = True
                break
        if not is_contained:
            final_valid_chars.append(cA)
    valid_chars = final_valid_chars

    if not valid_chars:
        print("Không cắt được chữ nào chuẩn trên biển số!")
        cv2.imshow("Thu xem may tinh thay gi", thresh)
        cv2.waitKey(0)
        return

    # --- SẮP XẾP CHỮ CÁI CHO BIỂN SỐ 2 DÒNG TỪ TRÊN XUỐNG DƯỚI, RỒI CẮT TRÁI QUA PHẢI ---
    # Phân loại line1 và line2 dựa trên thuật toán Gom Cụm (Row Clustering)
    sorted_by_y = sorted(valid_chars, key=lambda c: cv2.boundingRect(c)[1])
    line1, line2 = [], []
    
    if len(sorted_by_y) > 0:
        rows = []
        current_row = [sorted_by_y[0]]
        for i in range(1, len(sorted_by_y)):
            c = sorted_by_y[i]
            y_i = cv2.boundingRect(c)[1]
            y_prev = cv2.boundingRect(current_row[0])[1]  # So sánh với phần tử đầu dòng
            h_prev = cv2.boundingRect(current_row[0])[3]
            
            # Nếu chênh lệch Y nhỏ hơn 60% chiều cao chữ -> cùng 1 dòng
            if abs(y_i - y_prev) < h_prev * 0.6:
                current_row.append(c)
            else:
                rows.append(current_row)
                current_row = [c]
        rows.append(current_row)

        # Trích xuất tối đa 2 dòng hợp lệ (bỏ qua nhiễu là các dòng chỉ có 1 cục rác)
        rows_len = sorted(rows, key=lambda r: len(r), reverse=True)
        if len(rows_len) >= 2 and len(rows_len[1]) > 1:
            # Lấy 2 dòng dài nhất và sắp xếp lại trên/dưới
            r1, r2 = rows_len[0], rows_len[1]
            if cv2.boundingRect(r1[0])[1] > cv2.boundingRect(r2[0])[1]:
                r1, r2 = r2, r1
            line1, line2 = r1, r2
        else:
            # Biển 1 dòng
            line1 = rows_len[0]
            line2 = []

    # 3. Sắp xếp lại mỗi dòng từ TRÁI sang PHẢI hoàn chỉnh
    line1 = sorted(line1, key=lambda c: cv2.boundingRect(c)[0])
    line2 = sorted(line2, key=lambda c: cv2.boundingRect(c)[0])
    
    # Gắn thẻ nhãn để lúc vòng lặp chạy biết đang ở line nào
    valid_chars_sorted = [(c, True) for c in line1] + [(c, False) for c in line2]

    # --- PHẦN 3: ĐƯA VÀO AI ĐỂ NHẬN DIỆN (BẢO TOÀN CÔNG NGHỆ) ---
    bien_so_doc_duoc = ""
    line1_pred = ""
    line2_pred = ""
    
    # Khai báo HOG giống hệt lúc Train
    hog = cv2.HOGDescriptor((20, 20), (10, 10), (5, 5), (5, 5), 9)
    
    for c_tuple in valid_chars_sorted:
        c, is_line1 = c_tuple
        (x, y, w, h) = cv2.boundingRect(c)
        
        # Cắt sát mép chữ cái
        roi = thresh[y:y+h, x:x+w]
        
        # 🌟 MẸO KỸ THUẬT: ÉP KHUNG THÀNH HÌNH VUÔNG TRƯỚC KHI THÊM LỀ
        diff = abs(h - w)
        if h > w:
            top, bottom = 0, 0
            left, right = diff // 2, diff - diff // 2
        else:
            top, bottom = diff // 2, diff - diff // 2
            left, right = 0, 0
        roi_squared = cv2.copyMakeBorder(roi, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        
        border = 4 
        roi_padded = cv2.copyMakeBorder(roi_squared, border, border, border, border, cv2.BORDER_CONSTANT, value=0)
        roi_resized = cv2.resize(roi_padded, IMG_SIZE)
        
        # 🌟 Sử dụng HOG trích xuất xương sống của con chữ và đưa vào SVM
        features = hog.compute(roi_resized).flatten().reshape(1, -1)
        prediction = model.predict(features)[0]
        
        # 🌟 BỘ LỌC HẬU KỲ (POST-PROCESSING) - LUẬT BIỂN SỐ VIỆT NAM (Sửa lỗi cho SVM)
        is_1_line_plate = (len(line2) == 0)
        
        if is_1_line_plate:
            # LUẬT BIỂN Ô TÔ 1 DÒNG (Dài max 8-9 ký tự, VD: 51H 91991)
            char_idx = len(line1_pred)
            if char_idx >= 8: continue # Xóa ốc vít
            
            if char_idx in [0, 1]: 
                if prediction == 'B': prediction = '8'
                elif prediction == 'G': prediction = '6'
                elif prediction == 'S': prediction = '9'
                elif prediction == 'Z': prediction = '2'
                elif prediction in ['D', 'O']: prediction = '0'
                elif prediction == 'A': prediction = '4'
                elif prediction in ['T', 'I']: prediction = '1'
            elif char_idx == 2:
                if prediction == '8': prediction = 'B'
                elif prediction == '0': prediction = 'D'
                elif prediction == '1': prediction = 'T'
                elif prediction == '9': prediction = 'S'
                elif prediction == '2': prediction = 'Z'
                elif prediction == '6': prediction = 'G'
                elif prediction == '4': prediction = 'A'
            elif char_idx >= 3:
                # Toàn bộ phần sau là số
                if prediction == 'B': prediction = '8'
                elif prediction == 'G': prediction = '6'
                elif prediction == 'Z': prediction = '2'
                elif prediction == 'S': prediction = '9'
                elif prediction == 'A': prediction = '4'
                elif prediction in ['D', 'O']: prediction = '0'
                elif prediction in ['T', 'I']: prediction = '1'
                
            line1_pred += prediction
            bien_so_doc_duoc += prediction

        else:
            # LUẬT BIỂN 2 DÒNG
            if is_line1:
                char_idx = len(line1_pred)
                is_oto_2_dong = len(line1) <= 3
                max_chars = 3 if is_oto_2_dong else 5

                if char_idx >= max_chars: continue
                    
                if char_idx in [0, 1]:
                    if prediction == 'B': prediction = '8'
                    elif prediction == 'G': prediction = '6'
                    elif prediction == 'S': prediction = '9'
                    elif prediction == 'Z': prediction = '2'
                    elif prediction in ['D', 'O']: prediction = '0'
                    elif prediction == 'A': prediction = '4'
                    elif prediction in ['T', 'I']: prediction = '1'
                elif char_idx == 2:
                    if prediction == '8': prediction = 'B'
                    elif prediction == '0': prediction = 'D'
                    elif prediction == '1': prediction = 'T'
                    elif prediction == '9': prediction = 'S'
                    elif prediction == '2': prediction = 'Z'
                    elif prediction == '6': prediction = 'G'
                    elif prediction == '4': prediction = 'A'
                # char_idx = 3 có thể là Số (Xe máy) hoặc Chữ (Xe máy) nên giữ nguyên
                
                line1_pred += prediction
                bien_so_doc_duoc += prediction
                
            else:
                char_idx = len(line2_pred)
                if char_idx >= 5: continue
                    
                # Dòng dưới toàn bộ CẦN LÀ SỐ
                if prediction == 'B': prediction = '8'
                elif prediction == 'G': prediction = '6'
                elif prediction == 'Z': prediction = '2'
                elif prediction == 'S': prediction = '9'
                elif prediction == 'A': prediction = '4'
                elif prediction in ['D', 'O']: prediction = '0'
                elif prediction in ['T', 'I']: prediction = '1'
                
                line2_pred += prediction
                bien_so_doc_duoc += prediction

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
doc_bien_so("images156.jpg")
