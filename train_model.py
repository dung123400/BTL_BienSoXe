import os
import cv2
import numpy as np
from sklearn.svm import SVC
import joblib
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 1. Cấu hình đường dẫn và kích thước ảnh
DATASET_PATH = "dataset" # Tên thư mục chứa ảnh của bạn
IMG_SIZE = (20, 20)      # Kích thước chuẩn để AI học (rộng 20, cao 20)

# Khởi tạo 2 mảng để chứa dữ liệu
X = [] # X: Chứa đặc trưng của ảnh (chính là ma trận pixel)
y = [] # y: Chứa nhãn (tên của chữ cái/số đó)

# Khởi tạo thuật toán HOG cực kỳ mạnh mẽ để bắt nét cong/chéo thay vì bắt điểm ảnh
hog = cv2.HOGDescriptor((20, 20), (10, 10), (5, 5), (5, 5), 9)

print(f"Đang đọc dữ liệu từ thư mục '{DATASET_PATH}'...")

# 2. Đọc ảnh từ thư mục
# os.listdir() sẽ lấy ra danh sách các thư mục con (0, 1, 2... A, B, C...)
for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)
    
    # Bỏ qua nếu không phải là thư mục
    if not os.path.isdir(label_path):
        continue
        
    print(f"Đang xử lý thư mục ký tự: {label}")
    
    for img_file in os.listdir(label_path):
        img_path = os.path.join(label_path, img_file)
        
        # Đọc ảnh ở chế độ XÁM (Grayscale)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Bỏ qua nếu ảnh bị lỗi không đọc được
        if img is None:
            continue

        # Ép ảnh dataset thành Chữ Trắng - Nền Đen tuyệt đối
        _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 🌟 TƯƠNG TỰ NHƯ KHI NHẬN DIỆN: Ép ảnh học thành hình VUÔNG để không méo chữ
        coords = cv2.findNonZero(img)
        if coords is not None:
            bx, by, bw, bh = cv2.boundingRect(coords)
            img_roi = img[by:by+bh, bx:bx+bw]
            
            diff = abs(bh - bw)
            if bh > bw:
                top, bottom = 0, 0
                left, right = diff // 2, diff - diff // 2
            else:
                top, bottom = diff // 2, diff - diff // 2
                left, right = 0, 0
                
            img_squared = cv2.copyMakeBorder(img_roi, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            
            border = 4
            img_padded = cv2.copyMakeBorder(img_squared, border, border, border, border, cv2.BORDER_CONSTANT, value=0)
            img = img_padded
            
        # Resize ảnh về đúng kích thước 20x20
        img = cv2.resize(img, IMG_SIZE)
        
        # SỬ DỤNG SỨC MẠNH HOG (Bắt dáng nét các góc độ của chữ cái/số)
        features = hog.compute(img).flatten() 
        
        # Đưa dữ liệu vào danh sách
        X.append(features)
        y.append(label)

# Chuyển đổi list sang numpy array để SVM xử lý nhanh hơn
X = np.array(X)
y = np.array(y)

print(f"\nTổng số ảnh đã đọc: {len(X)}")
print("Bắt đầu huấn luyện mô hình SVM (Quá trình này có thể mất 1-3 phút)...")

# 3. Tạo và huấn luyện mô hình SVM (Thuật toán Chương 5)
# Sử dụng kernel='linear' vì bài toán phân loại chữ cái này có thể phân tách tuyến tính
model = SVC(kernel='linear', probability=True, random_state=42)

# Cho máy học
model.fit(X, y)

print("Huấn luyện thành công!")

# 4. Lưu lại 'bộ não' đã học
model_filename = 'svm_model.pkl'
joblib.dump(model, model_filename)
print(f"Đã lưu mô hình thành công vào file: {model_filename}")
print("Bây giờ bạn có thể dùng file này để nhận diện ký tự!")