import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- GIAI ĐOẠN 1: KHỞI TẠO DỮ LIỆU MẪU (DATA LOADING) ---
file_path = 'diabetes_data.csv' 

print(f"[-] Đang tải dữ liệu từ {file_path}...")

# Đọc dữ liệu
df = pd.read_csv(file_path)

# Tách dữ liệu đầu vào (X) và kết quả (y)
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']

# --- GIAI ĐOẠN 2: MODEL TRAINING ---
print("[-] Đang huấn luyện mô hình KNN...")

# Bước quan trọng: Chuẩn hóa dữ liệu để các chỉ số như Insulin (hàng trăm) 
# không lấn át chỉ số như Phả hệ (0.x)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Khởi tạo mô hình KNN tìm 5 người hàng xóm
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_scaled, y)

print("[OK] Mô hình đã sẵn sàng!")

# --- GIAI ĐOẠN 3: GIẢ LẬP NGƯỜI DÙNG (SIMULATION) ---
# TODO: Sau này sẽ thay thế bằng input() hoặc lấy từ API/Web Form.

# Thứ tự: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, Pedigree, Age
# Ví dụ: Một người có chỉ số Glucose cao (148) và BMI cao (33.6)
mock_user_input = [
    6,      # Số lần mang thai
    148,    # Glucose
    72,     # Huyết áp
    35,     # Độ dày da
    0,      # Insulin (Chưa đo được)
    33.6,   # BMI
    0.627,  # Phả hệ
    50      # Tuổi
]

print("\n--- BẮT ĐẦU DỰ ĐOÁN CHO NGƯỜI DÙNG MẪU ---")
print(f"Thông tin đầu vào: {mock_user_input}")

# --- GIAI ĐOẠN 4: XỬ LÝ VÀ DỰ ĐOÁN ---

# 1. Đóng gói dữ liệu vào DataFrame để khớp tên cột
user_df = pd.DataFrame([mock_user_input], columns=X.columns)

# 2. Chuẩn hóa dữ liệu người dùng theo đúng tỷ lệ của dữ liệu gốc
# NOTE: Phải dùng .transform(), KHÔNG ĐƯỢC dùng .fit_transform() ở đây
user_scaled = scaler.transform(user_df)

# 3. Hỏi mô hình
prediction = knn_model.predict(user_scaled)
probability = knn_model.predict_proba(user_scaled)

# --- GIAI ĐOẠN 5: XUẤT KẾT QUẢ THÔ (LOGGING) ---
result = "DƯƠNG TÍNH (Nguy cơ cao)" if prediction[0] == 1 else "ÂM TÍNH (An toàn)"
confidence = probability[0][prediction[0]] * 100

print("-" * 30)
print(f"Kết quả dự đoán: {result}")
print(f"Độ tin cậy của thuật toán: {confidence:.2f}%")
print("-" * 30)

# TODO: Cần lưu kết quả này vào database hoặc gửi email cảnh báo nếu dương tính.
