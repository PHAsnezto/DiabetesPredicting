import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import sys

# --- GIAI ĐOẠN 1 & 2: CHUẨN BỊ ---
def setup_model():
    try:
        df = pd.read_csv('diabetes_data.csv')
        X = df.drop(['Outcome'], axis=1)
        y = df['Outcome']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_scaled, y)
        
        return knn, scaler, X.columns
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file 'diabetes_data.csv'!")
        sys.exit()

# --- GIAI ĐOẠN 3: HÀM NHẬP LIỆU TƯƠNG TÁC ---
def get_user_input():
    print("\n" + "="*30)
    print(" NHẬP CHỈ SỐ SỨC KHỎE")
    print("="*30)
    
    # Danh sách các câu hỏi tương ứng với 8 cột dữ liệu
    questions = [
        ("Số lần mang thai", 0, 20),
        ("Nồng độ Glucose (sau 2h)", 0, 300),
        ("Huyết áp tâm trương (mm Hg)", 0, 150),
        ("Độ dày nếp gấp da (mm)", 0, 100),
        ("Nồng độ Insulin (mu U/ml)", 0, 900),
        ("Chỉ số khối cơ thể (BMI)", 0.0, 70.0),
        ("Chức năng phả hệ (0.0 - 2.5)", 0.0, 2.5),
        ("Độ tuổi", 1, 120)
    ]
    
    user_data = []
    for q_text, min_val, max_val in questions:
        while True:
            try:
                val = float(input(f"➤ {q_text} [{min_val}-{max_val}]: "))
                if min_val <= val <= max_val:
                    user_data.append(val)
                    break
                else:
                    print(f"⚠️ Vui lòng nhập trong khoảng từ {min_val} đến {max_val}.")
            except ValueError:
                print("⚠️ Lỗi: Bạn phải nhập một con số!")
    
    return user_data

# --- CHƯƠNG TRÌNH CHÍNH ---
def main():
    # 1. Khởi tạo
    model, scaler, column_names = setup_model()
    
    # 2. Lấy dữ liệu từ bàn phím
    data = get_user_input()
    
    # 3. Xử lý dự đoán
    user_df = pd.DataFrame([data], columns=column_names)
    user_scaled = scaler.transform(user_df)
    
    prediction = model.predict(user_scaled)
    probability = model.predict_proba(user_scaled)
    
    # 4. Xuất kết quả
    print("\n" + "*"*30)
    if prediction[0] == 1:
        print("🚩 KẾT QUẢ: CÓ NGUY CƠ CAO")
    else:
        print("✅ KẾT QUẢ: NGUY CƠ THẤP (AN TOÀN)")
    
    conf = probability[0][prediction[0]] * 100
    print(f"Độ tin cậy của thuật toán: {conf:.2f}%")
    print("*"*30 + "\n")

if __name__ == "__main__":
    main()
