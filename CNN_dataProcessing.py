import os
import pandas as pd

# Định nghĩa đường dẫn đến thư mục dữ liệu
DATA_DIR = "data"  # Thư mục chính chứa train, test, valid
OUTPUT_CSV = "FruitDataFrame.csv"  # File CSV đầu ra

# Danh sách để lưu thông tin nhãn
data = []

# Duyệt qua từng tập dữ liệu (train, test, valid)
for subset in ["train", "test", "valid"]:
    subset_path = os.path.join(DATA_DIR, subset)
    
    # Duyệt qua từng loại trái cây
    for fruit_type in os.listdir(subset_path):
        fruit_path = os.path.join(subset_path, fruit_type)
        images_path = os.path.join(fruit_path, "images")  # Thư mục chứa ảnh
        labels_path = os.path.join(fruit_path, "labels")  # Thư mục chứa file TXT
        
        # Kiểm tra nếu thư mục labels và images tồn tại
        if os.path.isdir(labels_path) and os.path.isdir(images_path):
            for file in os.listdir(labels_path):
                if file.endswith(".txt"):  # Nếu file nhãn là TXT (YOLO format)
                    label_path = os.path.join(labels_path, file)
                    
                    # Đọc file TXT và đếm số dòng (số bounding box)
                    with open(label_path, "r") as f:
                        bbox_count = sum(1 for _ in f)

                    # Lấy tên file ảnh tương ứng
                    image_file = file.replace(".txt", ".jpg")  # Giả định ảnh là JPG
                    image_path = os.path.join(images_path, image_file)

                    # Lưu thông tin vào danh sách (chỉ thêm nếu ảnh tồn tại)
                    if os.path.exists(image_path):
                        data.append([image_file, fruit_type, bbox_count, subset])

# Chuyển danh sách thành DataFrame và lưu CSV
df = pd.DataFrame(data, columns=["filename", "fruit_type", "count", "subset"])
df.to_csv(OUTPUT_CSV, index=False)

print(f"Đã lưu thông tin vào {OUTPUT_CSV}")
