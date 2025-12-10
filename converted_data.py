import os
import shutil

# Dùng cho xử lý dữ liệu trên kaggle

# Định nghĩa đường dẫn gốc của dữ liệu
source_dir = "data_source"  # Thư mục gốc chứa các loại trái cây (apple, orange, ...)
train_dest = "data/train"
valid_dest = "data/valid"
test_dest = "data/test"

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(train_dest, exist_ok=True)
os.makedirs(valid_dest, exist_ok=True)
os.makedirs(test_dest, exist_ok=True)

# Hàm sao chép toàn bộ nội dung của một thư mục (gồm cả file và thư mục con)
def copy_all(src, dst):
    if os.path.exists(src):
        os.makedirs(dst, exist_ok=True)
        for item in os.listdir(src):
            src_path = os.path.join(src, item)
            dst_path = os.path.join(dst, item)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)  # Sao chép cả thư mục con
            else:
                shutil.copy2(src_path, dst_path)  # Sao chép file

# Duyệt qua từng loại trái cây (apple, orange, banana, ...)
for fruit in os.listdir(source_dir):
    fruit_path = os.path.join(source_dir, fruit)
    if not os.path.isdir(fruit_path):  # Bỏ qua file, chỉ xử lý thư mục
        continue

    # Đường dẫn thư mục train, valid, test của mỗi loại trái cây
    train_source = os.path.join(fruit_path, "train")
    valid_source = os.path.join(fruit_path, "valid")
    test_source = os.path.join(fruit_path, "test")

    # Đường dẫn thư mục mới trong data1
    train_target = os.path.join(train_dest, fruit)
    valid_target = os.path.join(valid_dest, fruit)
    test_target = os.path.join(test_dest, fruit)

    # Sao chép dữ liệu từ train, valid, test (nếu có)
    copy_all(train_source, train_target)
    copy_all(valid_source, valid_target)
    copy_all(test_source, test_target)
