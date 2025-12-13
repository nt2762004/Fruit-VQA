# Visual Question Answering (VQA): Hỏi đáp về Trái cây qua Hình ảnh

Dự án này xây dựng một hệ thống AI có khả năng "nhìn" vào hình ảnh các loại trái cây và trả lời các câu hỏi tự nhiên liên quan đến **loại trái cây** và **số lượng** của chúng.

Hệ thống được thiết kế theo hướng tiếp cận **Modular (Mô-đun hóa)**, kết hợp giữa Thị giác máy tính (Computer Vision) và Xử lý ngôn ngữ tự nhiên (NLP).

**Link Dataset**: [Fruit Dataset](https://drive.google.com/file/d/1Y5H61uAQLOAltYrpXGmDLfDIYSVyqQoZ/view?usp=sharing)

## Cấu trúc Thư mục

```
├── CNN_cla.ipynb                 # Notebook huấn luyện mô hình phân loại trái cây (Classification)
├── CNN_reg.ipynb                 # Notebook huấn luyện mô hình đếm số lượng trái cây (Regression)
├── CreateQA.ipynb                # Notebook sinh dữ liệu câu hỏi - câu trả lời (QA Generation)
├── VQAModel_with_attention.ipynb # Notebook huấn luyện mô hình VQA chính (Seq2Seq + Attention)
├── VQAModel_no_attention.ipynb   # Notebook huấn luyện mô hình VQA cơ bản (Seq2Seq Baseline)
├── Model_Evaluate.ipynb          # Notebook đánh giá toàn diện hiệu năng hệ thống
├── FruitDataFrame.csv            # File dữ liệu metadata (tên ảnh, loại, số lượng)
├── data/                         # Thư mục chứa dữ liệu ảnh gốc (Tải từ link drive)
│   ├── train/                    # Tập huấn luyện
│   ├── test/                     # Tập kiểm tra
│   └── valid/                    # Tập validation
├── Images_QA/                    # Thư mục chứa các file JSON (cặp câu hỏi-đáp án cho từng ảnh sau khi chạy CreateQA.ipynb)
└── seq2seqData/                  # Thư mục chứa dữ liệu đã tiền xử lý (numpy arrays) cho Seq2Seq (sau khi chạy CreateQA.ipynb)
```

### 1. `CNN_cla.ipynb` (Phân loại Trái cây)
Notebook này xây dựng mô hình Convolutional Neural Network (CNN) để nhận diện loại trái cây trong ảnh.

*   **Mục tiêu:** Xác định xem ảnh chứa loại quả gì (Táo, Chuối, Cam, v.v.).
*   **Quy trình:**
    *   **Load dữ liệu:** Đọc ảnh từ thư mục `data/train`.
    *   **Tiền xử lý:** Resize ảnh, chuẩn hóa pixel (0-1), One-hot encoding nhãn.
    *   **Xây dựng Model:** Sử dụng kiến trúc CNN với các lớp Conv2D, MaxPooling và Dense.
    *   **Huấn luyện:** Tối ưu hóa hàm mất mát `categorical_crossentropy`.
    *   **Kết quả:** Lưu model `fruit_classifier_cnn.keras`.

### 2. `CNN_reg.ipynb` (Đếm số lượng)
Notebook này xây dựng mô hình CNN để dự đoán số lượng trái cây có trong ảnh (Bài toán Hồi quy).

*   **Mục tiêu:** Đếm số lượng quả trong ảnh.
*   **Quy trình:**
    *   **Load dữ liệu:** Sử dụng `FruitDataFrame.csv` để lấy nhãn số lượng (`count`).
    *   **Xây dựng Model:** Kiến trúc CNN tương tự nhưng lớp cuối cùng là 1 neuron (linear activation) để trả về giá trị thực.
    *   **Huấn luyện:** Tối ưu hóa hàm mất mát `mse` (Mean Squared Error).
    *   **Kết quả:** Lưu model `fruit_regression_cnn.keras`.

### 3. `CreateQA.ipynb` (Sinh dữ liệu VQA)
Notebook này tạo ra bộ dữ liệu huấn luyện cho mô hình ngôn ngữ từ dữ liệu ảnh và nhãn có sẵn.

*   **Mục tiêu:** Tạo ra các cặp Câu hỏi (Question) - Câu trả lời (Answer) tương ứng với từng ảnh.
*   **Quy trình:**
    *   **Đọc Metadata:** Lấy thông tin `fruit_type` và `count` từ file CSV.
    *   **Sinh câu hỏi theo mẫu (Template):**
        *   *Hỏi số lượng:* "Trong ảnh có bao nhiêu trái cây?", "Ảnh có mấy quả?"...
        *   *Hỏi loại:* "Đây là quả gì?", "Ảnh chứa loại trái cây nào?"...
        *   *Hỏi kết hợp:* "Có bao nhiêu quả táo trong hình?"...
    *   **Sinh câu trả lời:** Tạo câu trả lời tự nhiên tương ứng (VD: "Trong ảnh có 5 quả táo").
    *   **Lưu trữ:** Xuất ra các file JSON trong `Images_QA/` và các file `.npy` trong `seq2seqData/` (đã qua Tokenizer và Padding).

### 4. `VQAModel_with_attention.ipynb` (Mô hình VQA chính)
Đây là notebook chính kết hợp thông tin từ ảnh và câu hỏi để đưa ra câu trả lời.

*   **Mục tiêu:** Trả lời câu hỏi tự nhiên của người dùng về bức ảnh.
*   **Kiến trúc Mô hình (Seq2Seq + Attention):**
    *   **Encoder (Xử lý câu hỏi):** Sử dụng lớp LSTM để mã hóa câu hỏi thành vector ngữ nghĩa.
    *   **Image Feature Fusion:** Kết hợp vector đặc trưng từ ảnh (Output của `CNN_cla` và `CNN_reg`) vào trạng thái của Encoder.
    *   **Decoder (Sinh câu trả lời):** Sử dụng LSTM để sinh từng từ của câu trả lời.
    *   **Attention Mechanism:** Giúp mô hình tập trung vào các phần quan trọng của câu hỏi và thông tin ảnh tại mỗi bước sinh từ, nâng cao độ chính xác so với Seq2Seq thường.
*   **Inference (Dự đoán):**
    1.  Ảnh -> CNN Phân loại -> Loại trái cây.
    2.  Ảnh -> CNN Đếm -> Số lượng.
    3.  (Loại + Số lượng + Câu hỏi) -> VQA Model -> Câu trả lời.

### 5. `Model_Evaluate.ipynb` (Đánh giá)
Notebook dùng để đo lường độ chính xác của toàn bộ hệ thống.

*   **Mục tiêu:** Đánh giá khách quan hiệu năng trên tập Test.
*   **Các chỉ số:**
    *   **Accuracy:** Độ chính xác của câu trả lời.
    *   **BLEU Score:** Đánh giá độ tương đồng giữa câu trả lời của máy và câu trả lời mẫu (quan trọng trong bài toán sinh văn bản).
    *   **Confusion Matrix:** Phân tích các lỗi sai của mô hình phân loại.

## Yêu cầu cài đặt

Dự án yêu cầu các thư viện Python sau:

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn
```
