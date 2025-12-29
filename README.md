# Visual Question Answering (VQA) Project: Fruit Q&A from Images

This project builds an AI system that can "see" fruit images and answer natural questions about the **fruit type** and **quantity**.

The system uses a **Modular** approach, combining Computer Vision and Natural Language Processing (NLP).

## Folder Structure

```
├── CNN_cla.ipynb                 # Notebook for training fruit classification model
├── CNN_reg.ipynb                 # Notebook for training fruit counting model (Regression)
├── CreateQA.ipynb                # Notebook for generating Question-Answer data (QA Generation)
├── VQAModel_with_attention.ipynb # Notebook for training the main VQA model (Seq2Seq + Attention)
├── VQAModel_no_attention.ipynb   # Notebook for training the basic VQA model (Seq2Seq Baseline)
├── Model_Evaluate.ipynb          # Notebook for evaluating system performance
├── FruitDataFrame.csv            # Metadata file (image name, type, quantity)
├── data/                         # Folder containing original images (Download from drive link)
│   ├── train/                    # Training set
│   ├── test/                     # Test set
│   └── valid/                    # Validation set
├── Images_QA/                    # Folder containing JSON files (question-answer pairs for each image after running CreateQA.ipynb)
└── seq2seqData/                  # Folder containing preprocessed data (numpy arrays) for Seq2Seq (after running CreateQA.ipynb)
```

### 1. `CNN_cla.ipynb` (Fruit Classification)
This notebook builds a Convolutional Neural Network (CNN) model to identify the type of fruit in the image.

*   **Goal:** Identify what fruit is in the image (Apple, Banana, Orange, etc.).
*   **Process:**
    *   **Load Data:** Read images from `data/train`.
    *   **Preprocessing:** Resize images, normalize pixels (0-1), One-hot encoding labels.
    *   **Build Model:** Use CNN architecture with Conv2D, MaxPooling, and Dense layers.
    *   **Train:** Optimize `categorical_crossentropy` loss function.
    *   **Result:** Save model `fruit_classifier_cnn.keras`.

### 2. `CNN_reg.ipynb` (Counting Quantity)
This notebook builds a CNN model to predict the number of fruits in the image (Regression Problem).

*   **Goal:** Count the number of fruits in the image.
*   **Process:**
    *   **Load Data:** Use `FruitDataFrame.csv` to get quantity labels (`count`).
    *   **Build Model:** Similar CNN architecture but the last layer has 1 neuron (linear activation) to return a real number.
    *   **Train:** Optimize `mse` (Mean Squared Error) loss function.
    *   **Result:** Save model `fruit_regression_cnn.keras`.

### 3. `CreateQA.ipynb` (Generate VQA Data)
This notebook acts as a bridge, creating training data for the language model from available images and labels.

*   **Goal:** Create Question - Answer pairs for each image.
*   **Process:**
    *   **Read Metadata:** Get `fruit_type` and `count` info from the CSV file.
    *   **Generate Questions (Template):**
        *   *Ask quantity:* "How many fruits are in the image?", "Count the fruits?"...
        *   *Ask type:* "What fruit is this?", "Which fruit type is in the image?"...
        *   *Combined:* "How many apples are in the picture?"...
    *   **Generate Answers:** Create natural answers (e.g., "There are 5 apples in the image").
    *   **Save:** Export to JSON files in `Images_QA/` and `.npy` files in `seq2seqData/` (processed by Tokenizer and Padding).

### 4. `VQAModel_with_attention.ipynb` (Main VQA Model)
This is the heart of the project, combining information from the image and the question to give an answer.

*   **Goal:** Answer user's natural questions about the image.
*   **Model Architecture (Seq2Seq + Attention):**
    *   **Encoder (Question Processing):** Use LSTM layer to encode the question into a semantic vector.
    *   **Image Feature Fusion:** Combine feature vectors from the image (Output of `CNN_cla` and `CNN_reg`) into the Encoder state.
    *   **Decoder (Answer Generation):** Use LSTM to generate each word of the answer.
    *   **Attention Mechanism:** Helps the model focus on important parts of the question and image info at each generation step, improving accuracy compared to normal Seq2Seq.
*   **Inference (Prediction):**
    1.  Image -> CNN Classification -> Fruit Type.
    2.  Image -> CNN Counting -> Quantity.
    3.  (Type + Quantity + Question) -> VQA Model -> Answer.

### 5. `Model_Evaluate.ipynb` (Evaluation)
Notebook used to measure the accuracy of the whole system.

*   **Goal:** Objectively evaluate performance on the Test set.
*   **Metrics:**
    *   **Accuracy:** Accuracy of the answer.
    *   **BLEU Score:** Evaluate similarity between machine's answer and reference answer (important in text generation).
    *   **Confusion Matrix:** Analyze errors of the classification model.

## Installation Requirements

The project requires the following Python libraries:

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn
```

---

# Dự án Visual Question Answering (VQA): Hỏi đáp về Trái cây qua Hình ảnh

Dự án này xây dựng một hệ thống AI có khả năng "nhìn" vào hình ảnh các loại trái cây và trả lời các câu hỏi tự nhiên liên quan đến **loại trái cây** và **số lượng** của chúng.

Hệ thống được thiết kế theo hướng tiếp cận **Modular (Mô-đun hóa)**, kết hợp giữa Thị giác máy tính (Computer Vision) và Xử lý ngôn ngữ tự nhiên (NLP).

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
Notebook này đóng vai trò cầu nối, tạo ra bộ dữ liệu huấn luyện cho mô hình ngôn ngữ từ dữ liệu ảnh và nhãn có sẵn.

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
Đây là trái tim của dự án, nơi kết hợp thông tin từ ảnh và câu hỏi để đưa ra câu trả lời.

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
