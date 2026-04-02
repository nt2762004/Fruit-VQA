from __future__ import annotations

import hashlib
import html
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

from src.preprocess import build_image_feature_vector, load_seq2seq_data, resolve_path, scan_dataset
from src.predict import predict_answer, predict_fruit_attributes_from_models, load_regression_scale
from src.runtime_data import ensure_dataset_dir, get_runtime_cache_root


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SEQ2SEQ_DIR = PROJECT_ROOT / "seq2seqData"
CLASSIFIER_MODEL = PROJECT_ROOT / "fruit_classifier.keras"
REGRESSION_MODEL = PROJECT_ROOT / "fruit_regression.keras"
DEFAULT_MODEL = PROJECT_ROOT / "seq2seq_attention_full.keras"
THUMBNAIL_SIZE = (48, 48)
UPLOAD_MATCH_THRESHOLD = 0.82


def get_streamlit_secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


@st.cache_resource(show_spinner=False)
def load_runtime_data_dir() -> str:
    resolved_path = ensure_dataset_dir(
        DATA_DIR,
        download_url=get_streamlit_secret("DATA_ZIP_URL"),
        file_id=get_streamlit_secret("DATA_ZIP_FILE_ID"),
        cache_root=get_runtime_cache_root(),
    )
    return resolved_path.as_posix()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700;800&display=swap');

        :root {
            --bg-0: #07111e;
            --bg-1: #0a1830;
            --bg-2: #101b33;
            --panel: rgba(13, 20, 38, 0.74);
            --border: rgba(148, 163, 184, 0.18);
            --text: #e5eefc;
            --muted: #9fb0c9;
            --accent: #22c55e;
            --accent-2: #06b6d4;
            --accent-3: #f59e0b;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(34, 197, 94, 0.16), transparent 30%),
                radial-gradient(circle at top right, rgba(6, 182, 212, 0.14), transparent 22%),
                radial-gradient(circle at bottom left, rgba(245, 158, 11, 0.1), transparent 22%),
                linear-gradient(135deg, var(--bg-0) 0%, var(--bg-1) 48%, var(--bg-2) 100%);
            color: var(--text);
            font-family: 'Space Grotesk', sans-serif;
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1360px;
        }

        [data-testid="stSidebar"] {
            background: rgba(4, 9, 20, 0.9);
            border-right: 1px solid var(--border);
        }

        .hero {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.18), rgba(6, 182, 212, 0.14));
            border: 1px solid var(--border);
            border-radius: 30px;
            padding: 1.4rem 1.5rem 1.3rem;
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.26);
            backdrop-filter: blur(18px);
            margin-bottom: 1rem;
        }

        .eyebrow {
            color: var(--accent-3);
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-size: 0.74rem;
            margin-bottom: 0.45rem;
        }

        .hero-title {
            color: var(--text);
            font-size: clamp(2rem, 4vw, 3.55rem);
            line-height: 1.03;
            margin: 0;
            font-weight: 800;
        }

        .hero-copy {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.65;
            margin-top: 0.85rem;
            max-width: 980px;
        }

        .panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 1rem;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(16px);
        }

        .section-title {
            color: var(--text);
            font-size: 1.04rem;
            font-weight: 700;
            margin: 0.1rem 0 0.8rem;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 0.9rem 0 1rem;
        }

        .metric-card {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.76), rgba(15, 23, 42, 0.92));
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            min-height: 110px;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }

        .metric-value {
            color: var(--text);
            font-size: 1.45rem;
            font-weight: 800;
            margin-top: 0.35rem;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 0.25rem;
            line-height: 1.45;
        }

        .selection-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            background: rgba(15, 23, 42, 0.8);
            color: var(--muted);
            font-size: 0.85rem;
            margin-bottom: 0.75rem;
        }

        .match-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.72rem;
            border-radius: 999px;
            background: rgba(6, 182, 212, 0.15);
            border: 1px solid rgba(6, 182, 212, 0.28);
            color: #d8fbff;
            font-size: 0.86rem;
        }

        .result-card {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.17), rgba(6, 182, 212, 0.13));
            border: 1px solid rgba(34, 197, 94, 0.28);
            border-radius: 26px;
            padding: 1.1rem 1.15rem 1.2rem;
            box-shadow: 0 20px 46px rgba(0, 0, 0, 0.24);
            margin-top: 1rem;
        }

        .result-answer {
            color: #eefcf8;
            font-size: 1.65rem;
            font-weight: 800;
            line-height: 1.15;
            margin-top: 0.35rem;
            margin-bottom: 0.4rem;
        }

        .result-meta {
            color: var(--muted);
            font-size: 0.94rem;
            line-height: 1.65;
        }

        .empty-state {
            border: 1px dashed rgba(148, 163, 184, 0.25);
            border-radius: 22px;
            padding: 1rem;
            color: var(--muted);
            background: rgba(15, 23, 42, 0.42);
        }

        .footer-note {
            color: var(--muted);
            font-size: 0.85rem;
            margin-top: 1rem;
            line-height: 1.6;
        }

        div[data-testid="stButton"] > button {
            border: 0;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent), var(--accent-2));
            color: white;
            font-weight: 700;
            padding: 0.6rem 1rem;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        div[data-testid="stButton"] > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 24px rgba(6, 182, 212, 0.32);
        }

        .stTextArea textarea {
            background: rgba(15, 23, 42, 0.86);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 18px;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: rgba(15, 23, 42, 0.86);
            border: 1px dashed rgba(148, 163, 184, 0.35);
            border-radius: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_test_gallery(data_dir: str | Path) -> list[dict[str, object]]:
    data_path = Path(data_dir)
    rows: list[dict[str, object]] = []

    for subset in ("train", "valid", "test"):
        subset_path = data_path / subset
        if not subset_path.exists():
            continue

        for fruit_dir in sorted(path for path in subset_path.iterdir() if path.is_dir()):
            images_dir = fruit_dir / "images"
            labels_dir = fruit_dir / "labels"
            if not images_dir.exists() or not labels_dir.exists():
                continue

            for label_path in sorted(labels_dir.glob("*.txt")):
                with label_path.open("r", encoding="utf-8") as handle:
                    count = sum(1 for line in handle if line.strip())

                image_path = images_dir / f"{label_path.stem}.jpg"
                if not image_path.exists():
                    alternate = images_dir / f"{label_path.stem}.png"
                    if alternate.exists():
                        image_path = alternate
                    else:
                        continue

                rows.append(
                    {
                        "filename": image_path.name,
                        "fruit_type": fruit_dir.name,
                        "count": int(count),
                        "subset": subset,
                        "image_path": image_path.resolve().as_posix(),
                    }
                )

    if not rows:
        raise ValueError(f"No valid samples were found under {data_path}")

    test_rows = [row for row in rows if row["subset"] == "test"]
    return sorted(test_rows, key=lambda item: (str(item["fruit_type"]), str(item["filename"])))


@st.cache_data(show_spinner=False)
def load_reference_index(data_dir: str | Path) -> tuple[list[dict[str, object]], np.ndarray]:
    frame = scan_dataset(data_dir)
    reference_rows: list[dict[str, object]] = []
    signatures: list[np.ndarray] = []

    for row in frame.itertuples(index=False):
        image_path = Path(row.image_path).resolve()
        if not image_path.exists():
            continue

        with Image.open(image_path) as image:
            signature = image_to_signature(image)

        reference_rows.append(
            {
                "filename": str(row.filename),
                "fruit_type": str(row.fruit_type),
                "count": int(row.count),
                "subset": str(row.subset),
                "image_path": image_path.as_posix(),
            }
        )
        signatures.append(signature)

    if not reference_rows:
        raise ValueError(f"No valid reference images were found under {data_dir}")

    signature_matrix = np.vstack(signatures).astype(np.float32)
    norms = np.linalg.norm(signature_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return reference_rows, signature_matrix / norms


@st.cache_data(show_spinner=False)
def list_model_files(root_dir: str | Path) -> list[str]:
    root_path = Path(root_dir)
    model_paths = sorted(root_path.glob("*.keras"), key=lambda item: item.name.lower())
    if DEFAULT_MODEL.exists() and DEFAULT_MODEL not in model_paths:
        model_paths.insert(0, DEFAULT_MODEL)

    unique_paths: list[str] = []
    for path in model_paths:
        resolved = path.resolve().as_posix()
        if resolved not in unique_paths:
            unique_paths.append(resolved)
    return unique_paths


@st.cache_resource(show_spinner=False)
def load_prediction_bundle(model_path: str, seq2seq_dir: str, data_dir: str) -> dict[str, object]:
    seq2seq_data = load_seq2seq_data(seq2seq_dir)
    model = tf.keras.models.load_model(resolve_path(model_path), compile=False)
    fruit_classes = sorted(scan_dataset(data_dir)["fruit_type"].unique().tolist())

    return {
        "model": model,
        "tokenizer": seq2seq_data["tokenizer"],
        "max_length": int(seq2seq_data["max_length"]),
        "fruit_classes": fruit_classes,
        "count_scale": float(seq2seq_data.get("count_scale", 1.0)),
        "num_samples": int(seq2seq_data["image_features"].shape[0]),
    }


@st.cache_resource(show_spinner=False)
def load_vqa_model_bundle(model_path: str, seq2seq_dir: str, data_dir: str) -> dict[str, object]:
    return load_prediction_bundle(model_path, seq2seq_dir, data_dir)


def image_to_signature(image: Image.Image) -> np.ndarray:
    fitted_image = ImageOps.fit(
        image.convert("RGB"),
        THUMBNAIL_SIZE,
        method=Image.Resampling.BILINEAR,
    )
    return np.asarray(fitted_image, dtype=np.float32).reshape(-1) / 255.0


def match_uploaded_image(uploaded_bytes: bytes, reference_rows: list[dict[str, object]], normalized_signatures: np.ndarray) -> dict[str, object]:
    with Image.open(BytesIO(uploaded_bytes)) as image:
        query_signature = image_to_signature(image)

    query_norm = np.linalg.norm(query_signature)
    if query_norm == 0:
        query_norm = 1.0
    query_vector = query_signature / query_norm

    scores = normalized_signatures @ query_vector
    best_index = int(np.argmax(scores))
    best_row = dict(reference_rows[best_index])
    best_row["match_score"] = float(scores[best_index])
    return best_row


def predict_from_row(row: dict[str, object], question: str, model_path: str, seq2seq_dir: str, data_dir: str) -> dict[str, object]:
    bundle = load_vqa_model_bundle(model_path, seq2seq_dir, data_dir)
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    max_length = int(bundle["max_length"])
    fruit_classes = bundle["fruit_classes"]

    fruit_type = str(row["fruit_type"])
    count = int(row["count"])
    count_scale = float(bundle.get("count_scale", 1.0))
    image_feature_vector = build_image_feature_vector(fruit_type, count, fruit_classes, count_scale)
    answer = predict_answer(model, question, image_feature_vector, tokenizer, max_length)

    return {
        "image_path": str(row.get("image_path", "")),
        "question": question,
        "fruit_type": fruit_type,
        "count": count,
        "answer": answer,
        "feature_dim": int(image_feature_vector.shape[0]),
        "num_samples": int(bundle["num_samples"]),
    }


def render_metric(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="metric-value">{html.escape(value)}</div>
            <div class="metric-note">{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(result: dict[str, object]) -> None:
    st.markdown(
        f"""
        <div class="result-card">
            <div class="eyebrow">Kết quả dự đoán</div>
            <div class="result-answer">{html.escape(str(result.get('answer') or 'Không tạo được câu trả lời rõ ràng'))}</div>
            <div class="result-meta">
                <b>Câu hỏi:</b> {html.escape(str(result.get('question', '')))}<br>
                <b>Ảnh suy luận:</b> {html.escape(str(result.get('selection_note', '')))}<br>
                <b>Fruit type:</b> {html.escape(str(result.get('fruit_type', '')))}<br>
                <b>Count:</b> {html.escape(str(result.get('count', '')))}<br>
                <b>Feature dim:</b> {html.escape(str(result.get('feature_dim', '')))}<br>
                <b>Model:</b> {html.escape(Path(str(result.get('model_path', DEFAULT_MODEL))).name)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_selected_gallery_row(gallery_rows: list[dict[str, object]]) -> dict[str, object]:
    if "selected_gallery_path" not in st.session_state:
        st.session_state.selected_gallery_path = gallery_rows[0]["image_path"]

    selected_path = st.session_state.selected_gallery_path
    selected_row = next((row for row in gallery_rows if row["image_path"] == selected_path), gallery_rows[0])
    st.session_state.selected_gallery_path = selected_row["image_path"]
    return selected_row


def open_gallery_dialog(gallery_rows: list[dict[str, object]]) -> None:
    grouped_rows: dict[str, list[dict[str, object]]] = {}
    for row in gallery_rows:
        grouped_rows.setdefault(str(row["fruit_type"]), []).append(row)

    fruit_types = sorted(grouped_rows)

    if "selected_gallery_fruit_type" not in st.session_state or st.session_state.selected_gallery_fruit_type not in fruit_types:
        st.session_state.selected_gallery_fruit_type = fruit_types[0]

    initial_rows = grouped_rows[st.session_state.selected_gallery_fruit_type]
    if "selected_gallery_path" not in st.session_state or not any(
        row["image_path"] == st.session_state.selected_gallery_path for row in initial_rows
    ):
        st.session_state.selected_gallery_path = initial_rows[0]["image_path"]

    @st.dialog("Chọn ảnh từ thư viện")
    def _dialog() -> None:
        st.markdown("Chọn loại ảnh trước, sau đó chọn đúng ảnh bạn muốn dùng để dự đoán.")

        selected_fruit_type = st.selectbox(
            "Loại ảnh",
            fruit_types,
            key="selected_gallery_fruit_type",
            help="Lọc danh sách ảnh theo loại trái cây.",
        )

        rows_for_type = grouped_rows[selected_fruit_type]
        selected_index = 0
        if "selected_gallery_path" in st.session_state:
            for index, row in enumerate(rows_for_type):
                if row["image_path"] == st.session_state.selected_gallery_path:
                    selected_index = index
                    break

        selected_image_index = st.selectbox(
            "Ảnh dùng để dự đoán",
            list(range(len(rows_for_type))),
            index=selected_index,
            format_func=lambda value: f"{rows_for_type[int(value)]['filename']} · count {rows_for_type[int(value)]['count']} · {rows_for_type[int(value)]['subset']}",
            help="Chọn đúng ảnh mẫu để hệ thống suy luận fruit type và count.",
        )

        selected_row = rows_for_type[int(selected_image_index)]

        st.image(selected_row["image_path"], width="stretch")
        st.markdown(
            f"<div class='selection-chip'>{html.escape(str(selected_row['filename']))} | {html.escape(str(selected_row['fruit_type']))} | count {html.escape(str(selected_row['count']))}</div>",
            unsafe_allow_html=True,
        )

        confirm_col, cancel_col = st.columns(2)
        with confirm_col:
            if st.button("Dùng ảnh này", width="stretch"):
                st.session_state.selected_gallery_path = selected_row["image_path"]
                st.session_state.selected_source = "Thư viện ảnh"
                st.session_state.prediction_result = None
                st.session_state.show_gallery_dialog = False
                st.rerun()
        with cancel_col:
            if st.button("Đóng", width="stretch"):
                st.session_state.show_gallery_dialog = False
                st.rerun()

    _dialog()


def image_bytes_to_preview(uploaded_file) -> bytes:
    if uploaded_file is None:
        return b""
    return uploaded_file.getvalue()


@st.cache_resource(show_spinner=False)
def load_direct_recognition_bundle(data_dir: str | Path) -> dict[str, object]:
    fruit_classes = sorted(scan_dataset(data_dir)["fruit_type"].unique().tolist())
    classifier_model = tf.keras.models.load_model(resolve_path(CLASSIFIER_MODEL), compile=False)
    regression_model = tf.keras.models.load_model(resolve_path(REGRESSION_MODEL), compile=False)
    count_scale = load_regression_scale(REGRESSION_MODEL)

    return {
        "fruit_classes": fruit_classes,
        "classifier_model": classifier_model,
        "regression_model": regression_model,
        "count_scale": count_scale,
    }


def main() -> None:
    st.set_page_config(page_title="Fruit VQA Studio", page_icon="🍓", layout="wide")
    inject_styles()

    try:
        with st.spinner("Đang chuẩn bị dataset cho phiên chạy hiện tại..."):
            runtime_data_dir = load_runtime_data_dir()
    except Exception as error:  # noqa: BLE001
        st.error("Không thể khởi tạo dataset cho Streamlit Cloud.")
        st.info("Hãy đặt DATA_ZIP_URL hoặc DATA_ZIP_FILE_ID trong Streamlit secrets, hoặc đảm bảo thư mục data/ đã có sẵn trong repo.")
        st.code(str(error), language="text")
        st.stop()

    gallery_rows = load_test_gallery(runtime_data_dir)
    if not gallery_rows:
        st.error("Không tìm thấy ảnh trong dataset. Hãy kiểm tra lại cấu trúc train/valid/test sau khi giải nén zip.")
        st.stop()

    model_files = list_model_files(PROJECT_ROOT)
    if not model_files:
        st.error("Không tìm thấy file model .keras trong thư mục Final_version.")
        st.stop()

    if "selected_source" not in st.session_state:
        st.session_state.selected_source = "Thư viện ảnh"
    if "show_gallery_dialog" not in st.session_state:
        st.session_state.show_gallery_dialog = False
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None
    if "uploaded_file_bytes" not in st.session_state:
        st.session_state.uploaded_file_bytes = None
    if "uploaded_recognition_key" not in st.session_state:
        st.session_state.uploaded_recognition_key = None
    if "uploaded_recognition_row" not in st.session_state:
        st.session_state.uploaded_recognition_row = None

    selected_gallery_row = get_selected_gallery_row(gallery_rows)
    chosen_model = DEFAULT_MODEL.resolve().as_posix() if DEFAULT_MODEL.exists() else model_files[0]
    fruit_types = sorted({str(row["fruit_type"]) for row in gallery_rows})

    with st.sidebar:
        st.markdown("### Điều khiển")
        st.caption("Chọn nguồn ảnh, rồi đặt câu hỏi cho ảnh đã nạp.")

        source_mode = st.radio(
            "Nguồn ảnh",
            ["Thư viện ảnh", "Tải ảnh từ máy"],
            key="selected_source",
        )

        if source_mode == "Thư viện ảnh":
            st.markdown("<div class='selection-chip'>Ảnh hiện tại: thư viện mẫu</div>", unsafe_allow_html=True)
            if st.button("Mở popup chọn ảnh", width="stretch"):
                st.session_state.show_gallery_dialog = True
            st.caption("Popup sẽ cho phép chọn loại ảnh rồi chọn ảnh cụ thể để dự đoán.")
        else:
            st.markdown("<div class='selection-chip'>Ảnh hiện tại: tải từ máy</div>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Chọn ảnh từ máy", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
            st.caption("Ảnh upload sẽ được nhận diện trực tiếp bằng classifier/regression trước khi trả lời.")
            if uploaded_file is not None:
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.uploaded_file_bytes = image_bytes_to_preview(uploaded_file)

        if st.button("Xóa kết quả", width="stretch"):
            st.session_state.prediction_result = None

    if st.session_state.show_gallery_dialog:
        open_gallery_dialog(gallery_rows)

    st.markdown(
        """
        <div class="hero">
            <h1 class="hero-title">Fruit VQA Studio</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if source_mode == "Thư viện ảnh":
        active_row = selected_gallery_row
        active_image = active_row["image_path"]
        selection_note = f"{active_row['filename']} | {active_row['fruit_type']} | count {active_row['count']}"
    else:
        if st.session_state.uploaded_file_bytes is None:
            active_row = None
            active_image = None
            selection_note = "chưa có ảnh upload"
        else:
            if not CLASSIFIER_MODEL.exists() or not REGRESSION_MODEL.exists():
                st.warning("Chưa có fruit_classifier.keras hoặc fruit_regression.keras. Hãy train classifier/regression trước.")
                active_row = None
                active_image = st.session_state.uploaded_file_bytes
                selection_note = "chưa có model classifier/regression"
            else:
                uploaded_key = hashlib.sha1(st.session_state.uploaded_file_bytes).hexdigest()
                if st.session_state.uploaded_recognition_key != uploaded_key:
                    direct_bundle = load_direct_recognition_bundle(runtime_data_dir)
                    recognized_row = predict_fruit_attributes_from_models(
                        st.session_state.uploaded_file_bytes,
                        direct_bundle["classifier_model"],
                        direct_bundle["regression_model"],
                        direct_bundle["fruit_classes"],
                        float(direct_bundle["count_scale"]),
                    )
                    recognized_row["image_path"] = st.session_state.uploaded_file_name or "uploaded_image"
                    st.session_state.uploaded_recognition_key = uploaded_key
                    st.session_state.uploaded_recognition_row = recognized_row

                active_row = st.session_state.uploaded_recognition_row
            active_image = st.session_state.uploaded_file_bytes
            if active_row is not None:
                selection_note = (
                    f"Upload: {st.session_state.uploaded_file_name or 'ảnh từ máy'} | "
                    f"nhận diện trực tiếp {active_row['fruit_type']} (count {active_row['count']}, confidence {active_row.get('class_confidence', 0.0):.3f})"
                )

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    with metric_col_1:
        render_metric("Test classes", str(len(fruit_types)), f"Class đang có trong gallery: {len(fruit_types)}")
    with metric_col_2:
        render_metric("Ảnh gallery", str(len(gallery_rows)), "Ảnh mặc định hiển thị từ split test")
    with metric_col_3:
        render_metric("Model đang dùng", Path(chosen_model).name, "Dùng để sinh câu trả lời")

    left_col, right_col = st.columns([1.02, 0.98], gap="large")

    with left_col:
        st.markdown("<div class='section-title'>Ảnh đang nạp</div>", unsafe_allow_html=True)
        if active_image is None:
            st.markdown(
                "<div class='empty-state'>Chưa có ảnh upload. Hãy chọn ảnh trong thư viện hoặc tải ảnh từ máy lên.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.image(active_image, width="stretch")
            st.markdown(
                f"<div class='selection-chip'>{html.escape(selection_note)}</div>",
                unsafe_allow_html=True,
            )
            if source_mode == "Tải ảnh từ máy" and active_row is not None:
                st.markdown(
                    f"<div class='match-chip'>Suy luận trực tiếp: {html.escape(str(active_row['fruit_type']))} · count {html.escape(str(active_row['count']))} · confidence {float(active_row.get('class_confidence', 0.0)):.3f}</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("<div class='section-title'>Đặt câu hỏi</div>", unsafe_allow_html=True)
        with st.form("question_form", clear_on_submit=False):
            question = st.text_area(
                "Nhập câu hỏi của bạn",
                placeholder="Ví dụ: Trong ảnh có bao nhiêu trái cây?",
                height=140,
            )
            submitted = st.form_submit_button("Dự đoán", width="stretch")

        st.caption("Mô hình sẽ dùng ảnh đã nạp cùng câu hỏi của bạn để sinh câu trả lời.")

    if submitted:
        if not question.strip():
            st.warning("Hãy nhập câu hỏi trước khi dự đoán.")
        elif active_row is None or active_image is None:
            st.warning("Chưa có ảnh hợp lệ để dự đoán. Hãy chọn ảnh trong thư viện hoặc tải ảnh lên trước.")
        else:
            with st.spinner("Đang tải model và tạo dự đoán..."):
                try:
                    result = predict_from_row(active_row, question.strip(), chosen_model, str(SEQ2SEQ_DIR), runtime_data_dir)
                except Exception as error:  # noqa: BLE001
                    st.error("Không thể chạy mô hình dự đoán. Hãy kiểm tra lại file model hoặc môi trường TensorFlow.")
                    st.code(str(error), language="text")
                    st.stop()

            result["model_path"] = chosen_model
            result["selection_note"] = selection_note
            st.session_state.prediction_result = result

    if st.session_state.prediction_result is not None:
        render_result_card(st.session_state.prediction_result)

    st.markdown(
        "<div class='footer-note'>Nếu bạn dùng ảnh ngoài dataset, app sẽ nhận diện trực tiếp bằng classifier/regression rồi mới đưa feature vào VQA.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
