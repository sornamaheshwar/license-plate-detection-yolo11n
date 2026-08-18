import re

import cv2
import easyocr
import numpy as np
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/license_plate_yolo11n.pt"
BASELINE_MODEL_PATH = "yolo11n.pt"

CONFIDENCE_THRESHOLD = 0.25

# ------------------------------------------------------------
# Actual test-set results
# ------------------------------------------------------------

MAP50 = 0.9893
MAP50_95 = 0.7083
PRECISION = 0.9880
RECALL = 0.9714

TEST_IMAGES = 2048
TEST_INSTANCES = 2134


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="License Plate Detection | YOLO11n",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.6rem;
        font-weight: 650;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    detector = YOLO(
        MODEL_PATH
    )

    baseline = YOLO(
        BASELINE_MODEL_PATH
    )

    reader = easyocr.Reader(
        ["en"],
        gpu=torch.cuda.is_available()
    )

    return (
        detector,
        baseline,
        reader
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    text = text.upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# ============================================================
# PLATE CROPPING
# ============================================================

def prepare_plate_crop(
    image,
    box
):

    h, w = image.shape[:2]

    x1, y1, x2, y2 = [
        int(value)
        for value in box
    ]

    plate_width = x2 - x1
    plate_height = y2 - y1

    # Small horizontal padding
    pad_x = max(
        2,
        int(plate_width * 0.04)
    )

    # Slight vertical padding
    pad_y = max(
        2,
        int(plate_height * 0.12)
    )

    x1 = max(
        0,
        x1 - pad_x
    )

    y1 = max(
        0,
        y1 - pad_y
    )

    x2 = min(
        w,
        x2 + pad_x
    )

    y2 = min(
        h,
        y2 + pad_y
    )

    crop = image[
        y1:y2,
        x1:x2
    ]

    return crop


# ============================================================
# OCR PREPROCESSING
# ============================================================

def create_ocr_variants(crop):

    variants = {}

    # --------------------------------------------------------
    # Upscaling
    # --------------------------------------------------------

    enlarged = cv2.resize(
        crop,
        None,
        fx=6,
        fy=6,
        interpolation=cv2.INTER_CUBIC
    )

    variants["color"] = enlarged

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2GRAY
    )

    variants["gray"] = gray

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    variants["clahe"] = enhanced

    # --------------------------------------------------------
    # Sharpening
    # --------------------------------------------------------

    blur = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        2
    )

    sharpened = cv2.addWeighted(
        enhanced,
        1.3,
        blur,
        -0.3,
        0
    )

    variants["sharpened"] = sharpened

    # --------------------------------------------------------
    # OTSU threshold
    # --------------------------------------------------------

    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )

    variants["otsu"] = otsu

    # --------------------------------------------------------
    # Adaptive threshold
    # --------------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7
    )

    variants["adaptive"] = adaptive

    return variants


# ============================================================
# OCR
# ============================================================

def extract_plate_text(
    crop,
    reader
):

    variants = create_ocr_variants(
        crop
    )

    candidates = []

    for variant_name, image in variants.items():

        try:

            results = reader.readtext(
                image,
                detail=1,

                # License plates contain
                # alphanumeric characters
                allowlist=(
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "0123456789"
                ),

                paragraph=False,

                # OCR configuration
                decoder="beamsearch",
                mag_ratio=1.5,
                text_threshold=0.45,
                low_text=0.25,
                link_threshold=0.25,
                width_ths=0.7,
                contrast_ths=0.05,
                adjust_contrast=0.7
            )

        except Exception:
            continue

        for result in results:

            if len(result) < 3:
                continue

            text = normalize_text(
                result[1]
            )

            confidence = float(
                result[2]
            )

            if len(text) < 3:
                continue

            candidates.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "variant": variant_name
                }
            )

    if not candidates:
        return None

    # Select the highest-confidence candidate.
    best = max(
        candidates,
        key=lambda x: x["confidence"]
    )

    return best


# ============================================================
# DRAW FINE-TUNED DETECTIONS
# ============================================================

def draw_detections(
    image,
    result
):

    annotated = image.copy()

    for box in result.boxes:

        x1, y1, x2, y2 = [
            int(value)
            for value in box.xyxy[0]
        ]

        confidence = float(
            box.conf[0]
        )

        label = (
            f"License_Plate "
            f"{confidence:.2f}"
        )

        # Bounding box
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            3
        )

        # Text dimensions
        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )[0]

        text_x = x1

        text_y = max(
            y1 - 10,
            text_size[1] + 10
        )

        # Label background
        cv2.rectangle(
            annotated,
            (
                text_x,
                text_y - text_size[1] - 8
            ),
            (
                text_x + text_size[0] + 8,
                text_y + 4
            ),
            (255, 0, 0),
            -1
        )

        # Label text
        cv2.putText(
            annotated,
            label,
            (
                text_x + 4,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    return annotated


# ============================================================
# BASELINE INFERENCE
# ============================================================

def run_baseline(
    image,
    baseline_model
):

    results = baseline_model(
        image,
        conf=0.20,
        device=DEVICE,
        verbose=False
    )

    return results[0]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "About the Project"
    )

    st.write(
        """
        This prototype demonstrates domain-specific
        fine-tuning of a pretrained YOLO11n model
        for license plate detection.
        """
    )

    st.divider()

    st.subheader(
        "Architecture"
    )

    st.code(
        """
Streamlit
    ↓
YOLO11n
    ↓
License Plate Detection
    ↓
Plate Crop
    ↓
EasyOCR
    ↓
Candidate Text
        """,
        language="text"
    )

    st.divider()

    st.subheader(
        "Model"
    )

    st.write(
        "**YOLO11n**"
    )

    st.write(
        "Fine-tuned for the "
        "`License_Plate` class."
    )

    st.divider()

    st.write(
        f"**Inference device:** `{DEVICE}`"
    )

    st.divider()

    st.caption(
        "Prototype built to demonstrate "
        "computer-vision model fine-tuning, "
        "evaluation and deployment."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'License Plate Detection & OCR'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'YOLO11n domain adaptation for '
    'license plate detection'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODELS
# ============================================================

try:

    detector, baseline_model, reader = (
        load_models()
    )

except Exception as e:

    st.error(
        "Failed to load the required models."
    )

    st.exception(e)

    st.stop()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🔍 Live Detection",
        "⚖️ Pretrained vs Fine-Tuned",
        "📊 Model Evaluation"
    ]
)


# ============================================================
# TAB 1 — LIVE DETECTION
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">'
        'Upload an Image'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload a vehicle image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="live_upload"
    )

    if uploaded_file is not None:

        image_pil = Image.open(
            uploaded_file
        ).convert("RGB")

        image_rgb = np.array(
            image_pil
        )

        image_bgr = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR
        )

        # ----------------------------------------------------
        # YOLO inference
        # ----------------------------------------------------

        with st.spinner(
            "Running license plate detection..."
        ):

            results = detector(
                image_bgr,
                conf=CONFIDENCE_THRESHOLD,
                device=DEVICE,
                verbose=False
            )

        result = results[0]

        plate_count = len(
            result.boxes
        )

        # ----------------------------------------------------
        # Annotated image
        # ----------------------------------------------------

        annotated = draw_detections(
            image_bgr,
            result
        )

        annotated_rgb = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )

        if plate_count > 0:

            st.success(
                f"Detected {plate_count} "
                f"license plate(s)."
            )

        else:

            st.warning(
                "No license plate was detected "
                "in this image."
            )

        st.image(
            annotated_rgb,
            caption="YOLO11n Detection Result",
            use_container_width=True
        )

        # ----------------------------------------------------
        # Process plates
        # ----------------------------------------------------

        if plate_count > 0:

            st.markdown(
                '<div class="section-title">'
                'Detected License Plates'
                '</div>',
                unsafe_allow_html=True
            )

            for i, box in enumerate(
                result.boxes
            ):

                confidence = float(
                    box.conf[0]
                )

                coordinates = [
                    int(value)
                    for value in box.xyxy[0]
                ]

                crop = prepare_plate_crop(
                    image_bgr,
                    box.xyxy[0]
                )

                st.markdown(
                    f"### License Plate {i + 1}"
                )

                col1, col2 = st.columns(
                    [1, 2]
                )

                # ------------------------------------------------
                # Plate crop
                # ------------------------------------------------

                with col1:

                    crop_rgb = cv2.cvtColor(
                        crop,
                        cv2.COLOR_BGR2RGB
                    )

                    st.image(
                        crop_rgb,
                        caption="Detected Plate Crop",
                        use_container_width=True
                    )

                # ------------------------------------------------
                # Detection details
                # ------------------------------------------------

                with col2:

                    st.metric(
                        "Detection Confidence",
                        f"{confidence * 100:.2f}%"
                    )

                    st.write(
                        "**Bounding Box:**"
                    )

                    st.code(
                        str(coordinates)
                    )

                    # ------------------------------------------------
                    # OCR
                    # ------------------------------------------------

                    with st.spinner(
                        "Extracting text with EasyOCR..."
                    ):

                        ocr_result = (
                            extract_plate_text(
                                crop,
                                reader
                            )
                        )

                    st.write(
                        "**OCR Candidate Text**"
                    )

                    if ocr_result:

                        st.info(
                            ocr_result["text"]
                        )

                        st.metric(
                            "OCR Confidence",
                            f"{ocr_result['confidence'] * 100:.2f}%"
                        )

                        st.caption(
                            "Selected from "
                            f"{ocr_result['variant']} "
                            "preprocessing."
                        )

                        st.warning(
                            "OCR is an experimental "
                            "downstream component. The "
                            "displayed text is an OCR "
                            "candidate and has not been "
                            "independently validated. "
                            "Visually verify it against "
                            "the detected plate."
                        )

                    else:

                        st.warning(
                            "OCR could not extract "
                            "readable text from this plate."
                        )


# ============================================================
# TAB 2 — PRETRAINED VS FINE-TUNED
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">'
        'Pretrained YOLO11n vs Fine-Tuned YOLO11n'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        The pretrained YOLO11n model is a
        general-purpose COCO detector.

        License plates are not one of its
        detection classes.

        The fine-tuned model was adapted specifically
        for the `License_Plate` class.
        """
    )

    comparison_file = st.file_uploader(
        "Upload an image for model comparison",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="comparison_upload"
    )

    if comparison_file is not None:

        comparison_pil = Image.open(
            comparison_file
        ).convert("RGB")

        comparison_rgb = np.array(
            comparison_pil
        )

        comparison_bgr = cv2.cvtColor(
            comparison_rgb,
            cv2.COLOR_RGB2BGR
        )

        with st.spinner(
            "Running both models..."
        ):

            baseline_result = (
                run_baseline(
                    comparison_bgr,
                    baseline_model
                )
            )

            fine_tuned_results = detector(
                comparison_bgr,
                conf=CONFIDENCE_THRESHOLD,
                device=DEVICE,
                verbose=False
            )

            fine_tuned_result = (
                fine_tuned_results[0]
            )

        # ----------------------------------------------------
        # Baseline annotation
        # ----------------------------------------------------

        baseline_annotated = (
            baseline_result.plot()
        )

        # ----------------------------------------------------
        # Fine-tuned annotation
        # ----------------------------------------------------

        fine_tuned_annotated = (
            draw_detections(
                comparison_bgr,
                fine_tuned_result
            )
        )

        baseline_rgb = cv2.cvtColor(
            baseline_annotated,
            cv2.COLOR_BGR2RGB
        )

        fine_tuned_rgb = cv2.cvtColor(
            fine_tuned_annotated,
            cv2.COLOR_BGR2RGB
        )

        col1, col2 = st.columns(
            2
        )

        # ====================================================
        # PRETRAINED
        # ====================================================

        with col1:

            st.subheader(
                "Pretrained YOLO11n"
            )

            st.image(
                baseline_rgb,
                use_container_width=True
            )

            baseline_count = len(
                baseline_result.boxes
            )

            st.write(
                f"**Objects detected:** "
                f"{baseline_count}"
            )

            if baseline_count > 0:

                st.write(
                    "**Detected objects:**"
                )

                for box in (
                    baseline_result.boxes
                ):

                    class_id = int(
                        box.cls[0]
                    )

                    confidence = float(
                        box.conf[0]
                    )

                    class_name = (
                        baseline_model.names[
                            class_id
                        ]
                    )

                    st.write(
                        f"• {class_name} — "
                        f"{confidence * 100:.1f}%"
                    )

            st.caption(
                "General-purpose COCO detector"
            )

            st.info(
                "The pretrained model does not "
                "contain a License_Plate class."
            )

        # ====================================================
        # FINE-TUNED
        # ====================================================

        with col2:

            st.subheader(
                "Fine-Tuned YOLO11n"
            )

            st.image(
                fine_tuned_rgb,
                use_container_width=True
            )

            fine_tuned_count = len(
                fine_tuned_result.boxes
            )

            st.write(
                f"**License plates detected:** "
                f"{fine_tuned_count}"
            )

            if fine_tuned_count > 0:

                st.write(
                    "**License plates:**"
                )

                for box in (
                    fine_tuned_result.boxes
                ):

                    confidence = float(
                        box.conf[0]
                    )

                    st.write(
                        f"• License_Plate — "
                        f"{confidence * 100:.1f}%"
                    )

            st.caption(
                "Specialized for license plate detection"
            )

    else:

        st.info(
            "Upload an image above to compare "
            "the pretrained and fine-tuned models."
        )


# ============================================================
# TAB 3 — MODEL EVALUATION
# ============================================================

with tab3:

    st.markdown(
        '<div class="section-title">'
        'Fine-Tuned Model — Test Performance'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"""
        Evaluation was performed on a held-out test set
        containing **{TEST_IMAGES:,} images** and
        **{TEST_INSTANCES:,} annotated license plates**.
        """
    )

    col1, col2, col3, col4 = st.columns(
        4
    )

    with col1:

        st.metric(
            "mAP@50",
            f"{MAP50 * 100:.2f}%"
        )

    with col2:

        st.metric(
            "mAP@50–95",
            f"{MAP50_95 * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Precision",
            f"{PRECISION * 100:.2f}%"
        )

    with col4:

        st.metric(
            "Recall",
            f"{RECALL * 100:.2f}%"
        )

    st.divider()

    st.success(
        "Key takeaway: Fine-tuning adapted the "
        "general-purpose YOLO11n detector to "
        "recognize License_Plate as a "
        "domain-specific object."
    )

    st.markdown(
        """
        ### What this project demonstrates

        - Transfer learning from a pretrained YOLO11n model
        - Domain-specific object detection
        - Dataset preparation and YOLO annotation format
        - GPU-based model training
        - Quantitative model evaluation
        - Real-time inference
        - OCR integration as a downstream component
        - Interactive deployment with Streamlit
        """
    )

    st.caption(
        "Prototype built to demonstrate domain-specific "
        "computer-vision model fine-tuning, evaluation, "
        "and deployment."
    )