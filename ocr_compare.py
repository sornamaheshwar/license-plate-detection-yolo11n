import os
import re

import cv2
import easyocr
import pytesseract
from ultralytics import YOLO
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ============================================================
# CONFIGURATION
# ============================================================

YOLO_MODEL = (
    "runs/detect/runs/license_plate_yolo11n-3/"
    "weights/best.pt"
)

IMAGE_PATH = (
    "dataset/test/images/"
    "00d9db3d2c186504_jpg.rf.5a493e083834aa4b4748f09a073cc200.jpg"
)


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading YOLO...")

detector = YOLO(YOLO_MODEL)

print("Loading EasyOCR...")

easy_reader = easyocr.Reader(
    ["en"],
    gpu=True
)

print("Models loaded.")


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Keep only alphanumeric characters and uppercase them.
    """

    return re.sub(
        r"[^A-Z0-9]",
        "",
        text.upper()
    )


# ============================================================
# PLATE CROP
# ============================================================

def crop_plate(image, box):

    h, w = image.shape[:2]

    x1, y1, x2, y2 = [
        int(value)
        for value in box
    ]

    plate_width = x2 - x1
    plate_height = y2 - y1

    # Small padding
    pad_x = max(
        2,
        int(plate_width * 0.04)
    )

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

    return image[
        y1:y2,
        x1:x2
    ]


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_plate(crop):

    enlarged = cv2.resize(
        crop,
        None,
        fx=6,
        fy=6,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return {
        "color": enlarged,
        "gray": gray,
        "clahe": enhanced
    }


# ============================================================
# EASYOCR
# ============================================================

def run_easyocr(variants):

    candidates = []

    for name, image in variants.items():

        results = easy_reader.readtext(
            image,
            detail=1,
            allowlist=(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789"
            ),
            paragraph=False,
            decoder="beamsearch",
            mag_ratio=1.5,
            text_threshold=0.45,
            low_text=0.25,
            link_threshold=0.25,
            width_ths=0.7,
        )

        for result in results:

            text = normalize_text(
                result[1]
            )

            confidence = float(
                result[2]
            )

            if len(text) >= 3:

                candidates.append(
                    {
                        "text": text,
                        "confidence": confidence,
                        "variant": name
                    }
                )

    if not candidates:
        return None

    # Highest confidence EasyOCR result
    return max(
        candidates,
        key=lambda x: x["confidence"]
    )


# ============================================================
# TESSERACT
# ============================================================

def run_tesseract(variants):

    candidates = []

    # Character whitelist
    config = (
        "--psm 7 "
        "-c tessedit_char_whitelist="
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    for name, image in variants.items():

        text = pytesseract.image_to_string(
            image,
            config=config
        )

        text = normalize_text(
            text
        )

        if len(text) >= 3:

            candidates.append(
                {
                    "text": text,
                    "variant": name
                }
            )

    if not candidates:
        return None

    # Return the longest plausible candidate.
    # Tesseract does not provide a directly comparable
    # confidence value through this simple API.
    return max(
        candidates,
        key=lambda x: len(x["text"])
    )


# ============================================================
# YOLO DETECTION
# ============================================================

print()
print("=" * 65)
print("YOLO + DUAL OCR TEST")
print("=" * 65)

results = detector(
    IMAGE_PATH,
    device=0,
    conf=0.25,
    verbose=False
)

result = results[0]


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    IMAGE_PATH
)

if image is None:

    raise FileNotFoundError(
        IMAGE_PATH
    )


# ============================================================
# PROCESS DETECTED PLATES
# ============================================================

for i, box in enumerate(
    result.boxes
):

    detection_confidence = float(
        box.conf[0]
    )

    print()
    print("=" * 65)
    print(
        f"LICENSE PLATE {i + 1}"
    )
    print("=" * 65)

    print(
        f"YOLO confidence: "
        f"{detection_confidence * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Crop
    # --------------------------------------------------------

    crop = crop_plate(
        image,
        box.xyxy[0]
    )

    if crop.size == 0:

        print(
            "Invalid plate crop."
        )

        continue

    print(
        f"Plate crop: "
        f"{crop.shape[1]} x {crop.shape[0]}"
    )

    # --------------------------------------------------------
    # Save crop
    # --------------------------------------------------------

    os.makedirs(
        "runs/ocr_compare",
        exist_ok=True
    )

    crop_path = (
        f"runs/ocr_compare/"
        f"plate_{i + 1}.png"
    )

    cv2.imwrite(
        crop_path,
        crop
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    variants = preprocess_plate(
        crop
    )

    # --------------------------------------------------------
    # EasyOCR
    # --------------------------------------------------------

    easy_result = run_easyocr(
        variants
    )

    # --------------------------------------------------------
    # Tesseract
    # --------------------------------------------------------

    tesseract_result = run_tesseract(
        variants
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("EASYOCR")

    if easy_result:

        print(
            f"Text: "
            f"{easy_result['text']}"
        )

        print(
            f"Confidence: "
            f"{easy_result['confidence'] * 100:.2f}%"
        )

        print(
            f"Variant: "
            f"{easy_result['variant']}"
        )

    else:

        print(
            "No text detected."
        )

    print()
    print("TESSERACT")

    if tesseract_result:

        print(
            f"Text: "
            f"{tesseract_result['text']}"
        )

        print(
            f"Variant: "
            f"{tesseract_result['variant']}"
        )

    else:

        print(
            "No text detected."
        )

    # ========================================================
    # AGREEMENT
    # ========================================================

    print()
    print("OCR COMPARISON")

    if (
        easy_result
        and tesseract_result
    ):

        easy_text = easy_result[
            "text"
        ]

        tess_text = tesseract_result[
            "text"
        ]

        print(
            f"EasyOCR:    {easy_text}"
        )

        print(
            f"Tesseract:  {tess_text}"
        )

        if easy_text == tess_text:

            print()
            print(
                "RESULT: OCR ENGINES AGREE"
            )

            print(
                f"Candidate text: "
                f"{easy_text}"
            )

        else:

            print()
            print(
                "RESULT: OCR ENGINES DISAGREE"
            )

            print(
                "Do NOT automatically trust "
                "either result."
            )

            print(
                "The plate should be manually verified."
            )

    elif easy_result:

        print(
            "Only EasyOCR returned a result."
        )

    elif tesseract_result:

        print(
            "Only Tesseract returned a result."
        )

    else:

        print(
            "Neither OCR engine returned text."
        )

    print()
    print(
        f"Plate crop saved at: "
        f"{crop_path}"
    )