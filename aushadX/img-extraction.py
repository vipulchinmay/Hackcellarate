import re
import cv2
import base64
import requests
import numpy as np
import os
from dotenv import load_dotenv  # Import dotenv to load .env file

# Load .env file
load_dotenv()

# Access API Key from .env
API_KEY = os.getenv("API_KEY")

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blur)
    norm = cv2.normalize(clahe, None, 0, 255, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def rotate_image(image_np, angle):
    """Rotate the image by the specified angle."""
    (h, w) = image_np.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def image_to_text(image_np):
    # Encode image to base64 string
    _, buffer = cv2.imencode('.jpg', image_np)
    img_base64 = base64.b64encode(buffer).decode()

    url = f"https://vision.googleapis.com/v1/images:annotate?key={API_KEY}"

    # Prepare the request payload for text detection
    request_payload = {
        "requests": [
            {
                "image": {
                    "content": img_base64
                },
                "features": [
                    {
                        "type": "TEXT_DETECTION"
                    }
                ]
            }
        ]
    }

    response = requests.post(url, json=request_payload)
    response.raise_for_status()

    response_data = response.json()

    # Parse response to get detected text
    try:
        text = response_data['responses'][0]['fullTextAnnotation']['text']
    except (KeyError, IndexError):
        text = ""

    return text

def image_to_text_multiple_orientations(image_np):
    rotations = [0, 90, 180, 270]
    texts = []

    for angle in rotations:
        rotated_img = rotate_image(image_np, angle)
        text = image_to_text(rotated_img)
        texts.append((text, angle))

    # Choose the text with the maximum length (assuming more text means better detection)
    best_text, best_angle = max(texts, key=lambda x: len(x[0]))
    print(f"\n🔄 Best text detected at rotation: {best_angle}°")
    return best_text

def extract_info(text):
    info = {}

    # Extract MFD
    mfd_match = re.search(r'(MFD|Mfg)[^\d]*(\w{3,9}\.?\s?\d{4})', text, re.IGNORECASE)
    if mfd_match:
        info['MFD'] = mfd_match.group(2)

    # Extract EXP
    exp_match = re.search(r'(EXP|Exp)[^\d]*(\w{3,9}\.?\s?\d{4})', text, re.IGNORECASE)
    if exp_match:
        info['EXP'] = exp_match.group(2)

    # Extract MRP
    mrp_match = re.search(r'(MRP|Rs\.?|₹)[^\d]*(\d+(\.\d{1,2})?)', text, re.IGNORECASE)
    if mrp_match:
        info['MRP'] = f"₹{mrp_match.group(2)}"

    # Extract medicine name (first all-caps word of 5+ letters)
    med_name = re.findall(r'\b[A-Z][A-Z0-9\-]{4,}\b', text)
    if med_name:
        info['Medicine Name'] = med_name[0]

    return info

def main():
    if not API_KEY:
        print("❌ Error: API key not found. Make sure you have set it in the .env file.")
        return

    image_path = input("Enter path to medicine image: ").strip()
    preprocessed = preprocess_image(image_path)

    # Use the multiple orientation OCR function
    text = image_to_text_multiple_orientations(preprocessed)

    print("\n🔍 Extracted Text:\n", text)

    info = extract_info(text)
    print("\n📦 Important Extracted Information:")
    for key, value in info.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
