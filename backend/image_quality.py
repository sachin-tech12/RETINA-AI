
import cv2
import numpy as np


def assess_image_quality(image_bytes):
    # Convert uploaded image bytes into an OpenCV image
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Unable to read the uploaded image")

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Sharpness: Variance of Laplacian
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. Brightness: Mean grayscale intensity
    brightness = np.mean(gray)

    # 3. Contrast: Standard deviation of intensity
    contrast = np.std(gray)

    # Experimental thresholds for prototype testing
    sharpness_ok = sharpness >= 40
    brightness_ok = 50 <= brightness <= 200
    contrast_ok = contrast >= 30

    passed_checks = sum([
        sharpness_ok,
        brightness_ok,
        contrast_ok
    ])

    if passed_checks == 3:
        quality_status = "GOOD"
    elif passed_checks == 2:
        quality_status = "USABLE"
    else:
        quality_status = "REJECT"

    return {
        "quality_status": quality_status,
        "sharpness": round(float(sharpness), 2),
        "brightness": round(float(brightness), 2),
        "contrast": round(float(contrast), 2),
        "checks": {
            "sharpness": "PASS" if sharpness_ok else "FAIL",
            "brightness": "PASS" if brightness_ok else "FAIL",
            "contrast": "PASS" if contrast_ok else "FAIL"
        }
    }
