
import cv2
import numpy as np
import base64


def enhance_retinal_image(image_bytes):
    # Convert uploaded bytes to an OpenCV image
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image")

    # Convert BGR image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Separate lightness and color channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the lightness channel
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l)

    # Merge enhanced lightness with original color channels
    enhanced_lab = cv2.merge((enhanced_l, a, b))

    # Convert back to BGR
    enhanced_image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    # Encode enhanced image as JPEG
    success, buffer = cv2.imencode(
        ".jpg", enhanced_image
    )

    if not success:
        raise ValueError("Image encoding failed")

    # Convert image to Base64 for frontend display
    encoded_image = base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")

    return {
        "status": "enhanced",
        "method": "CLAHE",
        "enhanced_image": (
            "data:image/jpeg;base64," + encoded_image
        )
    }
