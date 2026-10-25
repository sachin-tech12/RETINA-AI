
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import io

from image_quality import assess_image_quality
from image_enhancement import enhance_retinal_image
from predict_model import predict_retinal_image
from gradcam import generate_gradcam


# =====================================
# RETINA AI - FastAPI Backend
# =====================================

app = FastAPI(title="RETINA AI Backend")


# =====================================
# Frontend CORS Configuration
# =====================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================
# Home Endpoint
# =====================================

@app.get("/")
def home():
    return {
        "message": "RETINA AI Backend is running successfully!"
    }


# =====================================
# Health Check Endpoint
# =====================================

@app.get("/api/health")
def health():
    return {
        "status": "online",
        "project": "RETINA AI",
        "ai_model": "EfficientNet-B0 (Class Weighted)",
        "model_status": "connected",
        "prediction_type": "experimental",
        "gradcam_status": "connected"
    }


# =====================================
# Retinal Image Upload + AI + Grad-CAM
# =====================================

@app.post("/api/upload")
async def upload_retinal_image(
    file: UploadFile = File(...)
):

    # Validate file type
    if file.content_type not in [
        "image/jpeg",
        "image/png"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are allowed"
        )

    # Read uploaded image
    image_bytes = await file.read()

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Image must be smaller than 10 MB"
        )

    # Validate image content
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()

            if image.format not in ["JPEG", "PNG"]:
                raise ValueError("Unsupported image format")

            image_width = image.width
            image_height = image.height
            image_format = image.format

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid JPG or PNG image"
        )

    # Step 1 - Image quality assessment
    quality_result = assess_image_quality(
        image_bytes
    )

    # Step 2 - Image enhancement
    enhancement_result = enhance_retinal_image(
        image_bytes
    )

    # Step 3 - Real trained AI prediction
    ai_result = predict_retinal_image(
        image_bytes
    )

    # Step 4 - Generate real Grad-CAM
    gradcam_result = generate_gradcam(
        image_bytes
    )

    # Step 5 - Return all results to frontend
    return {
        "status": "success",
        "message": "Retinal image processed successfully",
        "filename": file.filename,
        "image_width": image_width,
        "image_height": image_height,
        "format": image_format,
        "image_quality": quality_result,
        "image_enhancement": enhancement_result,
        "ai_prediction": ai_result,
        "gradcam": gradcam_result
    }
