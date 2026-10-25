
import io
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0


# RETINA AI - Model Configuration

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "retina_efficientnet_b0_weighted.pth"
)

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]


# Image Preprocessing

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load Trained Model

def load_model():

    model = efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        5
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model = model.to(DEVICE)
    model.eval()

    return model


model = load_model()

print("RETINA AI trained model loaded successfully!")


# Predict Retinal Image

def predict_retinal_image(image_bytes):

    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    class_index = predicted_class.item()

    return {
        "predicted_class": class_index,
        "severity": CLASS_NAMES[class_index],
        "confidence": round(
            confidence.item() * 100,
            2
        ),
        "model": "EfficientNet-B0 (Class Weighted)",
        "prediction_type": "experimental",
        "disclaimer": (
            "Academic AI prototype only. "
            "Not validated for clinical diagnosis."
        )
    }
