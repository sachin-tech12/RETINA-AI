
import io
import base64

import cv2
import numpy as np
import torch
from PIL import Image

from predict_model import model, transform, DEVICE, CLASS_NAMES


def generate_gradcam(image_bytes):

    # Read and preprocess the retinal image
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        original = image.resize((224, 224))
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    activations = []
    gradients = []

    # Final convolutional feature layer of EfficientNet-B0
    target_layer = model.features[-1]

    def forward_hook(module, inputs, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)

            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)

            predicted_class = int(output.argmax(dim=1).item())
            confidence = float(
                probabilities[0, predicted_class].item() * 100
            )

            # Backpropagate the predicted class score
            score = output[0, predicted_class]
            score.backward()

        # Calculate Grad-CAM
        activation = activations[0]
        gradient = gradients[0]

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1).squeeze(0)
        cam = torch.relu(cam).cpu().numpy()

        cam = cv2.resize(cam, (224, 224))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        heatmap = np.uint8(255 * cam)

        colored_heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        # Overlay heatmap on retinal image
        original_bgr = cv2.cvtColor(
            np.array(original),
            cv2.COLOR_RGB2BGR
        )

        overlay = cv2.addWeighted(
            original_bgr,
            0.6,
            colored_heatmap,
            0.4,
            0
        )

        # Convert images into base64 for frontend
        def encode_image(image_array):
            success, buffer = cv2.imencode(
                ".png",
                image_array
            )

            if not success:
                raise RuntimeError("Unable to encode Grad-CAM image")

            encoded = base64.b64encode(
                buffer.tobytes()
            ).decode("utf-8")

            return "data:image/png;base64," + encoded

        return {
            "predicted_class": predicted_class,
            "severity": CLASS_NAMES[predicted_class],
            "confidence": round(confidence, 2),
            "heatmap": encode_image(colored_heatmap),
            "overlay": encode_image(overlay),
            "explanation_type": "Grad-CAM",
            "disclaimer": (
                "Experimental explanation only. "
                "Highlighted regions are not confirmed lesions "
                "and are not a clinical diagnosis."
            )
        }

    finally:
        forward_handle.remove()
        backward_handle.remove()
        model.zero_grad(set_to_none=True)
