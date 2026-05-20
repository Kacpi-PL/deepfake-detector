import io
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from torchvision.transforms import v2
from torchvision.models import resnet50
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
FACE_DETECTOR_MODEL_PATH = "blaze_face_short_range.tflite"

base_options = python.BaseOptions(
    model_asset_path=FACE_DETECTOR_MODEL_PATH
)

face_detector_options = vision.FaceDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    min_detection_confidence=0.5,
)

face_detector = vision.FaceDetector.create_from_options(face_detector_options)
app = FastAPI(
    title="Deepfake Detector API",
    description="Faces Classification API - Real vs StyleGAN3",
)

def crop_face_with_mediapipe_tasks(
    image: Image.Image,
    margin_left: float = 0.25,
    margin_right: float = 0.25,
    margin_top: float = 0.45,
    margin_bottom: float = 0.20,
) -> Image.Image:


    image_np = np.array(image.convert("RGB"))
    height, width, _ = image_np.shape

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_np,
    )

    detection_result = face_detector.detect(mp_image)

    if not detection_result.detections:
        return image

    detection = max(
        detection_result.detections,
        key=lambda det: det.categories[0].score,
    )

    bbox = detection.bounding_box

    x1 = bbox.origin_x
    y1 = bbox.origin_y
    x2 = bbox.origin_x + bbox.width
    y2 = bbox.origin_y + bbox.height

    box_w = x2 - x1
    box_h = y2 - y1

    margin_x_left = int(box_w * margin_left)
    margin_x_right = int(box_w * margin_right)
    margin_y_top = int(box_h * margin_top)
    margin_y_bottom = int(box_h * margin_bottom)

    x1 = max(0, x1 - margin_x_left)
    y1 = max(0, y1 - margin_y_top)
    x2 = min(width, x2 + margin_x_right)
    y2 = min(height, y2 + margin_y_bottom)

    if x2 <= x1 or y2 <= y1:
        return image

    return image.crop((x1, y1, x2, y2))

def build_binary_resnet() -> nn.Module:
    model = resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(num_features, 2))
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_binary_resnet()


WEIGHTS_PATH = "../models/resnet50_deepfake_weights.pth"

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()

preprocess = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

CLASS_NAMES = {0: "StyleGAN3 (Fake)", 1: "Real"}


# Main API Endpoint
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # MediaPipe face detection + face crop
        image = crop_face_with_mediapipe_tasks(image, margin_left=0.25,
    margin_right=0.25,
    margin_top=0.45,
    margin_bottom=0.20,)

        # DEBUG: saving cropped face
        #image.save("debug_face_crop.jpg")

        # Preprocessing
        input_tensor = preprocess(image).unsqueeze(0).to(device)

        # Inference
        with torch.inference_mode():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()

        return {
            "filename": file.filename,
            "prediction": CLASS_NAMES[predicted_idx],
            "confidence": f"{confidence * 100:.2f}%",
        }

    except Exception as e:
        return {"error": f"Wystąpił błąd podczas przetwarzania obrazu: {str(e)}"}