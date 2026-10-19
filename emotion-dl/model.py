"""Model definition + shared helpers (loading, preprocessing, face detection)."""
import cv2
import numpy as np
import torch
import torch.nn as nn

IMG_SIZE = 48
# Same alphabetical order torchvision's ImageFolder uses for the FER-2013 folders.
DEFAULT_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.25),
    )


class EmotionCNN(nn.Module):
    """VGG-style CNN for 48x48 grayscale faces (48 -> 24 -> 12 -> 6 -> 3)."""

    def __init__(self, num_classes=7, width=64):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, width),
            conv_block(width, width * 2),
            conv_block(width * 2, width * 4),
            conv_block(width * 4, width * 8),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path="checkpoints/best.pt", device=None):
    """Returns (model, class_names)."""
    device = device or get_device()
    ckpt = torch.load(path, map_location=device)
    classes = ckpt.get("classes", DEFAULT_CLASSES)
    model = EmotionCNN(num_classes=len(classes), width=ckpt.get("width", 64)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, classes


def preprocess_face(gray_face):
    """gray_face: 2-D uint8 array (any size) -> tensor [1,1,48,48] normalised to [-1,1]."""
    face = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    face = face.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5
    return torch.from_numpy(face).unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def predict_probs(model, gray_face, device=None, fear_boost=1.0, classes=None):
    """Returns a numpy array of class probabilities for one grayscale face crop."""
    device = device or next(model.parameters()).device
    logits = model(preprocess_face(gray_face).to(device))
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    if fear_boost != 1.0 and classes is not None and "fear" in classes:
        fear_idx = classes.index("fear")
        probs[fear_idx] *= fear_boost
        probs = probs / probs.sum()
    return probs


def get_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_faces(detector, gray_img):
    return detector.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
    )


def generate_gradcam(model, gray_face, target_class=None, device=None):
    """Computes Grad-CAM attention heatmap for a given face crop.
    
    Returns:
        overlay: RGB uint8 image with heatmap blended over face
        cam_norm: 2D float array in [0, 1]
        target_class: index of the target class
    """
    device = device or next(model.parameters()).device
    tensor = preprocess_face(gray_face).to(device)
    tensor.requires_grad = True

    # Target the last convolutional layer in the 4th block
    target_layer = model.features[3][3]
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    logits = model(tensor)
    if target_class is None:
        target_class = int(logits.argmax(dim=1).item())

    model.zero_grad()
    score = logits[0, target_class]
    score.backward()

    h1.remove()
    h2.remove()

    if not activations or not gradients:
        base = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2RGB) if len(gray_face.shape) == 2 else gray_face
        return base, np.zeros((48, 48), dtype=np.float32), target_class

    act = activations[0].detach()
    grad = gradients[0].detach()

    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * act, dim=1, keepdim=True)
    cam = torch.relu(cam).squeeze().cpu().numpy()

    h, w = gray_face.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    cam_min, cam_max = cam_resized.min(), cam_resized.max()
    if cam_max > cam_min:
        cam_norm = (cam_resized - cam_min) / (cam_max - cam_min)
    else:
        cam_norm = np.zeros_like(cam_resized)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_norm), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    base_rgb = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2RGB) if len(gray_face.shape) == 2 else gray_face
    overlay = cv2.addWeighted(base_rgb, 0.6, heatmap_rgb, 0.4, 0)
    return overlay, cam_norm, target_class


def compute_valence_metrics(probs, classes):
    """Computes emotional valence, arousal, and mood categorization from predicted probabilities."""
    prob_dict = {c: float(p) for c, p in zip(classes, probs)}
    positive = prob_dict.get("happy", 0.0) + 0.5 * prob_dict.get("surprise", 0.0)
    negative = (
        prob_dict.get("angry", 0.0)
        + prob_dict.get("disgust", 0.0)
        + prob_dict.get("fear", 0.0)
        + prob_dict.get("sad", 0.0)
    )
    neutral = prob_dict.get("neutral", 0.0) + 0.5 * prob_dict.get("surprise", 0.0)

    valence_score = positive - negative
    arousal_score = (
        prob_dict.get("surprise", 0.0)
        + prob_dict.get("fear", 0.0)
        + prob_dict.get("angry", 0.0)
        + prob_dict.get("happy", 0.0)
    )

    if valence_score > 0.15:
        dominant_mood = "Positive 😊"
    elif valence_score < -0.15:
        dominant_mood = "Negative 😔"
    else:
        dominant_mood = "Neutral 😐"

    return {
        "valence_score": round(float(valence_score), 3),
        "arousal_score": round(float(arousal_score), 3),
        "dominant_mood": dominant_mood,
        "positive_ratio": round(float(positive), 3),
        "negative_ratio": round(float(negative), 3),
        "neutral_ratio": round(float(neutral), 3),
    }

