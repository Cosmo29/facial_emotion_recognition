"""Live webcam emotion recognition with the trained CNN. Press q to quit."""
import argparse
from collections import deque

import cv2
import numpy as np

from model import detect_faces, get_device, get_face_detector, load_model, predict_probs

COLORS = {"angry": (0, 0, 255), "disgust": (0, 128, 0), "fear": (128, 0, 128), "happy": (0, 200, 0),
          "neutral": (200, 200, 200), "sad": (255, 100, 0), "surprise": (0, 200, 255)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--smooth", type=int, default=5, help="average predictions over N frames")
    args = p.parse_args()

    device = get_device()
    model, classes = load_model(args.ckpt, device)
    detector = get_face_detector()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. Close other apps using it, or try --camera 1")

    history = deque(maxlen=args.smooth)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(detector, gray)

        if len(faces) == 0:
            history.clear()
        for (x, y, w, h) in faces:
            probs = predict_probs(model, gray[y:y + h, x:x + w], device)
            history.append(probs)
            avg = np.mean(history, axis=0)
            k = int(avg.argmax())
            label, color = classes[k], COLORS.get(classes[k], (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {avg[k] * 100:.0f}%", (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Emotion Recognition - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
