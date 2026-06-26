import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 128

model = load_model("model/deepfake_model.h5")
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# -------- IMAGE --------
def predict_image(path):
    img = cv2.imread(path)

    if img is None:
        return "Invalid Image ❌"

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0]

    if pred[0] > pred[1]:
        return f"REAL ✅ ({pred[0]:.2f})"
    else:
        return f"FAKE ❌ ({pred[1]:.2f})"


# -------- VIDEO --------
def predict_video(path):
    cap = cv2.VideoCapture(path)

    fake_scores = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 10 != 0:
            frame_count += 1
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)

        pred = model.predict(frame, verbose=0)[0]
        fake_scores.append(pred[1])

        frame_count += 1

    cap.release()

    if len(fake_scores) == 0:
        return "INVALID VIDEO ❌"

    avg_fake = np.mean(fake_scores)

    if avg_fake > 0.6:
        return f"FAKE VIDEO ❌ ({avg_fake:.2f})"
    else:
        return f"REAL VIDEO ✅ ({1 - avg_fake:.2f})"