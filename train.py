import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

DATASET_PATH = "dataset"
IMG_SIZE = 128

data = []
labels = []

print("🔄 Loading images...")

for label, folder in enumerate(["real", "fake"]):
    folder_path = os.path.join(DATASET_PATH, folder)

    for img in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img)

        # ✅ Read image safely
        image = cv2.imread(img_path)

        # 🚨 SKIP bad images
        if image is None:
            print(f"⚠️ Skipped invalid image: {img_path}")
            continue

        try:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            data.append(image)
            labels.append(label)
        except Exception as e:
            print(f"⚠️ Error processing: {img_path}")
            continue


print(f"✅ Total valid images: {len(data)}")

# Convert to numpy
data = np.array(data) / 255.0
labels = to_categorical(labels, 2)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.5),  # ✅ helps accuracy

    Dense(2, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("🚀 Training started...")

model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/deepfake_model.h5")
print("✅ Model trained and saved successfully!")