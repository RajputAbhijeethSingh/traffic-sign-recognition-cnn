import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# SETTINGS
data_path = "dataset"
img_size = 32
num_classes = 43
images = []
labels = []

print("Loading dataset...")

# LOAD DATASET
for class_id in range(num_classes):
    path = os.path.join(data_path, str(class_id))
    
    if not os.path.exists(path):
        continue
        
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            labels.append(class_id)

images = np.array(images)
labels = np.array(labels)

print("Total images loaded:", len(images))

# PREPROCESS
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

images = np.array([preprocess(img) for img in images])
images = images.reshape(-1, img_size, img_size, 1)

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# MODEL
model = Sequential([
    Conv2D(60, (5,5), activation='relu', input_shape=(32,32,1)),
    Conv2D(60, (5,5), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(30, (3,3), activation='relu'),
    Conv2D(30, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Dropout(0.5),
    Flatten(),
    Dense(500, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Training started...")

model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# SAVE MODEL
if not os.path.exists("model"):
    os.makedirs("model")

model.save("model/traffic_model.h5")

print("✅ Model saved successfully!")