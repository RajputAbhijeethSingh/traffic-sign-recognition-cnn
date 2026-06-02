import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# LOAD MODEL
model = load_model("model/traffic_model.h5")

classes = [
    "Speed Limit 20", "Speed Limit 30", "Speed Limit 50", "Speed Limit 60",
    "Speed Limit 70", "Speed Limit 80", "End Speed Limit 80",
    "Speed Limit 100", "Speed Limit 120", "No passing"
]

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

def predict_image(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (32, 32))
    img = preprocess(img)
    img = img.reshape(1, 32, 32, 1)

    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    probability = np.max(predictions)

    return classIndex, probability

def upload_image():
    file_path = filedialog.askopenfilename()
    
    if file_path:
        img = Image.open(file_path)
        img = img.resize((250, 250))
        img = ImageTk.PhotoImage(img)

        panel.config(image=img)
        panel.image = img

        classIndex, prob = predict_image(file_path)

        if classIndex < len(classes):
            label = classes[classIndex]
        else:
            label = str(classIndex)

        result_label.config(
            text=f"Prediction: {label}\nConfidence: {round(prob*100,2)}%"
        )

root = tk.Tk()
root.title("Traffic Sign Recognition")
root.geometry("400x500")

btn = tk.Button(root, text="Upload Image", command=upload_image)
btn.pack(pady=20)

panel = tk.Label(root)
panel.pack()

result_label = tk.Label(root, text="")
result_label.pack(pady=20)

root.mainloop()