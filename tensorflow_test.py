import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import tkinter as tk

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_acc:.4f}")


# --- Drawing app ---
CANVAS_SIZE = 280  # Larger canvas to draw on
LINE_WIDTH = 15

class DrawDigitApp:
    def __init__(self, model):
        self.model = model
        self.window = tk.Tk()
        self.window.title("Draw a Digit")

        self.canvas = tk.Canvas(self.window, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='white')
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)

        button_frame = tk.Frame(self.window)
        button_frame.pack()

        tk.Button(button_frame, text="Predict", command=self.predict_digit).pack(side=tk.LEFT)
        tk.Button(button_frame, text="Clear", command=self.clear_canvas).pack(side=tk.LEFT)

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        self.draw_image = ImageDraw.Draw(self.image)

        self.window.mainloop()

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x - LINE_WIDTH, y - LINE_WIDTH, x + LINE_WIDTH, y + LINE_WIDTH, fill='black')
        self.draw_image.ellipse([x - LINE_WIDTH, y - LINE_WIDTH, x + LINE_WIDTH, y + LINE_WIDTH], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_image.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=255)

    def predict_digit(self):
        # Resize to 28x28 and invert (white background -> black)
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)

        # Normalize and reshape for prediction
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28)

        prediction = self.model.predict(img_array)
        digit = np.argmax(prediction)

        print(f"Predicted Digit: {digit}")

# Launch the app
DrawDigitApp(model)