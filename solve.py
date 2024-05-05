from captcha.image import ImageCaptcha
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
import numpy as np

CHARACTERS = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]
CHAR_PER_LABEL = 5


def generate_captcha(text):
    image = ImageCaptcha(width=200, height=100, font_sizes=(50, 60, 70))
    captcha_text = text
    data = image.generate(captcha_text)
    image.write(captcha_text, "captcha_out.png")


def load_model():
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape=(40, 150, 1)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(1, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            #     tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.75),
            tf.keras.layers.Dense(len(CHARACTERS) * 5),
        ]
    )
    model.load_weights("./captcha_solver_model.weights.h5")
    return model


def predict_captcha(model, file_path):
    img = kimage.load_img(file_path, color_mode="grayscale", target_size=(40, 150))
    img_array = kimage.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Thêm một chiều để tạo batch
    img_array = img_array / 255.0  # Chuẩn hóa
    y_pred = model.predict(img_array)
    return one_hot_to_label(y_pred.squeeze())


def one_hot_to_char(x: np.array):
    y = np.array(x)
    y = y.squeeze()
    assert len(y) == len(CHARACTERS)
    idx = np.argmax(y)
    return CHARACTERS[idx]


def one_hot_to_label(x):
    y = np.array(x)
    y = y.squeeze()
    label_list = []
    assert len(y) == len(CHARACTERS * CHAR_PER_LABEL)
    for i in range(0, CHAR_PER_LABEL):
        start = i * len(CHARACTERS)
        end = start + len(CHARACTERS)
        label_list.append(one_hot_to_char(y[start:end]))
    return "".join(label_list)
