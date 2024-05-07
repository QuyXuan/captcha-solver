from captcha.image import ImageCaptcha
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.models import load_model
import numpy as np

CHARACTERS_FULL = [
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

CHARACTERS_NUMBER = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

CHAR_PER_LABEL = 5


class CleanCaptcha(ImageCaptcha):

    def create_captcha_image(self, chars, color, background):
        """Tạo hình ảnh CAPTCHA mà không có đường kẻ hay nhiễu."""

        image = super().create_captcha_image(chars, color, background)
        return image

    def create_noise_curve(self, image, color):
        """Bỏ qua việc tạo đường cong nhiễu."""
        return image

    def create_noise_dots(self, image, color, number=100):
        """Bỏ qua việc tạo chấm nhiễu."""
        return image


font_paths = [
    r"E:\CaptchaSolver\ARIAL.TTF"
]  # Chuột phải vào file ARIAL.TTF -> Copy Path -> Dán vào trong r""


def generate_captcha(captcha_text):
    # captcha = CleanCaptcha(
    #     width=140, height=50, fonts=font_paths, font_sizes=(30, 32, 34)
    # )
    captcha = ImageCaptcha(
        width=140, height=50, fonts=font_paths, font_sizes=(30, 32, 34)
    )
    data = captcha.generate(captcha_text)
    captcha.write(captcha_text, "captcha_out.png")


def predict_captcha(model, file_path):
    img = kimage.load_img(file_path, color_mode="grayscale", target_size=(40, 150))
    img_array = kimage.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    y_pred = model.predict(img_array)
    return one_hot_to_label(y_pred.squeeze())


def load_model(use_cnn=False):
    if use_cnn:
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
                tf.keras.layers.Dense(len(CHARACTERS_NUMBER) * 5),
            ]
        )
        model.load_weights("./captcha_solver_number_model.weights.h5")
    else:
        model = load_model("./captcha_solver_number_ocr_model.h5")
    return model


def predict_list_captcha(model, file_paths):
    results = []
    for file_path in file_paths:
        img = kimage.load_img(file_path, color_mode="grayscale", target_size=(40, 150))
        img_array = kimage.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        y_pred = model.predict(img_array)
        results.append(
            {
                "file_path": file_path,
                "predicted_label": one_hot_to_label(y_pred.squeeze()),
            }
        )
    return results


def one_hot_to_char(x: np.array):
    y = np.array(x)
    y = y.squeeze()
    assert len(y) == len(CHARACTERS_NUMBER)
    idx = np.argmax(y)
    return CHARACTERS_NUMBER[idx]


def one_hot_to_label(x):
    y = np.array(x)
    y = y.squeeze()
    label_list = []
    assert len(y) == len(CHARACTERS_NUMBER * CHAR_PER_LABEL)
    for i in range(0, CHAR_PER_LABEL):
        start = i * len(CHARACTERS_NUMBER)
        end = start + len(CHARACTERS_NUMBER)
        label_list.append(one_hot_to_char(y[start:end]))
    return "".join(label_list)
