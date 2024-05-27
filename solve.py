from captcha.image import ImageCaptcha
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import numpy as np

CHARACTERS_NUMBER_FULL = [
    "Y",
    "x",
    "8",
    "i",
    "C",
    "5",
    "n",
    "B",
    "P",
    "j",
    "7",
    "X",
    "g",
    "1",
    "U",
    "u",
    "z",
    "4",
    "a",
    "r",
    "0",
    "m",
    "K",
    "Q",
    "k",
    "d",
    "T",
    "N",
    "s",
    "W",
    "b",
    "c",
    "E",
    "G",
    "A",
    "R",
    "H",
    "w",
    "3",
    "M",
    "f",
    "y",
    "L",
    "Z",
    "v",
    "D",
    "p",
    "V",
    "I",
    "t",
    "o",
    "e",
    "J",
    "9",
    "q",
    "6",
    "F",
    "S",
    "l",
    "h",
    "2",
    "O",
]
CHARACTERS_NUMBER = ["5", "6", "7", "2", "0", "8", "3", "4", "1", "9"]
LOWERCASE_NUMBER = [
    "7",
    "r",
    "i",
    "e",
    "2",
    "4",
    "3",
    "8",
    "v",
    "5",
    "z",
    "0",
    "6",
    "9",
    "b",
    "c",
    "y",
    "s",
    "t",
    "j",
    "n",
    "d",
    "l",
    "g",
    "p",
    "o",
    "h",
    "k",
    "x",
    "w",
    "q",
    "m",
    "a",
    "f",
    "1",
    "u",
]

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
    captcha = CleanCaptcha(
        width=140, height=50, fonts=font_paths, font_sizes=(30, 32, 34)
    )
    # captcha = ImageCaptcha(
    #     width=140, height=50, fonts=font_paths, font_sizes=(30, 32, 34)
    # )
    data = captcha.generate(captcha_text)
    captcha.write(captcha_text, "captcha_out.png")


def predict_captcha(model, file_path):
    img = kimage.load_img(file_path, color_mode="grayscale", target_size=(40, 150))
    img_array = kimage.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    y_pred = model.predict(img_array)
    return one_hot_to_label(y_pred.squeeze())


def load_model(type_model):
    if type_model == "number":
        model = tf.keras.models.load_model("./number_model.h5", compile=False)
    elif type_model == "lowercase_letter_number":
        model = tf.keras.models.load_model(
            "./lowercase_letter_number_model.h5", compile=False
        )
    elif type_model == "full_letter_number":
        model = tf.keras.models.load_model(
            "./full_letter_number_model.h5", compile=False
        )
    return model


def predict_list_captcha(model, file_paths):
    results = []
    for file_path in file_paths:
        img = kimage.load_img(file_path, color_mode="grayscale", target_size=(40, 150))
        img_array = kimage.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = tf.image.convert_image_dtype(img_array, tf.float32)
        img_array = tf.image.resize(img_array, [40, 150])
        img_array = tf.transpose(img_array, perm=[0, 2, 1, 3])
        img_array /= 255.0
        predicts = model.predict(img_array)[0]
        pre_label = [np.argmax(pred) for pred in predicts]
        char_to_num = layers.StringLookup(
            vocabulary=list(CHARACTERS_NUMBER_FULL), num_oov_indices=0, mask_token=None
        )

        num_to_char = layers.StringLookup(
            vocabulary=char_to_num.get_vocabulary(),
            mask_token=None,
            num_oov_indices=0,
            invert=True,
        )
        pred_label = (
            tf.strings.reduce_join(num_to_char(pre_label)).numpy().decode("utf-8")
        )
        results.append(
            {
                "file_path": file_path,
                "predicted_label": pred_label,
            }
        )
    return results


def predict_ocr_model(model, type_model, image_path):
    img = kimage.load_img(image_path, color_mode="grayscale", target_size=(40, 150))
    img_array = kimage.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.image.convert_image_dtype(img_array, tf.float32)
    img_array = tf.image.resize(img_array, [40, 150])
    img_array = tf.transpose(img_array, perm=[0, 2, 1, 3])
    img_array /= 255.0

    predicts = model.predict(img_array)[0]
    pre_label = [np.argmax(pred) for pred in predicts]
    char_to_num = layers.StringLookup(
        vocabulary=list(
            CHARACTERS_NUMBER_FULL
            if type_model == "full_letter_number"
            else CHARACTERS_NUMBER if type_model == "number" else LOWERCASE_NUMBER
        ),
        num_oov_indices=0,
        mask_token=None,
    )
    num_to_char = layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(),
        mask_token=None,
        num_oov_indices=0,
        invert=True,
    )
    pred_label = tf.strings.reduce_join(num_to_char(pre_label)).numpy().decode("utf-8")
    return pred_label


def one_hot_to_char(x: np.array):
    y = np.array(x)
    y = y.squeeze()
    assert len(y) == len(CHARACTERS_NUMBER_FULL)
    idx = np.argmax(y)
    return CHARACTERS_NUMBER_FULL[idx]


def one_hot_to_label(x):
    y = np.array(x)
    y = y.squeeze()
    label_list = []
    assert len(y) == len(CHARACTERS_NUMBER_FULL * CHAR_PER_LABEL)
    for i in range(0, CHAR_PER_LABEL):
        start = i * len(CHARACTERS_NUMBER_FULL)
        end = start + len(CHARACTERS_NUMBER_FULL)
        label_list.append(one_hot_to_char(y[start:end]))
    return "".join(label_list)
