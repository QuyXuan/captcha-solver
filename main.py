import customtkinter as ctk
import tkinter as tk
from tkinter import PhotoImage
from tkinter import filedialog
from solve import generate_captcha, load_model, predict_captcha
import os
from PIL import Image, ImageTk


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

root = ctk.CTk()
root.geometry("800x500")
root.title("Captcha Solver")

global model
model = load_model()


def on_closing():
    if os.path.exists("captcha_out.png"):
        os.remove("captcha_out.png")
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)


def update_image(label, button, image_path):
    if os.path.exists(image_path):
        # photo = PhotoImage(file=image_path)
        # label.configure(image=photo)
        # label.image = photo
        image = Image.open(image_path)
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo)
        label.image = photo

        button.pack(pady=12, padx=10)
        button.configure(command=lambda: predict(image_path))
        entry2.pack(pady=12, padx=10)


def open_file():
    file_path = filedialog.askopenfilename(
        initialdir="/", title="Select file", filetypes=(("jpg files", "*.jpg"),)
    )
    if file_path:
        update_image(img_label, button_predict, file_path)


def generate():
    captcha_text = entry1.get()
    generate_captcha(captcha_text)
    update_image(img_label, button_predict, "captcha_out.png")


def predict(file_path):
    captcha_text = predict_captcha(model, file_path)
    entry2.delete(0, "end")
    entry2.insert(0, captcha_text)


frame = ctk.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

label = ctk.CTkLabel(master=frame, text="Generate Captcha", font=("Roboto", 24))
label.pack(pady=12, padx=10)

entry1 = ctk.CTkEntry(
    master=frame, placeholder_text="Enter your captcha", font=("Roboto", 12)
)
# entry1.pack(pady=12, padx=10)
entry1.pack_forget()

button_generate = ctk.CTkButton(
    master=frame, text="Generate", font=("Roboto", 12), command=generate
)
# button_generate.pack(pady=12, padx=10)
button_generate.pack_forget()

button_upload = ctk.CTkButton(
    master=frame, text="Upload", font=("Roboto", 12), command=open_file
)
button_upload.pack(pady=12, padx=10)

img_label = ctk.CTkLabel(master=frame, text="")
img_label.pack(pady=12, padx=10)

button_predict = ctk.CTkButton(master=frame, text="Predict", font=("Roboto", 12))
button_predict.pack_forget()

entry2 = ctk.CTkEntry(master=frame, font=("Roboto", 12))
entry2.pack_forget()

root.mainloop()
