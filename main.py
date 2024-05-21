import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, PhotoImage
from solve import load_model, predict_list_captcha, generate_captcha, predict_ocr_model
import os
from PIL import Image, ImageTk


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

root = ctk.CTk()
root.geometry("1000x700")
root.title("Captcha Solver")

global number_model
global lowercase_letter_number_model
global full_letter_number_model
number_model = load_model("number")
lowercase_letter_number_model = load_model("lowercase_letter_number")
full_letter_number_model = load_model("full_letter_number")
max_images_per_row = 5


def on_closing():
    if os.path.exists("captcha_out.png"):
        os.remove("captcha_out.png")
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)


def generate():
    captcha_text = textbox_captcha.get()
    generate_captcha(captcha_text)
    update_image("captcha_out.png")


def predict(file_path):
    captcha_text = textbox_captcha.get()
    if captcha_text.isnumeric():
        model = number_model
    elif captcha_text.isalpha() and captcha_text.islower():
        model = lowercase_letter_number_model
    else:
        model = full_letter_number_model
    captcha_text = predict_ocr_model(model, file_path)
    textbox_predict.delete(0, "end")
    textbox_predict.insert(0, captcha_text)


def update_image(image_path):
    if os.path.exists(image_path):
        image = Image.open(image_path)
        photo = ImageTk.PhotoImage(image)
        label_image.configure(image=photo)
        label_image.image = photo

        button_predict.pack(pady=12, padx=10)
        button_predict.configure(command=lambda: predict(image_path))
        textbox_predict.pack(pady=12, padx=10)


def update_flow_panel(file_paths, list_predictions=None):
    for widget in flow_frame.winfo_children():
        widget.destroy()

    row = 0
    column = 0
    if list_predictions is None:
        for file_path in file_paths:
            if column >= max_images_per_row:
                row += 1
                column = 0

            image = Image.open(file_path)
            photo = ImageTk.PhotoImage(image)
            label = ctk.CTkLabel(master=flow_frame, text="", image=photo)
            label.image = photo
            label.grid(row=row, column=column, padx=10, pady=10)
            column += 1
        flow_frame.update_idletasks()
        canvas.pack(padx=90, pady=10, fill="both")
        canvas.config(scrollregion=canvas.bbox("all"))
        button_predict.pack(pady=12, padx=10)
        button_predict.configure(command=lambda: handle_prediction(file_paths))
    else:
        for obj in list_predictions:
            if column >= max_images_per_row:
                row += 2
                column = 0

            image = Image.open(obj["file_path"])
            photo = ImageTk.PhotoImage(image)
            label_text = ctk.CTkLabel(master=flow_frame, text=obj["predicted_label"])
            label_text.grid(row=row, column=column, padx=10, pady=5)
            label_image = ctk.CTkLabel(master=flow_frame, text="", image=photo)
            label_image.image = photo
            label_image.grid(row=row + 1, column=column, padx=10, pady=(0, 5))
            column += 1
    flow_frame.update_idletasks()
    canvas.pack(padx=90, pady=10, fill="both")
    canvas.config(scrollregion=canvas.bbox("all"))


def open_file():
    file_paths = filedialog.askopenfilenames(
        initialdir="/",
        title="Select file",
        filetypes=(("image files", "*.jpg *.jpeg *.png"),),
    )
    if file_paths:
        update_flow_panel(file_paths)


def update_canvas_scrollregion(event):
    canvas.configure(scrollregion=canvas.bbox("all"))


def handle_prediction(file_paths):
    results = predict_list_captcha(full_letter_number_model, file_paths)
    for result in results:
        print(
            f"File: {result['file_path']}, Predicted Label: {result['predicted_label']}"
        )
    update_flow_panel(file_paths, results)


frame = ctk.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)
frame.pack_propagate(False)

label_title = ctk.CTkLabel(master=frame, text="Captcha Solver", font=("Roboto", 24))
label_title.pack(pady=12, padx=10)

button_upload = ctk.CTkButton(
    master=frame, text="Upload", font=("Roboto", 12), command=open_file
)
# button_upload.pack(pady=12, padx=10)
button_upload.pack_forget()

textbox_captcha = ctk.CTkEntry(
    master=frame, placeholder_text="Enter your captcha", font=("Roboto", 12)
)
textbox_captcha.pack(pady=12, padx=10)

button_generate = ctk.CTkButton(
    master=frame, text="Generate", font=("Roboto", 12), command=generate
)
button_generate.pack(pady=12, padx=10)

label_image = ctk.CTkLabel(master=frame, text="")
label_image.pack(pady=12, padx=10)

canvas = ctk.CTkCanvas(master=frame, width=900, height=300, bg="#212121")
scrollbar = ctk.CTkScrollbar(master=canvas, command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack_forget()
scrollbar.pack(side="right", fill="y")

flow_frame = ctk.CTkFrame(canvas, width=canvas.winfo_width(), height=200)
flow_frame.bind("<Configure>", update_canvas_scrollregion)
canvas.create_window((0, 0), window=flow_frame, anchor="nw")


button_predict = ctk.CTkButton(master=frame, text="Predict", font=("Roboto", 12))
button_predict.pack_forget()

textbox_predict = ctk.CTkEntry(master=frame, font=("Roboto", 12))
textbox_predict.pack_forget()

root.mainloop()

# Nếu muốn chạy chức năng predict nhiều file cùng 1 lúc
# thì MỞ COMMENT dòng 129 và COMMENT dòng 130 -> 143
# Và ngược lại nếu như muốn sử dụng chức năng generate và predict
# thì COMMENT dòng 129 và MỞ COMMENT dòng 130 -> 143
