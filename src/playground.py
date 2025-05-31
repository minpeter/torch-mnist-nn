import tkinter as tk
from tkinter import Canvas, Button, Label, Frame, messagebox
from PIL import Image, ImageDraw, ImageOps, ImageFilter, ImageTk
import numpy as np
import torch
import torchvision.transforms as T
import os
from datetime import datetime

import datasets
from model import NeuralNetwork

# <<<< Settings <<<<
DATA_DIRECTORY = "./data"
CHECKPOINT_DIRECTORY = os.path.join(DATA_DIRECTORY, "my_mnist_checkpoint")
USER_DRAWINGS_DATASET_PATH = os.path.join(DATA_DIRECTORY, "user_drawn_digits.parquet")
DEBUG_IMAGE_SAVE_PATH = os.path.join(
    DATA_DIRECTORY, "debug_playground_processed_input.png"
)

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

CANVAS_SIZE = 280
PEN_THICKNESS = 20
PROCESSED_DISPLAY_SIZE = 84
# >>>> Settings >>>>


# --- 이미지 전처리 함수 (이전과 동일) ---
def preprocess_drawn_image(pil_image):
    img = pil_image.copy()
    img = ImageOps.invert(img)
    bbox = img.getbbox()
    if bbox is None:
        blank_pil = Image.new("L", (28, 28), "black")
        blank_tensor = T.ToTensor()(blank_pil)
        normalize_transform = T.Normalize(MNIST_MEAN, MNIST_STD)
        blank_tensor_normalized = normalize_transform(blank_tensor)
        return blank_tensor_normalized.view(1, -1), blank_pil
    digit_cropped = img.crop(bbox)
    crop_width, crop_height = digit_cropped.size
    target_side_length = 20
    if crop_width > crop_height:
        new_width = target_side_length
        new_height = int(crop_height * (target_side_length / crop_width))
    else:
        new_height = target_side_length
        new_width = int(crop_width * (target_side_length / crop_height))
    new_width = max(1, new_width)
    new_height = max(1, new_height)
    digit_resized = digit_cropped.resize(
        (new_width, new_height), Image.Resampling.LANCZOS
    )
    final_img_pil = Image.new("L", (28, 28), "black")
    paste_x = (28 - new_width) // 2
    paste_y = (28 - new_height) // 2
    final_img_pil.paste(digit_resized, (paste_x, paste_y))
    try:
        os.makedirs(os.path.dirname(DEBUG_IMAGE_SAVE_PATH), exist_ok=True)
        final_img_pil.save(DEBUG_IMAGE_SAVE_PATH)
    except Exception as e:
        print(f"Debug: Error saving final preprocessed PIL image: {e}")
    img_tensor = T.ToTensor()(final_img_pil)
    img_tensor = T.Normalize(MNIST_MEAN, MNIST_STD)(img_tensor)
    img_tensor = img_tensor.view(1, -1)
    return img_tensor, final_img_pil


# --- GUI 애플리케이션 클래스 (UI 흐름 및 레이아웃 대폭 수정) ---
class DigitRecognizerApp:
    STATE_DRAWING = "drawing"
    STATE_LABELING = "labeling"

    def __init__(self, master_window, model_instance, device_instance):
        self.master = master_window
        self.model = model_instance
        self.device = device_instance
        if self.model:
            self.model.eval()

        self.master.title("MNIST Playground")  # 더 간결한 제목
        # self.master.geometry("450x580") # 창 크기 자동 조절에 맡기거나 최소 크기 설정

        self.collected_data = []
        self.current_processed_pil_for_saving = None
        self.current_state = None

        # --- 상단 프레임: 그림판과 (전처리된 이미지 + 예측 결과) ---
        top_display_frame = Frame(self.master)
        top_display_frame.pack(pady=10, padx=10, fill="x", anchor="n")

        # 그림판 프레임 (왼쪽)
        drawing_frame = Frame(top_display_frame)
        drawing_frame.pack(side=tk.LEFT, padx=(0, 15), anchor="n")  # 오른쪽 여백
        Label(
            drawing_frame,
            text=f"Draw Digit ({CANVAS_SIZE}x{CANVAS_SIZE}):",
            font=("Arial", 10),
        ).pack(pady=(0, 5))
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        self.canvas = Canvas(
            drawing_frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="white",
            cursor="cross",
            borderwidth=2,
            relief="groove",
        )
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_pos)

        # 전처리 이미지 및 예측 결과 표시 프레임 (오른쪽)
        processed_info_frame = Frame(top_display_frame)
        processed_info_frame.pack(side=tk.LEFT, anchor="n", fill=tk.BOTH, expand=True)
        Label(
            processed_info_frame,
            text=f"Processed ({PROCESSED_DISPLAY_SIZE}x{PROCESSED_DISPLAY_SIZE}):",
            font=("Arial", 10),
        ).pack(pady=(0, 5))
        self.processed_photo_image = None
        self.processed_image_label = Label(
            processed_info_frame, borderwidth=2, relief="groove"
        )
        self.processed_image_label.pack()

        self.prediction_label = Label(
            processed_info_frame,
            text="Prediction will appear here.",
            font=("Arial", 18, "bold"),
            height=3,
            justify=tk.CENTER,
            wraplength=PROCESSED_DISPLAY_SIZE + 20,
        )
        self.prediction_label.pack(pady=(10, 0), fill="x", expand=True)

        # --- 하단 버튼 프레임들 (상황에 따라 하나만 보임) ---
        # 액션 버튼 프레임 ("Predict", "Clear")
        self.action_buttons_frame = Frame(self.master)
        # pack()은 go_to_state_drawing에서 호출
        self.predict_button = Button(
            self.action_buttons_frame,
            text="Predict Digit",
            command=self.predict_digit,
            width=12,
            height=2,
            font=("Arial", 10, "bold"),
        )
        self.predict_button.pack(side=tk.LEFT, padx=5)
        self.clear_button = Button(
            self.action_buttons_frame,
            text="Clear Canvas",
            command=self.clear_canvas_and_reset_state,
            width=12,
            height=2,
            font=("Arial", 10, "bold"),
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # 레이블링 섹션 프레임 (숫자 버튼, "Skip")
        self.labeling_section_frame = Frame(self.master)
        # pack()은 go_to_state_labeling에서 호출
        self.labeling_instruction_label = Label(
            self.labeling_section_frame,
            text="Correct label? Click digit to SAVE, or SKIP:",
            font=("Arial", 10),
        )
        self.labeling_instruction_label.pack(pady=(5, 0))

        self.digit_buttons_frame = Frame(self.labeling_section_frame)
        self.digit_buttons_frame.pack(pady=5)
        self.digit_buttons = []
        for i in range(10):
            button = Button(
                self.digit_buttons_frame,
                text=str(i),
                width=3,
                height=1,
                font=("Arial", 10, "bold"),
                command=lambda digit=i: self.save_labeled_drawing(digit),
            )
            button.grid(row=i // 5, column=i % 5, padx=2, pady=2)
            self.digit_buttons.append(button)

        self.skip_button = Button(
            self.labeling_section_frame,
            text="Skip & Next Drawing",
            command=self.skip_drawing,
            width=20,
            height=2,
            font=("Arial", 10, "bold"),
        )
        self.skip_button.pack(pady=5)

        self.last_x, self.last_y = None, None
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.go_to_state_drawing()  # 초기 상태 설정

    def _clear_drawing_canvas_and_pil(self):
        self.canvas.delete("all")
        self.pil_draw.rectangle(
            [0, 0, CANVAS_SIZE, CANVAS_SIZE], fill="white", outline="white"
        )
        self.last_x, self.last_y = None, None
        self.current_processed_pil_for_saving = None

    def go_to_state_drawing(self):
        self.current_state = self.STATE_DRAWING
        self._clear_drawing_canvas_and_pil()
        self._update_processed_image_display(Image.new("L", (28, 28), "lightgray"))
        self.prediction_label.config(text="Draw a digit and click 'Predict'.")

        self.labeling_section_frame.pack_forget()  # 레이블링 UI 숨김
        self.action_buttons_frame.pack(pady=10)  # 예측/클리어 버튼 UI 보임
        self.predict_button.config(state=tk.NORMAL)
        self.clear_button.config(state=tk.NORMAL)
        print("UI State: Drawing")

    def go_to_state_labeling(self):
        self.current_state = self.STATE_LABELING
        # 예측 결과는 predict_digit에서 이미 prediction_label에 설정됨

        self.action_buttons_frame.pack_forget()  # 예측/클리어 버튼 UI 숨김
        self.labeling_section_frame.pack(pady=10)  # 레이블링 UI 보임
        for btn in self.digit_buttons:
            btn.config(state=tk.NORMAL)
        self.skip_button.config(state=tk.NORMAL)
        print("UI State: Labeling")

    def _update_processed_image_display(self, pil_img_28x28):  # 이전과 동일
        if pil_img_28x28:
            upscaled_pil = pil_img_28x28.resize(
                (PROCESSED_DISPLAY_SIZE, PROCESSED_DISPLAY_SIZE),
                Image.Resampling.NEAREST,
            )
            self.processed_photo_image = ImageTk.PhotoImage(upscaled_pil)
            self.processed_image_label.config(image=self.processed_photo_image)
        else:
            blank_pil = Image.new(
                "L", (PROCESSED_DISPLAY_SIZE, PROCESSED_DISPLAY_SIZE), "lightgray"
            )
            self.processed_photo_image = ImageTk.PhotoImage(blank_pil)
            self.processed_image_label.config(image=self.processed_photo_image)

    def paint_on_canvas(self, event):  # 이전과 동일
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x,
                self.last_y,
                x,
                y,
                width=PEN_THICKNESS,
                fill="black",
                capstyle=tk.ROUND,
                smooth=tk.TRUE,
            )
            self.pil_draw.line(
                [(self.last_x, self.last_y), (x, y)], fill="black", width=PEN_THICKNESS
            )
        self.last_x = x
        self.last_y = y

    def reset_last_pos(self, event):  # 이전과 동일
        self.last_x, self.last_y = None, None

    def clear_canvas_and_reset_state(self):
        print("Clear button clicked. Resetting to drawing state.")
        self.go_to_state_drawing()

    def predict_digit(self):
        if self.current_state != self.STATE_DRAWING:
            return
        if self.model is None:
            self.prediction_label.config(text="Model not loaded!")
            return
        if self.pil_image.getbbox() is None:
            self.prediction_label.config(text="Canvas is empty. Please draw.")
            self._update_processed_image_display(Image.new("L", (28, 28), "lightgray"))
            self.current_processed_pil_for_saving = None
            return

        pil_for_preprocessing = self.pil_image.copy()
        try:
            img_tensor, final_pil_for_display = preprocess_drawn_image(
                pil_for_preprocessing
            )
            img_tensor = img_tensor.to(self.device)

            if final_pil_for_display:
                self._update_processed_image_display(final_pil_for_display)
                self.current_processed_pil_for_saving = final_pil_for_display
            else:
                self._update_processed_image_display(
                    Image.new("L", (28, 28), "lightgray")
                )
                self.current_processed_pil_for_saving = None
                self.prediction_label.config(text="Could not process drawing.")
                # 여기서 상태 변경 없이 현재 상태(드로잉) 유지
                return
        except Exception as e:
            self.prediction_label.config(text=f"Processing error: {e}")
            print(f"Error during preprocessing: {e}")
            self._update_processed_image_display(Image.new("L", (28, 28), "red"))
            self.current_processed_pil_for_saving = None
            # 여기서 상태 변경 없이 현재 상태(드로잉) 유지
            return

        try:
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
            result_text = f"Prediction: {predicted_class.item()}\n(Confidence: {confidence.item() * 100:.1f}%)"
            self.prediction_label.config(text=result_text)  # 예측 결과 업데이트
            print(f"Prediction made: {result_text.replace('\n', ' ')}")
            self.go_to_state_labeling()  # 예측 성공 후 레이블링 상태로 전환
        except Exception as e:
            self.prediction_label.config(text=f"Prediction error: {e}")
            print(f"Error during prediction: {e}")
            # 여기서 상태 변경 없이 현재 상태(드로잉) 유지

    def save_labeled_drawing(self, digit_label_int):
        if self.current_state != self.STATE_LABELING:
            return
        if self.current_processed_pil_for_saving:
            self.collected_data.append(
                {
                    "image": self.current_processed_pil_for_saving,
                    "label": digit_label_int,
                }
            )
            print(
                f"Drawing labeled as {digit_label_int} and queued. Total: {len(self.collected_data)}"
            )
            # 팝업 제거, 다음 상태로 바로 전환
            self.go_to_state_drawing()
        else:
            # 이 메시지는 일반적으로 보이지 않아야 함 (버튼이 비활성화되므로)
            # messagebox.showwarning("Error", "No processed image available to save.")
            print("Error: No processed image to save, but save button was clicked.")
            self.go_to_state_drawing()  # 안전하게 초기 상태로

    def skip_drawing(self):
        if self.current_state != self.STATE_LABELING:
            return
        print("Drawing skipped. Resetting for next input.")
        self.go_to_state_drawing()

    def on_closing(self):  # 이전과 동일 (경로 부분만 DATA_DIRECTORY 사용)
        if not self.collected_data:
            if messagebox.askokcancel(
                "Quit", "No drawings were labeled for saving. Quit anyway?"
            ):
                self.master.destroy()
            return

        if messagebox.askokcancel(
            "Quit & Save",
            f"You have {len(self.collected_data)} labeled drawings. "
            f"Save them to '{os.path.basename(USER_DRAWINGS_DATASET_PATH)}' in '{DATA_DIRECTORY}' and quit?",
        ):
            if self.collected_data:
                os.makedirs(os.path.dirname(USER_DRAWINGS_DATASET_PATH), exist_ok=True)
                features = datasets.Features(
                    {"image": datasets.Image(), "label": datasets.Value("int8")}
                )
                new_data_dict = {"image": [], "label": []}
                for item in self.collected_data:
                    new_data_dict["image"].append(item["image"])
                    new_data_dict["label"].append(item["label"])

                if not new_data_dict["image"]:
                    print("No new valid data to save after processing collected items.")
                    self.master.destroy()
                    return

                new_dataset_chunk = datasets.Dataset.from_dict(
                    new_data_dict, features=features
                )
                final_dataset_to_save = new_dataset_chunk
                save_message_prefix = f"Saved {len(new_dataset_chunk)}"

                if os.path.exists(USER_DRAWINGS_DATASET_PATH):
                    try:
                        existing_dataset = datasets.load_dataset(
                            "parquet",
                            data_files=USER_DRAWINGS_DATASET_PATH,
                            features=features,
                            split="train",
                        )
                        final_dataset_to_save = datasets.concatenate_datasets(
                            [existing_dataset, new_dataset_chunk]
                        )
                        save_message_prefix = f"Appended {len(new_dataset_chunk)} new (Total: {len(final_dataset_to_save)})"
                    except Exception as e:
                        print(
                            f"Error loading or concatenating existing dataset: {e}. Saving only new data."
                        )

                try:
                    final_dataset_to_save.to_parquet(USER_DRAWINGS_DATASET_PATH)
                    print(
                        f"{save_message_prefix} drawings to '{USER_DRAWINGS_DATASET_PATH}'."
                    )
                    # 팝업 제거
                    # messagebox.showinfo("Saved", f"{save_message_prefix} drawings to '{USER_DRAWINGS_DATASET_PATH}'.")
                except Exception as e:
                    print(f"FATAL: Error saving dataset to Parquet: {e}")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fallback_path = os.path.join(
                        DATA_DIRECTORY,
                        f"user_drawn_digits_fallback_{timestamp}.parquet",
                    )
                    try:
                        new_dataset_chunk.to_parquet(fallback_path)
                        messagebox.showerror(
                            "Save Error",
                            f"Error saving to main file. New data saved to '{fallback_path}'.\nError: {e}",
                        )
                    except Exception as e_fallback:
                        messagebox.showerror(
                            "Fatal Save Error",
                            f"Could not save data to main or fallback file.\nError: {e_fallback}",
                        )
            else:
                print("No new drawings were collected to save.")
            self.master.destroy()


# --- 메인 실행 부분 (이전과 동일) ---
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Playground using device: {DEVICE}")
    os.makedirs(DATA_DIRECTORY, exist_ok=True)
    loaded_model = None
    try:
        loaded_model = NeuralNetwork.load_from_checkpoint(
            directory_path=CHECKPOINT_DIRECTORY, device=DEVICE
        )
        print("MNIST Model loaded successfully for playground.")
    except FileNotFoundError:
        message_fnf = (
            f"Checkpoint files not found in '{CHECKPOINT_DIRECTORY}'.\n"
            f"Required: '{NeuralNetwork.DEFAULT_CONFIG_FILENAME}' and '{NeuralNetwork.DEFAULT_WEIGHTS_FILENAME}'.\n"
            "Please run train.py first to create the checkpoint.\nExiting application."
        )
        print(message_fnf)
        try:
            temp_root_fnf = tk.Tk()
            temp_root_fnf.withdraw()
            messagebox.showerror("Model Load Error", message_fnf)
            temp_root_fnf.destroy()
        except tk.TclError:
            pass
        exit()
    except Exception as e:
        message_e = (
            f"Error loading model from '{CHECKPOINT_DIRECTORY}': {e}\n"
            f"Please ensure the model checkpoint is valid.\nExiting application."
        )
        print(message_e)
        try:
            temp_root_e = tk.Tk()
            temp_root_e.withdraw()
            messagebox.showerror("Model Load Error", message_e)
            temp_root_e.destroy()
        except tk.TclError:
            pass
        exit()

    root = tk.Tk()
    app = DigitRecognizerApp(root, loaded_model, DEVICE)
    root.mainloop()
