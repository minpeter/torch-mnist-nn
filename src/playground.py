import tkinter as tk
from tkinter import Canvas, Button, Label, Frame, messagebox
from PIL import Image, ImageDraw, ImageOps, ImageFilter, ImageTk  # ImageTk 추가

from model import NeuralNetwork
import os
import torch
import torchvision.transforms as T


# <<<< Settings <<<<
CHECKPOINT_DIRECTORY = "./my_mnist_checkpoint"

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

CANVAS_SIZE = 280
PEN_THICKNESS = 20  # 280x280 캔버스 기준 펜 두께
PROCESSED_DISPLAY_SIZE = 84  # 28x28 이미지를 84x84로 확대 표시 (3배)
# >>>> Settings >>>>


# --- 이미지 전처리 함수 (반환값 변경) ---
def preprocess_drawn_image(pil_image):
    img = pil_image.copy()
    img = ImageOps.invert(img)
    bbox = img.getbbox()

    if bbox is None:
        blank_pil = Image.new("L", (28, 28), "black")
        blank_tensor = T.ToTensor()(blank_pil)  # [0,1] 범위의 텐서
        normalize_transform = T.Normalize(MNIST_MEAN, MNIST_STD)
        blank_tensor_normalized = normalize_transform(blank_tensor)
        return blank_tensor_normalized.view(1, -1), blank_pil  # 빈 PIL 이미지도 반환

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

    # Optional: Gaussian blur
    # final_img_pil = final_img_pil.filter(ImageFilter.GaussianBlur(radius=0.5))

    # 디버깅용 파일 저장은 그대로 두거나 필요시 주석 해제
    try:
        final_img_pil.save("debug_playground_processed_input.png")
        # print("Debug: Saved final preprocessed PIL image as debug_playground_processed_input.png")
    except Exception as e:
        print(f"Debug: Error saving final preprocessed PIL image: {e}")

    img_tensor = T.ToTensor()(final_img_pil)
    img_tensor = T.Normalize(MNIST_MEAN, MNIST_STD)(img_tensor)
    img_tensor = img_tensor.view(1, -1)

    return img_tensor, final_img_pil  # 전처리된 텐서와 28x28 PIL 이미지 반환


# --- GUI 애플리케이션 클래스 (수정됨) ---
class DigitRecognizerApp:
    def __init__(self, master_window, model_instance, device_instance):
        self.master = master_window
        self.model = model_instance
        self.device = device_instance
        if self.model:
            self.model.eval()

        self.master.title("MNIST Playground - Draw & Predict")

        # 메인 프레임 (그림판과 전처리 이미지 표시 영역을 옆으로 배치하기 위함)
        main_app_frame = Frame(self.master)
        main_app_frame.pack(pady=10, padx=10)

        # --- 그림판 프레임 (왼쪽) ---
        drawing_frame = Frame(main_app_frame)
        drawing_frame.pack(side=tk.LEFT, padx=10)

        Label(
            drawing_frame,
            text="Draw Digit Here ({}x{}):".format(CANVAS_SIZE, CANVAS_SIZE),
        ).pack(pady=(0, 5))
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        self.canvas = Canvas(
            drawing_frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="white",
            cursor="cross",
        )
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_pos)

        # --- 전처리된 이미지 표시 프레임 (오른쪽) ---
        processed_display_frame = Frame(main_app_frame)
        processed_display_frame.pack(side=tk.LEFT, padx=10, anchor="n")  # anchor 추가

        Label(
            processed_display_frame,
            text="Processed Input ({}x{}):".format(
                PROCESSED_DISPLAY_SIZE, PROCESSED_DISPLAY_SIZE
            ),
        ).pack(pady=(0, 5))
        self.processed_photo_image = None  # PhotoImage 참조 유지용
        self.processed_image_label = Label(
            processed_display_frame, borderwidth=2, relief="solid"
        )

        # 초기 빈 이미지 표시
        self._update_processed_image_display(
            Image.new("L", (28, 28), "gray")
        )  # 초기 회색 이미지
        self.processed_image_label.pack()

        # --- 예측 결과 레이블 (하단 중앙) ---
        self.prediction_label = Label(
            self.master, text="Draw a digit and click 'Predict'.", font=("Arial", 16)
        )
        self.prediction_label.pack(
            pady=10
        )  # pack으로 변경하고 main_app_frame 아래에 배치

        # --- 버튼 프레임 (가장 하단 중앙) ---
        button_frame = Frame(self.master)
        button_frame.pack(pady=10)

        self.predict_button = Button(
            button_frame, text="Predict", command=self.predict_digit, width=15
        )
        self.predict_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = Button(
            button_frame, text="Clear", command=self.clear_canvas, width=15
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.last_x, self.last_y = None, None

    def _update_processed_image_display(self, pil_img_28x28):
        """28x28 PIL 이미지를 받아 확대하여 GUI에 표시합니다."""
        if pil_img_28x28:
            upscaled_pil = pil_img_28x28.resize(
                (PROCESSED_DISPLAY_SIZE, PROCESSED_DISPLAY_SIZE),
                Image.Resampling.NEAREST,  # 픽셀 아트를 확대할 때 좋음
            )
            self.processed_photo_image = ImageTk.PhotoImage(upscaled_pil)
            self.processed_image_label.config(image=self.processed_photo_image)
        else:  # None이 전달된 경우 (예: 빈 캔버스 초기화)
            blank_pil = Image.new(
                "L", (PROCESSED_DISPLAY_SIZE, PROCESSED_DISPLAY_SIZE), "gray"
            )
            self.processed_photo_image = ImageTk.PhotoImage(blank_pil)
            self.processed_image_label.config(image=self.processed_photo_image)

    def paint_on_canvas(self, event):
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

    def reset_last_pos(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.pil_draw.rectangle(
            [0, 0, CANVAS_SIZE, CANVAS_SIZE], fill="white", outline="white"
        )
        self.prediction_label.config(text="Canvas cleared. Draw again.")
        self.last_x, self.last_y = None, None
        # 전처리 이미지 표시 영역도 초기화
        self._update_processed_image_display(Image.new("L", (28, 28), "gray"))

    def predict_digit(self):
        if self.model is None:
            self.prediction_label.config(text="Model not loaded. Cannot predict.")
            return

        pil_for_preprocessing = self.pil_image.copy()

        try:
            # preprocess_drawn_image는 (텐서, 28x28 PIL 이미지)를 반환
            img_tensor, final_pil_for_display = preprocess_drawn_image(
                pil_for_preprocessing
            )
            img_tensor = img_tensor.to(self.device)

            # 전처리된 28x28 PIL 이미지를 GUI에 표시
            self._update_processed_image_display(final_pil_for_display)

        except Exception as e:
            self.prediction_label.config(text=f"Image processing error: {e}")
            print(f"Error during preprocessing: {e}")
            # 에러 발생 시 전처리 이미지 표시 영역도 초기화하거나 에러 메시지 이미지 표시
            self._update_processed_image_display(
                Image.new("L", (28, 28), "red")
            )  # 예시: 에러 시 빨간색
            return

        try:
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)

            result_text = f"Prediction: {predicted_class.item()} (Confidence: {confidence.item() * 100:.2f}%)"
            self.prediction_label.config(text=result_text)
            print(f"Prediction made: {result_text}")

        except Exception as e:
            self.prediction_label.config(text=f"Prediction error: {e}")
            print(f"Error during prediction: {e}")

        self.last_x, self.last_y = None, None


# --- 메인 실행 부분 ---
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Playground using device: {DEVICE}")

    loaded_model = None
    # model.py의 클래스 속성을 사용하여 경로 구성
    config_path = os.path.join(
        CHECKPOINT_DIRECTORY, NeuralNetwork.DEFAULT_CONFIG_FILENAME
    )
    weights_path = os.path.join(
        CHECKPOINT_DIRECTORY, NeuralNetwork.DEFAULT_WEIGHTS_FILENAME
    )

    try:
        loaded_model = NeuralNetwork.load_from_checkpoint(
            directory_path=CHECKPOINT_DIRECTORY, device=DEVICE
        )
        print("MNIST Model loaded successfully for playground.")
    except FileNotFoundError:
        print(f"Error: Checkpoint files not found in {CHECKPOINT_DIRECTORY}.")
        print(f"Please ensure '{config_path}' and '{weights_path}' exist there.")
        print("Run train.py first to create the checkpoint.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please ensure the model checkpoint at {CHECKPOINT_DIRECTORY} is valid.")

    if loaded_model is None:
        print("Exiting playground due to model loading failure.")
        try:
            temp_root = tk.Tk()
            temp_root.withdraw()
            messagebox.showerror(
                "Model Load Error",
                f"Failed to load model from '{CHECKPOINT_DIRECTORY}'.\nPlease run train.py first.\nExiting application.",
            )
            temp_root.destroy()
        except tk.TclError:
            pass
        exit()

    root = tk.Tk()
    app = DigitRecognizerApp(root, loaded_model, DEVICE)
    root.mainloop()
