import tkinter as tk
from tkinter import Canvas, Button, Label, Frame
from PIL import Image, ImageDraw, ImageOps
import torch
import torchvision.transforms as T

from model import NeuralNetwork  # model.py에서 NeuralNetwork 클래스 임포트
import os

# <<<< Settings <<<<
CHECKPOINT_DIRECTORY = "./my_mnist_checkpoint"

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

CANVAS_SIZE = 280
PEN_THICKNESS = int(
    (CANVAS_SIZE // 28) * 1.5
)  # 펜 두께를 약간 더 두껍게 조정해볼 수 있습니다.
# >>>> Settings >>>>


# --- Image Preprocessing Function ---
def preprocess_drawn_image(pil_image):
    """Preprocesses the drawn PIL Image to match model input."""
    # 1. Convert to Grayscale (already 'L' mode but explicit)
    img = pil_image.convert("L")

    # 2. Invert colors (white bg, black drawing -> black bg, white drawing like MNIST)
    img = ImageOps.invert(img)

    # 3. Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # --- Debug: Save preprocessed image before tensor conversion ---
    # try:
    #     img.save("debug_preprocessed_pil.png")
    #     print("Debug: Saved preprocessed PIL image as debug_preprocessed_pil.png")
    # except Exception as e:
    #     print(f"Debug: Error saving PIL image: {e}")
    # --- End Debug ---

    # 4. Convert to PyTorch Tensor (scales pixel values to [0.0, 1.0])
    transform_to_tensor = T.ToTensor()
    img_tensor = transform_to_tensor(img)

    # --- Debug: Print tensor shape and value range after ToTensor ---
    # print(f"Debug: Tensor shape after ToTensor: {img_tensor.shape}")
    # print(f"Debug: Tensor min/max after ToTensor: {img_tensor.min().item()}, {img_tensor.max().item()}")
    # --- End Debug ---

    # 5. Normalize (using mean and std from training)
    normalize_transform = T.Normalize(MNIST_MEAN, MNIST_STD)
    img_tensor = normalize_transform(img_tensor)

    # --- Debug: Print tensor shape and value range after Normalize ---
    # print(f"Debug: Tensor shape after Normalize: {img_tensor.shape}")
    # print(f"Debug: Tensor min/max after Normalize: {img_tensor.min().item()}, {img_tensor.max().item()}")
    # --- End Debug ---

    # 6. Flatten and add batch dimension for model input: [1, 28, 28] -> [1, 784]
    img_tensor = img_tensor.view(1, -1)  # (1, 784)

    # --- Debug: Print final tensor shape ---
    # print(f"Debug: Final tensor shape for model: {img_tensor.shape}")
    # --- End Debug ---

    return img_tensor


# --- GUI Application Class ---
class DigitRecognizerApp:
    def __init__(self, master_window, model_instance, device_instance):
        self.master = master_window
        self.model = model_instance
        self.device = device_instance
        if self.model:  # 모델이 성공적으로 로드된 경우에만 eval() 호출
            self.model.eval()

        self.master.title("MNIST Playground - Draw a Digit")

        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self.pil_draw = ImageDraw.Draw(self.pil_image)

        self.canvas = Canvas(
            self.master,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="white",
            cursor="cross",
        )
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint_on_canvas)
        self.canvas.bind(
            "<ButtonRelease-1>", self.reset_last_pos
        )  # __init__에서 한 번만 바인딩

        self.prediction_label = Label(
            self.master,
            text="Draw a digit and click 'Predict'.",  # 영어로 변경
            font=("Arial", 16),
        )
        self.prediction_label.grid(row=1, column=0, columnspan=2, pady=10)

        button_frame = Frame(self.master)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)

        self.predict_button = Button(
            button_frame, text="Predict", command=self.predict_digit, width=15
        )
        self.predict_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = Button(
            button_frame, text="Clear", command=self.clear_canvas, width=15
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.last_x, self.last_y = None, None

    def paint_on_canvas(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x,
                self.last_y,
                x,
                y,
                width=PEN_THICKNESS,  # 두께 조정 (기존 *2 제거 또는 값 자체를 키움)
                fill="black",
                capstyle=tk.ROUND,
                smooth=tk.TRUE,
            )
            self.pil_draw.line(
                [(self.last_x, self.last_y), (x, y)],
                fill="black",
                width=PEN_THICKNESS,  # Tkinter와 동일하게
            )
        self.last_x = x
        self.last_y = y

    def reset_last_pos(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        # PIL Image를 완전히 새로 만드는 대신, 기존 Draw 객체로 흰색 사각형을 그려 초기화
        self.pil_draw.rectangle(
            [0, 0, CANVAS_SIZE, CANVAS_SIZE], fill="white", outline="white"
        )
        self.prediction_label.config(text="Canvas cleared. Draw again.")  # 영어로 변경
        self.last_x, self.last_y = None, None

    def predict_digit(self):
        if self.model is None:
            self.prediction_label.config(text="Model not loaded. Cannot predict.")
            return

        if self.pil_image is None:  # 사실상 이 조건은 거의 발생 안 함
            self.prediction_label.config(
                text="Please draw a digit first."
            )  # 영어로 변경
            return

        # --- Debug: 현재 PIL 이미지를 예측 전에 저장 ---
        # try:
        #     self.pil_image.save("debug_drawn_image_before_predict.png")
        #     print("Debug: Saved drawn image before prediction.")
        # except Exception as e:
        #     print(f"Debug: Error saving drawn image: {e}")
        # --- End Debug ---

        try:
            img_tensor = preprocess_drawn_image(
                self.pil_image.copy()
            )  # 원본 수정을 피하기 위해 복사본 전달
            img_tensor = img_tensor.to(self.device)
        except Exception as e:
            self.prediction_label.config(
                text=f"Image processing error: {e}"
            )  # 영어로 변경
            print(f"Error during preprocessing: {e}")  # 콘솔에도 에러 출력
            return

        # --- Debug: 모델 입력 직전 텐서 확인 ---
        # print(f"Debug: Tensor to model input - shape: {img_tensor.shape}, device: {img_tensor.device}")
        # --- End Debug ---

        try:
            with torch.no_grad():
                outputs = self.model(img_tensor)
                # --- Debug: 모델 출력 확인 ---
                # print(f"Debug: Model outputs (logits): {outputs}")
                # --- End Debug ---
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)

            result_text = f"Prediction: {predicted_class.item()} (Confidence: {confidence.item() * 100:.2f}%)"  # 영어로 변경
            self.prediction_label.config(text=result_text)
            print(f"Prediction made: {result_text}")  # 콘솔에도 예측 결과 출력

        except Exception as e:
            self.prediction_label.config(text=f"Prediction error: {e}")
            print(f"Error during prediction: {e}")  # 콘솔에도 에러 출력

        self.last_x, self.last_y = None, None


# --- Main Execution ---
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Playground using device: {DEVICE}")

    loaded_model = None  # 초기화
    try:
        # model.py에 정의된 클래스 메소드를 사용하여 모델 로드
        # 경로는 NeuralNetwork 클래스 내부의 기본 파일명을 사용하도록 함
        loaded_model = NeuralNetwork.load_from_checkpoint(
            directory_path=CHECKPOINT_DIRECTORY, device=DEVICE
        )
        print("MNIST Model loaded successfully for playground.")
    except FileNotFoundError:
        print(f"Error: Checkpoint files not found in {CHECKPOINT_DIRECTORY}.")
        print(
            f"Please ensure '{NeuralNetwork.DEFAULT_CONFIG_FILENAME}' and '{NeuralNetwork.DEFAULT_WEIGHTS_FILENAME}' exist there."
        )
        print("Run train.py first to create the checkpoint.")
    except Exception as e:  # 다른 로딩 에러 (예: 설정 파일 형식 오류 등)
        print(f"Error loading model: {e}")
        print(f"Please ensure the model checkpoint at {CHECKPOINT_DIRECTORY} is valid.")

    if loaded_model is None:
        print("Exiting playground due to model loading failure.")
        # GUI를 시작하지 않고 종료
        # Tkinter root 윈도우를 생성하기 전에 exit() 호출
        exit_root = tk.Tk()
        exit_root.withdraw()  # 메인 윈도우 숨기기
        tk.messagebox.showerror(
            "Model Load Error",
            f"Failed to load model from '{CHECKPOINT_DIRECTORY}'.\nPlease run train.py first.\nExiting application.",
        )
        exit_root.destroy()
        exit()

    root = tk.Tk()
    app = DigitRecognizerApp(root, loaded_model, DEVICE)
    root.mainloop()
