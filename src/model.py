import torch.nn as nn
import json
from safetensors.torch import save_file, load_file
import os


# 책에서 설명한 시그모이드 뉴런을 사용한 간단한 3계층 신경망
class NeuralNetwork(nn.Module):
    DEFAULT_WEIGHTS_FILENAME = "mnist_nn_model.safetensors"
    DEFAULT_CONFIG_FILENAME = "mnist_nn_config.json"

    def __init__(self, input_size=28 * 28, hidden_size=100, output_size=10):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.sigmoid = nn.Sigmoid()  # 시그모이드 활성화 함수
        self.layer2 = nn.Linear(self.hidden_size, self.output_size)
        # CrossEntropyLoss는 내부적으로 Softmax를 포함하므로 출력층에 Softmax 불필요

    def forward(self, x):
        out = self.layer1(x)
        out = self.sigmoid(out)
        out = self.layer2(out)  # 로짓(raw scores) 반환
        return out

    def save_checkpoint(self, directory_path):
        """모델 설정과 가중치를 지정된 디렉토리에 기본 파일명으로 저장합니다."""
        os.makedirs(directory_path, exist_ok=True)

        # 전체 파일 경로 생성
        config_path = os.path.join(directory_path, self.DEFAULT_CONFIG_FILENAME)
        weights_path = os.path.join(directory_path, self.DEFAULT_WEIGHTS_FILENAME)

        # 모델 설정 저장
        model_config_to_save = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "architecture": self.__class__.__name__,
        }
        with open(config_path, "w") as f:
            json.dump(model_config_to_save, f, indent=4)
        print(f"Model configuration saved to {config_path}")

        # 모델 가중치 저장
        save_file(self.state_dict(), weights_path)
        print(f"Model weights saved to {weights_path}")

    @classmethod
    def load_from_checkpoint(cls, directory_path, device):
        """지정된 디렉토리에서 기본 파일명을 사용하여 모델 설정과 가중치를 로드합니다."""
        # 전체 파일 경로 생성
        config_path = os.path.join(directory_path, cls.DEFAULT_CONFIG_FILENAME)
        weights_path = os.path.join(directory_path, cls.DEFAULT_WEIGHTS_FILENAME)

        # 1. 모델 설정 로드
        try:
            with open(config_path, "r") as f:
                model_config = json.load(f)
        except FileNotFoundError:
            print(f"Error: Model configuration file not found at {config_path}.")
            raise
        except Exception as e:
            print(f"Error loading model configuration from {config_path}: {e}")
            raise

        # 2. 로드된 설정을 사용하여 모델 인스턴스 생성
        model = cls(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"],
            output_size=model_config["output_size"],
        ).to(device)
        print(
            f"Model instance created on {device} with architecture from config: "
            f"input_size={model_config['input_size']}, "
            f"hidden_size={model_config['hidden_size']}, "
            f"output_size={model_config['output_size']}"
        )

        # 3. 모델 가중치 로드
        try:
            state_dict = load_file(weights_path, device=str(device))
            model.load_state_dict(state_dict)
            print(f"Model weights loaded from {weights_path}")
        except FileNotFoundError:
            print(f"Error: Model weights file not found at {weights_path}.")
            raise
        except Exception as e:
            print(f"Error loading model weights from {weights_path}: {e}")
            raise

        model.eval()
        return model
