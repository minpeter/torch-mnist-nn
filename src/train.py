import torch
import torch.nn as nn
import torch.optim as optim

from model import NeuralNetwork
from utils import load_multiple_datasets, evaluate_model  # 수정된 임포트

# <<<< Setting & Hyperparameters <<<<
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = 28 * 28  # MNIST 이미지 크기 (28x28 = 784 픽셀)
HIDDEN_SIZE = 100  # 은닉층의 뉴런 수 (책에서 30 또는 100 사용)
OUTPUT_SIZE = 10  # 출력층의 뉴런 수 (0~9 숫자 클래스)
LEARNING_RATE = 3.0  # 학습률 (책에서 eta=3.0 사용)

BATCH_SIZE = 10

EPOCHS = 20  # 학습 에포크 수 (책에서는 30 에포크, 여기서는 시간 단축 위해 줄임)

CHECKPOINT_DIRECTORY = "./data/checkpoint"
# >>>> Setting & Hyperparameters >>>>

print(f"Using device: {DEVICE}")


def train_single_model(
    model, train_loader, test_loader, criterion, optimizer, epochs, device
):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 순전파
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 역전파 및 가중치 업데이트 (경사 하강법)
            optimizer.zero_grad()  # 이전 배치의 기울기 초기화
            loss.backward()  # 기울기 계산
            optimizer.step()  # 가중치 업데이트

            running_loss += loss.item()
            if (i + 1) % 1000 == 0:  # 1000 미니배치마다 로그 출력
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )
        avg_epoch_loss = running_loss / len(train_loader)
        print(
            f"Epoch {epoch + 1} Complete. Average Training Loss: {avg_epoch_loss:.4f}"
        )

        # 매 에포크마다 테스트
        evaluate_model(
            model,
            device,
            test_loader,
            criterion,
            epoch_num=epoch + 1,
            phase="During Training",
        )


if __name__ == "__main__":
    train_loader, test_loader = load_multiple_datasets(
        dataset_ids=["minpeter/mnist", "minpeter/mnist-user-input"],
        batch_size_param=BATCH_SIZE,
    )
    print("MNIST data loaded.")

    # 모델, 손실 함수, 옵티마이저 초기화
    model = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()  # 다중 클래스 분류에 적합 (Softmax + NLLLoss)

    # 책에서 Stochastic Gradient Descent (SGD)를 사용했으므로 여기서도 SGD 사용
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    print("Model, Criterion, Optimizer initialized.")
    print(
        f"Training with architecture: Input: {model.input_size}, Hidden: {model.hidden_size}, Output: {model.output_size}"
    )
    print("Starting training...")

    train_single_model(
        model, train_loader, test_loader, criterion, optimizer, EPOCHS, DEVICE
    )
    print("Training finished.")

    model.save_checkpoint(CHECKPOINT_DIRECTORY)
    print(
        f"\nModel checkpoint (config and weights) saved in directory: {CHECKPOINT_DIRECTORY}"
    )
    print("\nTo run inference, use infer.py")
