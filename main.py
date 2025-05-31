import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 설정 및 하이퍼파라미터 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

INPUT_SIZE = 28 * 28  # MNIST 이미지 크기 (28x28 = 784 픽셀)
HIDDEN_SIZE = 100  # 은닉층의 뉴런 수 (책에서 30 또는 100 사용)
OUTPUT_SIZE = 10  # 출력층의 뉴런 수 (0~9 숫자 클래스)
LEARNING_RATE = 3.0  # 학습률 (책에서 eta=3.0 사용)
BATCH_SIZE = 10  # 미니배치 크기 (책에서 10 사용)
EPOCHS = 10  # 학습 에포크 수 (책에서는 30 에포크, 여기서는 시간 단축 위해 줄임)


# --- 2. MNIST 데이터셋 로드 및 전처리 ---
def load_mnist_data():
    # Hugging Face datasets 라이브러리에서 MNIST 로드
    dataset = load_dataset("minpeter/mnist")

    # 이미지 전처리: PIL Image -> Tensor, Normalize, Flatten
    # 책에서는 0.0 (흰색) ~ 1.0 (검은색) 스케일 사용.
    # ToTensor()는 [0, 255] PIL 이미지를 [0.0, 1.0] Tensor로 변환
    # Normalize는 평균 0.1307, 표준편차 0.3081로 정규화 (MNIST 데이터셋의 통계치)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,)
            ),  # 괄호 안의 쉼표는 단일 채널임을 명시
            transforms.Lambda(lambda x: x.view(-1)),  # 이미지를 1차원 벡터로 펼침 (784)
        ]
    )

    def apply_transform(examples):
        # 'image' 필드에 transform 적용
        # Hugging Face dataset의 이미지는 PIL Image 객체로 로드됨
        examples["pixel_values"] = [
            transform(image.convert("L")) for image in examples["image"]
        ]
        # label은 그대로 사용
        return examples

    # 데이터셋에 전처리 적용
    # remove_columns=['image']는 변환된 'pixel_values'를 사용하고 원본 'image'는 제거하기 위함
    processed_dataset = dataset.with_transform(apply_transform)

    train_dataset = processed_dataset["train"]
    test_dataset = processed_dataset["test"]

    # PyTorch DataLoader 생성
    # HuggingFace Dataset은 map-style dataset이므로, column을 지정해줘야 함
    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch])
        return pixel_values, labels

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, test_loader


# --- 3. 신경망 모델 정의 ---
# 책에서 설명한 시그모이드 뉴런을 사용한 간단한 3계층 신경망
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()  # 시그모이드 활성화 함수
        self.layer2 = nn.Linear(hidden_size, output_size)
        # CrossEntropyLoss는 내부적으로 Softmax를 포함하므로 출력층에 Softmax 불필요

        # 책의 설명대로 가중치/편향 초기화 (선택 사항, PyTorch 기본 초기화도 좋음)
        # 여기서는 PyTorch 기본 초기화를 사용하겠습니다.
        # 필요하다면 아래와 같이 초기화 가능:
        # nn.init.normal_(self.layer1.weight, mean=0, std=1.0/np.sqrt(input_size)) # 예시
        # nn.init.normal_(self.layer1.bias, mean=0, std=1.0) # 예시

    def forward(self, x):
        out = self.layer1(x)
        out = self.sigmoid(out)
        out = self.layer2(out)  # 로짓(raw scores) 반환
        return out


# --- 4. 학습 함수 ---
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()  # 모델을 학습 모드로 설정
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

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
        print(f"Epoch {epoch + 1} Complete. Average Loss: {avg_epoch_loss:.4f}")
        # 매 에포크마다 테스트 (책에서처럼)
        evaluate_model(model, test_loader, criterion, epoch_num=epoch + 1)


# --- 5. 평가 함수 ---
def evaluate_model(model, test_loader, criterion, epoch_num=None):
    model.eval()  # 모델을 평가 모드로 설정
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():  # 기울기 계산 비활성화
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(
                outputs.data, 1
            )  # 가장 높은 값을 가진 인덱스(예측된 숫자) 반환
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader)

    if epoch_num is not None:
        print(
            f"Epoch {epoch_num} Test Results: Accuracy: {correct}/{total} ({accuracy:.2f}%), Avg Loss: {avg_test_loss:.4f}"
        )
    else:
        print(f"Final Test Accuracy: {correct}/{total} ({accuracy:.2f}%)")
        print(f"Final Average Test Loss: {avg_test_loss:.4f}")
    return accuracy


# --- 6. 메인 실행 부분 ---
if __name__ == "__main__":
    # 데이터 로드
    train_loader, test_loader = load_mnist_data()
    print("MNIST data loaded.")

    # 모델, 손실 함수, 옵티마이저 초기화
    model = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()  # 다중 클래스 분류에 적합 (Softmax + NLLLoss)

    # 책에서 Stochastic Gradient Descent (SGD)를 사용했으므로 여기서도 SGD 사용
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    print("Model, Criterion, Optimizer initialized.")
    print("Starting training...")

    # 모델 학습
    train_model(model, train_loader, criterion, optimizer, EPOCHS)

    print("Training finished.")

    # 최종 모델 평가
    print("\nEvaluating final model on test set:")
    evaluate_model(model, test_loader, criterion)

    # (선택 사항) 일부 예측 결과 시각화
    print("\nVisualizing some test predictions...")
    model.eval()
    with torch.no_grad():
        dataiter = iter(test_loader)
        try:
            images, labels = next(dataiter)
        except StopIteration:  # 데이터로더가 비었을 경우
            print("Test loader is empty, cannot visualize.")
        else:
            images_to_show = images[:5].to(DEVICE)
            labels_to_show = labels[:5].to(DEVICE)

            outputs = model(images_to_show)
            _, predicted = torch.max(outputs.data, 1)

            fig, axes = plt.subplots(1, 5, figsize=(12, 3))
            # Normalize 할 때 사용한 평균과 표준편차를 역으로 적용해서 시각화
            # (단, 여기서는 간단히 0-1 범위로만 가정하고 시각화)
            # 입력 이미지는 이미 flatten 되었으므로 reshape 필요
            unnormalize = transforms.Normalize((-0.1307 / 0.3081,), (1.0 / 0.3081,))

            for i in range(5):
                ax = axes[i]
                # 역정규화 및 채널, 높이, 너비 순서로 변경
                img_display = unnormalize(images_to_show[i].cpu().view(1, 28, 28))
                img_display = img_display.squeeze().numpy()  # (28,28)

                ax.imshow(img_display, cmap="gray")
                ax.set_title(
                    f"Pred: {predicted[i].item()}\nTrue: {labels_to_show[i].item()}"
                )
                ax.axis("off")
            plt.tight_layout()
            plt.show()
