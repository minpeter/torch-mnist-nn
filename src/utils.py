import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms


def load_mnist_data(batch_size_param=10):
    """MNIST 데이터셋을 로드하고 전처리하여 DataLoader를 반환합니다."""
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
        train_dataset, batch_size=batch_size_param, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size_param, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, test_loader


def evaluate_model(
    model,
    device_param,
    test_loader_param,
    criterion_param,
    epoch_num=None,
    phase="Evaluation",
):
    """주어진 모델을 테스트 데이터로 평가하고 정확도와 손실을 출력합니다."""
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():  # 기울기 계산 비활성화
        for images, labels in test_loader_param:
            images = images.to(device_param)
            labels = labels.to(device_param)
            outputs = model(images)
            loss = criterion_param(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(
                outputs.data, 1
            )  # 가장 높은 값을 가진 인덱스(예측된 숫자) 반환
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader_param)

    log_prefix = f"Epoch {epoch_num} " if epoch_num is not None else ""
    print(
        f"{log_prefix}({phase}) Test Results: Accuracy: {correct}/{total} ({accuracy:.2f}%), Avg Loss: {avg_test_loss:.4f}"
    )

    return accuracy
