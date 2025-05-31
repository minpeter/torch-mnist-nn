import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from torchvision import transforms


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


def apply_transform_fn(examples):
    # 'image' 필드에 transform 적용
    # Hugging Face dataset의 이미지는 PIL Image 객체로 로드됨
    examples["pixel_values"] = [
        transform(image.convert("L")) for image in examples["image"]
    ]
    # label은 그대로 사용
    return examples


# PyTorch DataLoader 생성
# HuggingFace Dataset은 map-style dataset이므로, column을 지정해줘야 함
def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return pixel_values, labels


def load_multiple_datasets(
    dataset_ids: list[str],
    batch_size_param: int = 10,
    train_split_ratio: float = 0.8,
    random_seed: int = 42,
):
    """
    여러 데이터셋 ID를 입력받아 로드하고, 필요시 분할하며, 전처리하여
    통합된 DataLoader를 반환합니다.

    Args:
        dataset_ids (list[str]): 로드할 Hugging Face 데이터셋 ID의 리스트.
        batch_size_param (int): DataLoader의 배치 크기.
        train_split_ratio (float): 'train' 스플릿만 있을 경우, 학습 데이터로 사용할 비율.
                                   (1.0 - train_split_ratio)가 테스트 데이터 비율이 됩니다.
        random_seed (int): 데이터셋 분할 시 사용할 랜덤 시드.

    Returns:
        tuple: (train_loader, test_loader)
               오류 발생 또는 데이터셋을 찾을 수 없는 경우 (None, None)을 반환할 수 있습니다.
    """
    all_train_datasets = []
    all_test_datasets = []

    print(f"Attempting to load datasets: {dataset_ids}")

    for dataset_id in dataset_ids:
        print(f"\nProcessing dataset: {dataset_id}")
        try:
            # 데이터셋 로드 시도
            # 일부 데이터셋은 특정 'subset' (config name)이 필요할 수 있습니다.
            # 예: 'beans' 데이터셋은 기본 config가 로드됨.
            # 만약 특정 subset을 로드해야 한다면 dataset_id를 "dataset_name/subset_name" 형식으로 사용해야 합니다.
            # 여기서는 일반적인 경우를 가정합니다.
            dataset = load_dataset(
                dataset_id, trust_remote_code=True
            )  # trust_remote_code=True might be needed for some datasets
        except Exception as e:
            print(f"Failed to load dataset {dataset_id}. Error: {e}. Skipping.")
            continue

        current_train_ds = None
        current_test_ds = None

        if "test" in dataset:
            print(f"Found 'train' and 'test' splits for {dataset_id}.")
            current_train_ds = dataset["train"]
            current_test_ds = dataset["test"]
        elif "train" in dataset:
            print(
                f"Found only 'train' split for {dataset_id}. Splitting into train/test ({train_split_ratio * 100}% / {(1 - train_split_ratio) * 100}%)."
            )
            # Hugging Face datasets 라이브러리의 train_test_split 사용
            if len(dataset["train"]) == 0:
                print(f"Train split for {dataset_id} is empty. Skipping.")
                continue
            if len(dataset["train"]) < 2:  # train_test_split은 최소 2개의 샘플이 필요
                print(
                    f"Train split for {dataset_id} has less than 2 samples. Using all for training, no test split from this dataset."
                )
                current_train_ds = dataset["train"]
                # current_test_ds는 None으로 유지
            else:
                try:
                    split_result = dataset["train"].train_test_split(
                        test_size=(1.0 - train_split_ratio),
                        shuffle=True,
                        seed=random_seed,
                    )
                    current_train_ds = split_result["train"]
                    current_test_ds = split_result["test"]
                except Exception as e:
                    print(
                        f"Could not split dataset {dataset_id}. Error: {e}. Using entire set for training for this dataset."
                    )
                    current_train_ds = dataset[
                        "train"
                    ]  # 오류 시 전체를 학습용으로 사용

        else:
            print(
                f"Dataset {dataset_id} does not have a 'train' or 'test' split. Skipping."
            )
            continue

        # 각 데이터셋 부분에 전처리 적용
        # with_transform은 새로운 컬럼('pixel_values')을 추가하고, collate_fn에서 이를 사용합니다.
        # 원본 'image' 컬럼은 DataLoader에서 사용하지 않으면 무시됩니다.
        # 명시적으로 제거하고 싶다면 .map(apply_transform_fn, batched=True, remove_columns=['image']) 를 사용할 수 있습니다.
        if current_train_ds and len(current_train_ds) > 0:
            all_train_datasets.append(
                current_train_ds.with_transform(apply_transform_fn)
            )
            print(
                f"Added {len(current_train_ds)} samples from {dataset_id} to training set."
            )
        if current_test_ds and len(current_test_ds) > 0:
            all_test_datasets.append(current_test_ds.with_transform(apply_transform_fn))
            print(
                f"Added {len(current_test_ds)} samples from {dataset_id} to test set."
            )

    if not all_train_datasets:
        print(
            "No training data could be loaded or processed. Returning None for DataLoaders."
        )
        return None, None

    # 여러 데이터셋이 로드된 경우 하나로 합침
    final_train_dataset = (
        concatenate_datasets(all_train_datasets) if all_train_datasets else None
    )
    final_test_dataset = (
        concatenate_datasets(all_test_datasets) if all_test_datasets else None
    )

    print(
        f"\nTotal training samples: {len(final_train_dataset) if final_train_dataset else 0}"
    )
    print(
        f"Total testing samples: {len(final_test_dataset) if final_test_dataset else 0}"
    )

    train_loader = None
    if final_train_dataset and len(final_train_dataset) > 0:
        train_loader = DataLoader(
            final_train_dataset,
            batch_size=batch_size_param,
            shuffle=True,
            collate_fn=collate_fn,
        )
    else:
        print("Warning: Final training dataset is empty or None.")

    test_loader = None
    if final_test_dataset and len(final_test_dataset) > 0:
        test_loader = DataLoader(
            final_test_dataset,
            batch_size=batch_size_param,
            shuffle=False,
            collate_fn=collate_fn,
        )
    else:
        print("Warning: Final test dataset is empty or None. Test loader will be None.")
        # 필요하다면 빈 DataLoader 대신 None을 반환하는 것이 더 명확할 수 있습니다.
        # evaluate_model 함수에서 test_loader_param이 None일 경우를 처리하도록 수정해야 할 수 있습니다.

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
    if test_loader_param is None:  # test_loader가 None일 경우 평가를 건너뜀
        print(f"{phase} skipped: No test data available.")
        return 0.0  # 또는 적절한 기본값

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

    if (
        total == 0
    ):  # 테스트 데이터가 있었으나 루프를 돌지 않은 경우 (예: 배치가 비어있음)
        print(
            f"{phase} Test Results: No data to evaluate. Accuracy: 0/0 (0.00%), Avg Loss: 0.0000"
        )
        return 0.0

    accuracy = 100 * correct / total
    avg_test_loss = test_loss / len(
        test_loader_param
    )  # len(test_loader_param)은 배치의 수

    log_prefix = f"Epoch {epoch_num} " if epoch_num is not None else ""
    print(
        f"{log_prefix}({phase}) Test Results: Accuracy: {correct}/{total} ({accuracy:.2f}%), Avg Loss: {avg_test_loss:.4f}"
    )

    return accuracy
