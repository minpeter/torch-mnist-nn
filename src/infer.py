import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms  # 시각화 함수 내에서 사용

from model import NeuralNetwork
from utils import load_mnist_data, evaluate_model

# <<<< Inference Settings <<<<
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIRECTORY = "./data/my_mnist_checkpoint"
# >>>> Inference Settings >>>>

print(f"Using device: {DEVICE}")


# visualize_predictions 함수는 infer.py에 유지 (utils로 옮겨도 무방)
def visualize_predictions(model, device, vis_test_loader, num_images=5):
    model.eval()
    images_shown = 0
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
    mean, std = (0.1307,), (0.3081,)  # utils의 값과 일치해야 함

    # Normalize 할 때 사용한 평균과 표준편차를 역으로 적용해서 시각화
    # (단, 여기서는 간단히 0-1 범위로만 가정하고 시각화)
    # 입력 이미지는 이미 flatten 되었으므로 reshape 필요
    unnormalize = transforms.Normalize((-mean[0] / std[0],), (1.0 / std[0],))
    with torch.no_grad():
        for images, labels in vis_test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            for i in range(images.size(0)):
                if images_shown < num_images:
                    ax = axes[images_shown]
                    # 역정규화 및 채널, 높이, 너비 순서로 변경
                    img_display = images[i].cpu().view(1, 28, 28)
                    img_display = unnormalize(img_display)
                    img_display = img_display.squeeze().numpy()

                    ax.imshow(img_display, cmap="gray")
                    ax.set_title(
                        f"Pred: {predicted[i].item()}\nTrue: {labels[i].item()}"
                    )
                    ax.axis("off")
                    images_shown += 1
                else:
                    break
            if images_shown >= num_images:
                break
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        loaded_model = NeuralNetwork.load_from_checkpoint(
            directory_path=CHECKPOINT_DIRECTORY, device=DEVICE
        )
    except Exception as e:
        print(
            f"Failed to load model from directory {CHECKPOINT_DIRECTORY}: {e}. Make sure to run train.py first."
        )
        exit()

    _train_loader, test_loader = load_mnist_data()
    print("Test data loaded.")

    criterion = nn.CrossEntropyLoss()
    print("\nEvaluating loaded model on test set:")
    # utils에서 임포트한 evaluate_model 사용
    evaluate_model(
        loaded_model, DEVICE, test_loader, criterion, phase="Loaded Model Inference"
    )

    print("\nVisualizing some test predictions with the loaded model...")
    visualize_predictions(loaded_model, DEVICE, test_loader, num_images=5)
