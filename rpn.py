import torch
import torchvision
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator


def draw_rpn_anchors(model, image, level=0, max_anchors=2000):
    """
    Draw RPN anchors from a specific FPN level over an input image.

    Args:
        model: a torchvision FasterRCNN model (e.g., fasterrcnn_resnet50_fpn)
        image: a 3xHxW tensor (torch.Tensor) or PIL.Image
        level: which FPN level to visualize anchors from (0 = P2)
        max_anchors: limit number of anchors to draw for visibility
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Transform image
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image.transpose(2, 0, 1) / 255.0).float()
    elif not isinstance(image, torch.Tensor):
        image = torchvision.transforms.ToTensor()(image)

    image = image.to(device)
    images = [image]
    original_image_sizes = [tuple(image.shape[1:])]

    # Pass image through model's transform pipeline (adds padding, normalization)
    with torch.no_grad():
        transformed = model.transform(images)
        image_tensor = transformed.tensors  # (B, 3, H, W)
        image_shape = transformed.image_sizes[0]

        # Get FPN features
        features = model.backbone(image_tensor.tensors)[level]  # pick one level

        # Use the anchor generator directly
        anchor_generator = model.rpn.anchor_generator
        feature_maps = [features]
        anchors = anchor_generator(feature_maps, [image_shape])[0]  # list of boxes (Tensor [N, 4])

        anchors = anchors.cpu().numpy()
        H, W = image_tensor.shape[2:]

        # Plot image
        fig, ax = plt.subplots(1, figsize=(10, 10))
        img_np = image_tensor[0].cpu()
        img_np = to_pil_image(img_np)
        ax.imshow(img_np)

        # Draw anchors
        for box in anchors[:max_anchors]:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=0.4,
                                     edgecolor='red', facecolor='none', alpha=0.5)
            ax.add_patch(rect)

        ax.set_title(f"RPN Anchors (level {level}, showing up to {max_anchors})")
        plt.axis('off')
        plt.show()


if __name__ == "__main__()":

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    img = Image.open("your_image.jpg").convert("RGB")
    draw_rpn_anchors(model, img, level=0)  # visualize P2 anchors
