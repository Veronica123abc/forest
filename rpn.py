import torch
import torchvision
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection.rpn import AnchorGenerator

def draw_rpn_anchors(model, image, level=0, max_anchors=1000):
    """
    Visualize RPN anchors from a specific FPN level over an input image.

    Args:
        model: a torchvision FasterRCNN model (e.g., fasterrcnn_resnet50_fpn)
        image: a 3xHxW Tensor or a PIL.Image
        level: index of FPN level to visualize anchors from (0 = P2)
        max_anchors: max number of anchors to draw (for performance)
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Convert image to tensor if needed
    if not isinstance(image, torch.Tensor):
        image = torchvision.transforms.ToTensor()(image)
    image = image.to(device)

    images = [image]
    original_image_sizes = [tuple(image.shape[1:])]  # (H, W)

    # Use model's internal transform to get resized, normalized image
    with torch.no_grad():
        # model.transform returns an ImageList object, which is essentially a batch
        transformed = model.transform(images)
        image_tensor = transformed.tensors  # (B, 3, H, W)
        image_shape = transformed.image_sizes[0]

        # Get FPN features
        features_dict = model.backbone(image_tensor)
        features_list = list(features_dict.values())
        selected_feature = features_list[level:level+1]  # wrap in list for anchor_generator

        # Generate anchors
        anchor_generator = model.rpn.anchor_generator
        anchors = anchor_generator(selected_feature, [image_shape])[0]
        anchors = anchors.cpu().numpy()

        # Plot original image
        img_np = to_pil_image(image.cpu())
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img_np)

        # Draw anchors (up to max_anchors)
        for box in anchors[:max_anchors]:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=0.5,
                                     edgecolor='red', facecolor='none', alpha=0.3)
            ax.add_patch(rect)

        ax.set_title(f"RPN Anchors (FPN Level {level})")
        plt.axis('off')
        plt.show()
