import os
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import json


class DeadTreeDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms

        # Get all image files
        self.image_files = []
        for file in os.listdir(data_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Check if corresponding annotation file exists
                annotation_file = os.path.join(data_dir, file.rsplit('.', 1)[0] + '.txt')
                if os.path.exists(annotation_file):
                    self.image_files.append(file)

        print(f"Found {len(self.image_files)} images with annotations")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        # Load annotations
        annotation_file = os.path.join(self.data_dir,
                                       self.image_files[idx].rsplit('.', 1)[0] + '.txt')

        boxes = []
        labels = []

        img_width, img_height = image.size

        if os.path.exists(annotation_file) and os.path.getsize(annotation_file) > 0:
            with open(annotation_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        parts = line.split()
                        if len(parts) >= 5:
                            category = int(parts[0])
                            center_x = float(parts[1])
                            center_y = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])

                            # Convert YOLO format to absolute coordinates
                            abs_center_x = center_x * img_width
                            abs_center_y = center_y * img_height
                            abs_width = width * img_width
                            abs_height = height * img_height

                            # Convert to bounding box format (x1, y1, x2, y2)
                            x1 = abs_center_x - abs_width / 2
                            y1 = abs_center_y - abs_height / 2
                            x2 = abs_center_x + abs_width / 2
                            y2 = abs_center_y + abs_height / 2

                            # Ensure coordinates are within image bounds
                            x1 = max(0, min(x1, img_width - 1))
                            y1 = max(0, min(y1, img_height - 1))
                            x2 = max(x1 + 1, min(x2, img_width))
                            y2 = max(y1 + 1, min(y2, img_height))

                            boxes.append([x1, y1, x2, y2])
                            # For dead tree detection, we'll use category + 1 since 0 is background
                            labels.append(category + 1 if category > 0 else 1)

        # Convert to tensors
        if len(boxes) == 0:
            # If no annotations, create dummy box to avoid training issues
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        # Calculate area
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Assume all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([idx])

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            image = transforms.ToTensor()(image)

        return image, target


def get_model(num_classes):
    # Load a pre-trained model and replace the classifier
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    running_loss = 0.0

    for i, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

        if i % print_freq == 0:
            print(f"Epoch [{epoch}], Step [{i}/{len(data_loader)}], Loss: {losses.item():.4f}")

    return running_loss / len(data_loader)


def main():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN for dead tree detection')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing images and annotation files')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--save_model', type=str, default='dead_tree_model.pth',
                        help='Path to save the trained model')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes (including background)')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Create dataset and data loader
    dataset = DeadTreeDataset(args.data_dir)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    # Create model
    model = get_model(args.num_classes)
    model.to(device)

    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print(f"Starting training for {args.num_epochs} epochs...")

    # Training loop
    for epoch in range(args.num_epochs):
        avg_loss = train_one_epoch(model, optimizer, data_loader, device, epoch + 1)
        lr_scheduler.step()

        print(f"Epoch [{epoch + 1}/{args.num_epochs}] completed. Average Loss: {avg_loss:.4f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    torch.save(model.state_dict(), args.save_model)
    print(f"Training completed! Model saved as: {args.save_model}")

    # Save model info
    model_info = {
        'num_classes': args.num_classes,
        'architecture': 'fasterrcnn_resnet50_fpn',
        'epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr
    }

    with open(args.save_model.replace('.pth', '_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2)


if __name__ == '__main__':
    main()