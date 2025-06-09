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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Use proper evaluation metrics
try:
    from torchmetrics.detection import MeanAveragePrecision

    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics not available. Install with: pip install torchmetrics")
    print("Falling back to basic evaluation...")


class DeadTreeDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_files=None, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms

        # Use provided image files or discover them
        if image_files is not None:
            self.image_files = image_files
        else:
            # Get all image files
            self.image_files = []
            for file in os.listdir(data_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Check if corresponding annotation file exists
                    annotation_file = os.path.join(data_dir, file.rsplit('.', 1)[0] + '.txt')
                    if os.path.exists(annotation_file):
                        self.image_files.append(file)

        print(f"Dataset contains {len(self.image_files)} images with annotations")

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


def evaluate_model_with_torchmetrics(model, data_loader, device):
    """Evaluate model using torchmetrics (recommended)"""
    model.eval()

    # Initialize the metric
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=None,  # Use default IoU thresholds [0.5:0.05:0.95]
        rec_thresholds=None,  # Use default recall thresholds
        max_detection_thresholds=[1, 10, 100],
        class_metrics=True,  # Compute per-class metrics
        sync_on_compute=True
    )

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating with torchmetrics"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get predictions
            predictions = model(images)

            # Convert predictions and targets to CPU for metric calculation
            preds = []
            targs = []

            for pred, target in zip(predictions, targets):
                pred_dict = {
                    'boxes': pred['boxes'].cpu(),
                    'scores': pred['scores'].cpu(),
                    'labels': pred['labels'].cpu()
                }
                preds.append(pred_dict)

                target_dict = {
                    'boxes': target['boxes'].cpu(),
                    'labels': target['labels'].cpu()
                }
                targs.append(target_dict)

            # Update metric
            metric.update(preds, targs)

    # Compute final metrics
    result = metric.compute()

    # Extract key metrics
    metrics = {
        'mAP': result['map'].item(),  # mAP@0.5:0.95
        'mAP_50': result['map_50'].item(),  # mAP@0.5
        'mAP_75': result['map_75'].item(),  # mAP@0.75
        'mAP_small': result['map_small'].item(),  # mAP for small objects
        'mAP_medium': result['map_medium'].item(),  # mAP for medium objects
        'mAP_large': result['map_large'].item(),  # mAP for large objects
        'mAR_1': result['mar_1'].item(),  # mAR with max 1 detection per image
        'mAR_10': result['mar_10'].item(),  # mAR with max 10 detections per image
        'mAR_100': result['mar_100'].item(),  # mAR with max 100 detections per image
    }

    # Add per-class metrics if available
    if 'map_per_class' in result and result['map_per_class'] is not None:
        for i, class_map in enumerate(result['map_per_class']):
            if not torch.isnan(class_map):
                metrics[f'mAP_class_{i + 1}'] = class_map.item()

    return metrics, result


def evaluate_model_basic(model, data_loader, device):
    """Basic evaluation without torchmetrics (fallback)"""
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Basic evaluation"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get loss during evaluation (model returns loss dict when targets provided)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            total_loss += losses.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

    # Return basic metrics
    return {
        'avg_loss': avg_loss,
        'mAP': 0.0,  # Placeholder
        'mAP_50': 0.0,  # Placeholder
        'mAP_75': 0.0  # Placeholder
    }, {}


def evaluate_model(model, data_loader, device):
    """Main evaluation function that chooses the best available method"""
    if TORCHMETRICS_AVAILABLE:
        return evaluate_model_with_torchmetrics(model, data_loader, device)
    else:
        return evaluate_model_basic(model, data_loader, device)


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
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
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--val_batch_size', type=int, default=8,
                        help='Batch size for validation')

    args = parser.parse_args()

    # Check if torchmetrics is available and inform user
    if TORCHMETRICS_AVAILABLE:
        print("✓ Using torchmetrics for comprehensive evaluation metrics")
    else:
        print("⚠ torchmetrics not available. Using basic evaluation.")
        print("  Install torchmetrics for better metrics: pip install torchmetrics")

    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Get all image files and split into train/validation
    all_image_files = []
    for file in os.listdir(args.data_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            annotation_file = os.path.join(args.data_dir, file.rsplit('.', 1)[0] + '.txt')
            if os.path.exists(annotation_file):
                all_image_files.append(file)

    print(f"Found {len(all_image_files)} total images with annotations")

    # Split dataset
    if len(all_image_files) < 2:
        print("Error: Need at least 2 images for train/validation split")
        return

    train_files, val_files = train_test_split(
        all_image_files,
        test_size=args.val_split,
        random_state=42,
        shuffle=True
    )

    print(f"Training set: {len(train_files)} images")
    print(f"Validation set: {len(val_files)} images")

    # Create datasets
    train_dataset = DeadTreeDataset(args.data_dir, train_files)
    val_dataset = DeadTreeDataset(args.data_dir, val_files)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
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

    # Track training history
    training_history = {
        'epochs': [],
        'train_loss': [],
        'val_mAP': [],
        'val_mAP_50': [],
        'val_mAP_75': []
    }

    best_mAP = 0.0

    # Training loop
    for epoch in range(args.num_epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'=' * 50}")

        # Training
        avg_train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch + 1)
        lr_scheduler.step()

        # Validation
        print(f"\nRunning validation...")
        val_metrics, detailed_results = evaluate_model(model, val_loader, device)

        # Print results
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Training Loss: {avg_train_loss:.4f}")

        if TORCHMETRICS_AVAILABLE:
            print(f"  Validation mAP@0.5:0.95: {val_metrics['mAP']:.4f}")
            print(f"  Validation mAP@0.5: {val_metrics['mAP_50']:.4f}")
            print(f"  Validation mAP@0.75: {val_metrics['mAP_75']:.4f}")
            print(f"  Validation mAR@100: {val_metrics['mAR_100']:.4f}")

            # Print per-class metrics if available
            for key, value in val_metrics.items():
                if key.startswith('mAP_class_'):
                    class_id = key.split('_')[-1]
                    print(f"  Class {class_id} mAP@0.5:0.95: {value:.4f}")
        else:
            print(f"  Validation Loss: {val_metrics['avg_loss']:.4f}")

        # Save training history
        training_history['epochs'].append(epoch + 1)
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_mAP'].append(val_metrics.get('mAP', 0.0))
        training_history['val_mAP_50'].append(val_metrics.get('mAP_50', 0.0))
        training_history['val_mAP_75'].append(val_metrics.get('mAP_75', 0.0))

        # Save best model
        current_mAP = val_metrics.get('mAP', 0.0)
        if current_mAP > best_mAP:
            best_mAP = current_mAP
            best_model_path = args.save_model.replace('.pth', '_best.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved: {best_model_path} (mAP: {best_mAP:.4f})")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_metrics': val_metrics,
                'best_mAP': best_mAP,
                'training_history': training_history
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

    # Save final model
    torch.save(model.state_dict(), args.save_model)
    print(f"\nTraining completed! Final model saved as: {args.save_model}")
    print(f"Best validation mAP@0.5:0.95: {best_mAP:.4f}")

    # Save training results
    results = {
        'training_args': vars(args),
        'training_history': training_history,
        'best_mAP': best_mAP,
        'final_metrics': val_metrics,
        'torchmetrics_used': TORCHMETRICS_AVAILABLE,
        'model_info': {
            'num_classes': args.num_classes,
            'architecture': 'fasterrcnn_resnet50_fpn',
            'epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'validation_split': args.val_split
        }
    }

    with open(args.save_model.replace('.pth', '_training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot training curves
    try:
        plt.figure(figsize=(15, 10))

        # Plot 1: Training Loss
        plt.subplot(2, 3, 1)
        plt.plot(training_history['epochs'], training_history['train_loss'], 'b-', label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)

        # Plot 2: Validation mAP@0.5:0.95
        plt.subplot(2, 3, 2)
        plt.plot(training_history['epochs'], training_history['val_mAP'], 'r-', label='mAP@0.5:0.95')
        plt.xlabel('Epoch')
        plt.ylabel('mAP@0.5:0.95')
        plt.title('Validation mAP@0.5:0.95')
        plt.legend()
        plt.grid(True)

        # Plot 3: Validation mAP@0.5
        plt.subplot(2, 3, 3)
        plt.plot(training_history['epochs'], training_history['val_mAP_50'], 'g-', label='mAP@0.5')
        plt.xlabel('Epoch')
        plt.ylabel('mAP@0.5')
        plt.title('Validation mAP@0.5')
        plt.legend()
        plt.grid(True)

        # Plot 4: Validation mAP@0.75
        plt.subplot(2, 3, 4)
        plt.plot(training_history['epochs'], training_history['val_mAP_75'], 'm-', label='mAP@0.75')
        plt.xlabel('Epoch')
        plt.ylabel('mAP@0.75')
        plt.title('Validation mAP@0.75')
        plt.legend()
        plt.grid(True)

        # Plot 5: Combined mAP metrics
        plt.subplot(2, 3, 5)
        plt.plot(training_history['epochs'], training_history['val_mAP'], 'r-', label='mAP@0.5:0.95')
        plt.plot(training_history['epochs'], training_history['val_mAP_50'], 'g-', label='mAP@0.5')
        plt.plot(training_history['epochs'], training_history['val_mAP_75'], 'm-', label='mAP@0.75')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.title('All mAP Metrics')
        plt.legend()
        plt.grid(True)

        # Plot 6: Learning curve overview
        ax1 = plt.subplot(2, 3, 6)
        ax1.plot(training_history['epochs'], training_history['train_loss'], 'b-', label='Train Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(training_history['epochs'], training_history['val_mAP'], 'r-', label='Val mAP')
        ax2.set_ylabel('Validation mAP', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        plt.title('Training Overview')

        plt.tight_layout()
        plt.savefig(args.save_model.replace('.pth', '_training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved as: {args.save_model.replace('.pth', '_training_curves.png')}")

    except Exception as e:
        print(f"Could not generate training plots: {e}")


if __name__ == '__main__':
    main()