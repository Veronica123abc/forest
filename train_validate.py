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
from collections import defaultdict
import matplotlib.pyplot as plt


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


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    if x2_min <= x1_max or y2_min <= y1_max:
        return 0.0

    intersection = (x2_min - x1_max) * (y2_min - y1_max)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def calculate_ap(gt_boxes, pred_boxes, pred_scores, iou_threshold=0.5):
    """Calculate Average Precision for a single class"""
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0
    if len(gt_boxes) == 0:
        return 0.0
    if len(pred_boxes) == 0:
        return 0.0

    # Sort predictions by confidence score (descending)
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]

    # Track which ground truth boxes have been matched
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)

    # Arrays to store precision and recall values
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))

    for i, pred_box in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1

        # Find best matching ground truth box
        for j, gt_box in enumerate(gt_boxes):
            if gt_matched[j]:
                continue
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        # Check if prediction is correct
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp[i] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[i] = 1

    # Calculate cumulative precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recall = tp_cumsum / len(gt_boxes)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

    # Calculate AP using the 11-point interpolation method
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0

    return ap


def evaluate_model(model, data_loader, device, num_classes):
    """Evaluate model and calculate mAP metrics"""
    model.eval()

    # Collect all predictions and ground truths
    all_predictions = defaultdict(list)
    all_ground_truths = defaultdict(list)

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get predictions
            predictions = model(images)

            # Process each image in the batch
            for i, (pred, target) in enumerate(zip(predictions, targets)):
                # Ground truth
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()

                # Predictions
                pred_boxes = pred['boxes'].cpu().numpy()
                pred_scores = pred['scores'].cpu().numpy()
                pred_labels = pred['labels'].cpu().numpy()

                # Group by class
                for class_id in range(1, num_classes):  # Skip background class
                    # Ground truth for this class
                    gt_mask = gt_labels == class_id
                    class_gt_boxes = gt_boxes[gt_mask]

                    # Predictions for this class
                    pred_mask = pred_labels == class_id
                    class_pred_boxes = pred_boxes[pred_mask]
                    class_pred_scores = pred_scores[pred_mask]

                    all_ground_truths[class_id].append(class_gt_boxes)
                    all_predictions[class_id].append({
                        'boxes': class_pred_boxes,
                        'scores': class_pred_scores
                    })

    # Calculate AP for each class
    ap_results = {}
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    for class_id in range(1, num_classes):
        class_aps = []

        for iou_threshold in iou_thresholds:
            # Combine all ground truths and predictions for this class
            all_gt_boxes = []
            all_pred_boxes = []
            all_pred_scores = []

            for gt_boxes_list, pred_dict in zip(all_ground_truths[class_id], all_predictions[class_id]):
                if len(gt_boxes_list) > 0:
                    all_gt_boxes.extend(gt_boxes_list)
                if len(pred_dict['boxes']) > 0:
                    all_pred_boxes.extend(pred_dict['boxes'])
                    all_pred_scores.extend(pred_dict['scores'])

            if len(all_gt_boxes) > 0 or len(all_pred_boxes) > 0:
                all_gt_boxes = np.array(all_gt_boxes) if all_gt_boxes else np.array([]).reshape(0, 4)
                all_pred_boxes = np.array(all_pred_boxes) if all_pred_boxes else np.array([]).reshape(0, 4)
                all_pred_scores = np.array(all_pred_scores) if all_pred_scores else np.array([])

                ap = calculate_ap(all_gt_boxes, all_pred_boxes, all_pred_scores, iou_threshold)
                class_aps.append(ap)
            else:
                class_aps.append(0.0)

        ap_results[class_id] = {
            'AP@0.5': class_aps[0],
            'AP@0.75': class_aps[5],
            'AP@0.5:0.95': np.mean(class_aps)
        }

    # Calculate overall mAP
    overall_metrics = {
        'mAP@0.5': np.mean([ap_results[cid]['AP@0.5'] for cid in ap_results]),
        'mAP@0.75': np.mean([ap_results[cid]['AP@0.75'] for cid in ap_results]),
        'mAP@0.5:0.95': np.mean([ap_results[cid]['AP@0.5:0.95'] for cid in ap_results])
    }

    return overall_metrics, ap_results


def collate_fn(batch):
    return tuple(zip(*batch))
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
        'val_mAP_50': [],
        'val_mAP_75': [],
        'val_mAP_50_95': []
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
        val_metrics, class_metrics = evaluate_model(model, val_loader, device, args.num_classes)

        # Print results
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Validation mAP@0.5: {val_metrics['mAP@0.5']:.4f}")
        print(f"  Validation mAP@0.75: {val_metrics['mAP@0.75']:.4f}")
        print(f"  Validation mAP@0.5:0.95: {val_metrics['mAP@0.5:0.95']:.4f}")

        # Print per-class metrics
        for class_id, metrics in class_metrics.items():
            print(f"  Class {class_id} - AP@0.5: {metrics['AP@0.5']:.4f}, "
                  f"AP@0.75: {metrics['AP@0.75']:.4f}, "
                  f"AP@0.5:0.95: {metrics['AP@0.5:0.95']:.4f}")

        # Save training history
        training_history['epochs'].append(epoch + 1)
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_mAP_50'].append(val_metrics['mAP@0.5'])
        training_history['val_mAP_75'].append(val_metrics['mAP@0.75'])
        training_history['val_mAP_50_95'].append(val_metrics['mAP@0.5:0.95'])

        # Save best model
        current_mAP = val_metrics['mAP@0.5:0.95']
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
        plt.figure(figsize=(15, 5))

        # Plot 1: Training Loss
        plt.subplot(1, 3, 1)
        plt.plot(training_history['epochs'], training_history['train_loss'], 'b-', label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)

        # Plot 2: Validation mAP@0.5
        plt.subplot(1, 3, 2)
        plt.plot(training_history['epochs'], training_history['val_mAP_50'], 'r-', label='mAP@0.5')
        plt.xlabel('Epoch')
        plt.ylabel('mAP@0.5')
        plt.title('Validation mAP@0.5')
        plt.legend()
        plt.grid(True)

        # Plot 3: Validation mAP@0.5:0.95
        plt.subplot(1, 3, 3)
        plt.plot(training_history['epochs'], training_history['val_mAP_50_95'], 'g-', label='mAP@0.5:0.95')
        plt.xlabel('Epoch')
        plt.ylabel('mAP@0.5:0.95')
        plt.title('Validation mAP@0.5:0.95')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(args.save_model.replace('.pth', '_training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved as: {args.save_model.replace('.pth', '_training_curves.png')}")

    except Exception as e:
        print(f"Could not generate training plots: {e}")


if __name__ == '__main__':
    main()