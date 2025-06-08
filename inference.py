import os
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
import json
import cv2


def get_model(num_classes):
    """Load the model architecture"""
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_model(model_path, num_classes, device):
    """Load trained model weights"""
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    """Load and preprocess image for inference"""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image), image


def draw_predictions(image, predictions, confidence_threshold=0.5, class_names=None):
    """Draw bounding boxes and labels on image"""
    draw = ImageDraw.Draw(image)

    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()

    # Filter predictions by confidence threshold
    high_conf_indices = scores >= confidence_threshold
    boxes = boxes[high_conf_indices]
    scores = scores[high_conf_indices]
    labels = labels[high_conf_indices]

    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box
        color = colors[label % len(colors)]

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Prepare label text
        if class_names and label < len(class_names):
            class_name = class_names[label]
        else:
            class_name = f"Class_{label}"

        label_text = f"{class_name}: {score:.2f}"

        # Get text bounding box for background
        bbox = draw.textbbox((x1, y1), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Draw background rectangle for text
        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1],
                       fill=color, outline=color)

        # Draw text
        draw.text((x1 + 2, y1 - text_height - 2), label_text,
                  fill='white', font=font)

    return image


def run_inference(model, image_tensor, device):
    """Run inference on a single image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        predictions = model([image_tensor])
    return predictions[0]


def process_single_image(model, image_path, device, confidence_threshold=0.5,
                         class_names=None, save_path=None):
    """Process a single image and return results"""
    # Load and preprocess image
    image_tensor, original_image = preprocess_image(image_path)

    # Run inference
    predictions = run_inference(model, image_tensor, device)

    # Filter predictions by confidence
    high_conf_mask = predictions['scores'] >= confidence_threshold
    filtered_predictions = {
        'boxes': predictions['boxes'][high_conf_mask],
        'scores': predictions['scores'][high_conf_mask],
        'labels': predictions['labels'][high_conf_mask]
    }

    # Draw predictions on image
    result_image = draw_predictions(original_image.copy(), filtered_predictions,
                                    confidence_threshold, class_names)

    # Save result if path provided
    if save_path:
        result_image.save(save_path)
        print(f"Result saved to: {save_path}")

    # Print detection results
    boxes = filtered_predictions['boxes'].cpu().numpy()
    scores = filtered_predictions['scores'].cpu().numpy()
    labels = filtered_predictions['labels'].cpu().numpy()

    print(f"\nDetected {len(boxes)} objects in {os.path.basename(image_path)}:")
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        class_name = class_names[label] if class_names and label < len(class_names) else f"Class_{label}"
        print(f"  {i + 1}. {class_name}: {score:.3f} at [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")

    return result_image, filtered_predictions


def process_directory(model, input_dir, output_dir, device, confidence_threshold=0.5,
                      class_names=None):
    """Process all images in a directory"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(image_extensions)]

    print(f"Processing {len(image_files)} images...")

    total_detections = 0
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, f"result_{image_file}")

        try:
            result_image, predictions = process_single_image(
                model, image_path, device, confidence_threshold,
                class_names, output_path)
            total_detections += len(predictions['boxes'])
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    print(f"\nProcessing complete! Total detections: {total_detections}")


def main():
    parser = argparse.ArgumentParser(description='Run inference with trained Faster R-CNN model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model (.pth file)')
    parser.add_argument('--image_path', type=str,
                        help='Path to input image')
    parser.add_argument('--input_dir', type=str,
                        help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save results')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Confidence threshold for detections')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes (including background)')
    parser.add_argument('--class_names', type=str, nargs='+',
                        default=['background', 'dead_tree'],
                        help='Names of classes')

    args = parser.parse_args()

    # Validate arguments
    if not args.image_path and not args.input_dir:
        parser.error("Either --image_path or --input_dir must be provided")

    if not os.path.exists(args.model_path):
        parser.error(f"Model file not found: {args.model_path}")

    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Try to load model info if available
    model_info_path = args.model_path.replace('.pth', '_info.json')
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
            print(f"Model info loaded: {model_info}")
            if 'num_classes' in model_info:
                args.num_classes = model_info['num_classes']

    # Load model
    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path, args.num_classes, device)
    print("Model loaded successfully!")

    # Process image(s)
    if args.image_path:
        # Process single image
        if not os.path.exists(args.image_path):
            print(f"Error: Image file not found: {args.image_path}")
            return

        output_path = os.path.join(args.output_dir,
                                   f"result_{os.path.basename(args.image_path)}")
        os.makedirs(args.output_dir, exist_ok=True)

        result_image, predictions = process_single_image(
            model, args.image_path, device, args.confidence_threshold,
            args.class_names, output_path)

        print(f"\nInference completed for {args.image_path}")

    elif args.input_dir:
        # Process directory of images
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory not found: {args.input_dir}")
            return

        process_directory(model, args.input_dir, args.output_dir, device,
                          args.confidence_threshold, args.class_names)


if __name__ == '__main__':
    main()