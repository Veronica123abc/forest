import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import urllib

# Helper function to load the DeepLabV3+ model architecture with a custom checkpoint

def load_model(checkpoint_path, num_classes=21):
    from torchvision.models.segmentation import deeplabv3_resnet101
    model = deeplabv3_resnet101(weights="COCO_WITH_VOC_LABELS_V1") #pretrained=False, num_classes=num_classes)
    #state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    #model.load_state_dict(state_dict)
    model.eval()
    return model

# Preprocessing function matching Kaggle's notebook

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor, img

# Inference function

def run_inference(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)['out']
        pred_mask = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
    return pred_mask

# Main entry for inference

def main(image_path, checkpoint_path, output_mask_path):
    model = load_model(checkpoint_path)
    img_tensor, img_pil = preprocess_image(image_path)
    pred_mask = run_inference(model, img_tensor)
    # Convert prediction mask to an image
    pred_mask_img = Image.fromarray((pred_mask * 15).astype(np.uint8))
    pred_mask_img = pred_mask_img.resize(img_pil.size, resample=Image.NEAREST)  # Resize to original image size
    pred_mask_img.save(output_mask_path)
    print(f"Segmentation mask saved to {output_mask_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image (jpg/png)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--output', type=str, required=True, help='Path to save predicted mask image')
    args = parser.parse_args()
    main(args.image, args.checkpoint, args.output)
